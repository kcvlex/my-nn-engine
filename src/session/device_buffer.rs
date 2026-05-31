use std::alloc::Layout;
use std::ffi::c_void;

#[cfg(feature = "cuda")]
mod ffi {
    use std::ffi::c_void;
    use std::os::raw::c_int;

    #[allow(non_camel_case_types)]
    pub type cudaError_t = c_int;
    #[allow(non_camel_case_types)]
    pub type cudaMemcpyKind = c_int;

    pub const CUDA_SUCCESS: cudaError_t = 0;
    pub const CUDA_MEMCPY_HOST_TO_DEVICE: cudaMemcpyKind = 1;

    #[link(name = "cudart", kind = "dylib")]
    extern "C" {
        pub fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> cudaError_t;
        pub fn cudaFree(dev_ptr: *mut c_void) -> cudaError_t;
        pub fn cudaMemset(dev_ptr: *mut c_void, value: c_int, count: usize) -> cudaError_t;
        pub fn cudaMemcpy(
            dst: *mut c_void,
            src: *const c_void,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t;
        pub fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> cudaError_t;
        pub fn cudaFreeHost(ptr: *mut c_void) -> cudaError_t;
    }
}

/// Page-locked (pinned) host buffer. The GPU can DMA from pinned memory
/// asynchronously, so `cudaMemcpyAsync` from it does not block the host on an
/// internal staging copy -- required for the copy stream to overlap compute.
pub struct PinnedHostBuffer {
    ptr: *mut c_void,
    size: usize,
}

unsafe impl Send for PinnedHostBuffer {}
unsafe impl Sync for PinnedHostBuffer {}

impl PinnedHostBuffer {
    #[cfg(feature = "cuda")]
    pub fn alloc(size: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { ffi::cudaHostAlloc(&mut ptr, size.max(1), 0) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError(err));
        }
        Ok(Self { ptr, size })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn alloc(_size: usize) -> Result<Self, CudaError> {
        Err(CudaError(-1))
    }

    pub fn ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    /// # Safety
    /// The buffer must not be aliased while the returned slice is held.
    pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u8, self.size) }
    }
}

impl Drop for PinnedHostBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if !self.ptr.is_null() {
            unsafe {
                ffi::cudaFreeHost(self.ptr);
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CudaError(pub i32);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error: {}", self.0)
    }
}

impl std::error::Error for CudaError {}

#[derive(Debug, Clone, Copy)]
enum BufferKind {
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    Cuda,
    Host,
}

const HOST_ALIGN: usize = 256;

pub struct DeviceBuffer {
    ptr: *mut c_void,
    size: usize,
    kind: BufferKind,
}

unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    #[cfg(feature = "cuda")]
    pub fn alloc_zeroed(size: usize) -> Result<Self, CudaError> {
        use ffi::*;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, size) };
        if err != CUDA_SUCCESS {
            return Err(CudaError(err));
        }
        let err = unsafe { cudaMemset(ptr, 0, size) };
        if err != CUDA_SUCCESS {
            unsafe {
                cudaFree(ptr);
            }
            return Err(CudaError(err));
        }
        Ok(Self {
            ptr,
            size,
            kind: BufferKind::Cuda,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn alloc_zeroed(_size: usize) -> Result<Self, CudaError> {
        Err(CudaError(-1))
    }

    pub fn alloc_zeroed_host(size: usize) -> Result<Self, CudaError> {
        let layout = host_layout(size);
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(CudaError(-2));
        }
        Ok(Self {
            ptr: ptr as *mut c_void,
            size,
            kind: BufferKind::Host,
        })
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }

    /// # Safety
    /// The caller must ensure that `src` points to a valid memory region of at least `len` bytes,
    /// and that the data is properly initialized.
    #[cfg(feature = "cuda")]
    pub unsafe fn host_to_device(&self, src: *const c_void, len: usize) -> Result<(), CudaError> {
        use ffi::*;
        let err = unsafe { cudaMemcpy(self.ptr, src, len, CUDA_MEMCPY_HOST_TO_DEVICE) };
        if err != CUDA_SUCCESS {
            return Err(CudaError(err));
        }
        Ok(())
    }

    /// # Safety
    /// The caller must ensure that `src` points to a valid memory region of at least `len` bytes,
    /// and that the data is properly initialized.
    #[cfg(not(feature = "cuda"))]
    pub unsafe fn host_to_device(&self, _src: *const c_void, _len: usize) -> Result<(), CudaError> {
        Err(CudaError(-1))
    }
}

fn host_layout(size: usize) -> Layout {
    Layout::from_size_align(size.max(1), HOST_ALIGN).expect("invalid host buffer layout")
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        match self.kind {
            BufferKind::Cuda => {
                #[cfg(feature = "cuda")]
                unsafe {
                    ffi::cudaFree(self.ptr);
                }
            }
            BufferKind::Host => unsafe {
                std::alloc::dealloc(self.ptr as *mut u8, host_layout(self.size));
            },
        }
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DeviceBuffer(ptr={:p}, size={}, kind={:?})",
            self.ptr, self.size, self.kind
        )
    }
}
