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

pub struct DeviceBuffer {
    ptr: *mut c_void,
    size: usize,
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
        Ok(Self { ptr, size })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn alloc_zeroed(_size: usize) -> Result<Self, CudaError> {
        Err(CudaError(-1))
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

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        if !self.ptr.is_null() {
            unsafe {
                ffi::cudaFree(self.ptr);
            }
        }
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer(ptr={:p}, size={})", self.ptr, self.size)
    }
}
