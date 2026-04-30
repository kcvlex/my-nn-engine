use std::ffi::c_void;

use cuda_runtime_sys::cudaError_t;
use cuda_runtime_sys::cudaFree;
use cuda_runtime_sys::cudaMalloc;
use cuda_runtime_sys::cudaMemset;

#[derive(Debug, Clone, Copy)]
pub struct CudaError(pub cudaError_t);

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cuda error: {:?}", self.0)
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
    pub fn alloc_zeroed(size: usize) -> Result<Self, CudaError> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, size) };
        if err != cudaError_t::cudaSuccess {
            return Err(CudaError(err));
        }
        let err = unsafe { cudaMemset(ptr, 0, size) };
        if err != cudaError_t::cudaSuccess {
            unsafe {
                cudaFree(ptr);
            }
            return Err(CudaError(err));
        }
        Ok(Self { ptr, size })
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                cudaFree(self.ptr);
            }
        }
    }
}

impl std::fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer(ptr={:p}, size={})", self.ptr, self.size)
    }
}
