use std::path::Path;

use inkwell::support::load_library_permanently;

use crate::codegen::cpu::blas::Backend;

impl Backend {
    pub fn shared_library_name(&self) -> &'static str {
        match self {
            Self::OpenBLAS => "libopenblas.so",
            Self::MKL => "libmkl_rt.so",
        }
    }
}

/// Try to load a BLAS shared lib (MKL preferred) and the matching OpenMP
/// runtime into the LLVM JIT global symbol space so kernel modules can resolve
/// `cblas_*` and `__kmpc_*` symbols. Returns the picked backend.
pub(super) fn load_jit_runtime() -> Result<Backend, String> {
    let backend = [Backend::MKL, Backend::OpenBLAS]
        .into_iter()
        .find(|b| load_library_permanently(Path::new(b.shared_library_name())).is_ok())
        .ok_or_else(|| "no BLAS shared library found".to_string())?;

    // MKL uses libiomp5 internally; share the same runtime to avoid contention
    // between two OpenMP runtimes. Otherwise fall back to LLVM's libomp.
    let omp_lib = if backend == Backend::MKL {
        "libiomp5.so"
    } else {
        "libomp.so"
    };
    load_library_permanently(Path::new(omp_lib))
        .map_err(|e| format!("Failed to load {omp_lib}: {e:?}"))?;
    Ok(backend)
}
