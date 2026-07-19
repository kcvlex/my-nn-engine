use std::collections::HashMap;
use std::collections::HashSet;
use std::io::BufWriter;
use std::io::ErrorKind;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;

use itertools::zip_eq;
use log::info;
use rayon::prelude::*;

use crate::codegen::cuda::*;
use crate::codegen::*;
use crate::options::Options;
use crate::options::Target;
use crate::schedule::ir::AllocPlace;
use crate::schedule::ir::Step;
use crate::schedule::Schedule;
use crate::session::send_initializer_to_device;
use crate::session::DeviceBuffer;
use crate::session::InitializerSource;
use crate::session::ModelLoadError;
use crate::session::PersistentBuffers;
use crate::session::PinnedHostBuffer;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

static CUDA_LOCK: Mutex<()> = Mutex::new(());

pub fn cuda_lock() -> MutexGuard<'static, ()> {
    CUDA_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn nvcc_invocation_error(e: std::io::Error) -> SessionError {
    if e.kind() == ErrorKind::NotFound {
        SessionError::OtherError(
            "nvcc not found in PATH. Install the CUDA Toolkit and ensure nvcc is on PATH."
                .to_string(),
        )
    } else {
        SessionError::OtherError(format!("nvcc invocation failed: {e}"))
    }
}

type InitType =
    unsafe extern "C" fn(*const *const u8, *const *mut std::ffi::c_void) -> *mut std::ffi::c_void;
type RunType = unsafe extern "C" fn(*mut std::ffi::c_void, *const *mut u8, *const *const u8);
type DestroyType = unsafe extern "C" fn(*mut std::ffi::c_void);

fn host_resident_initializer_indices(schedule: &Schedule) -> HashSet<usize> {
    // Only meaningful for the prefetch scheduler; otherwise initializers are resident in VRAM.
    let Target::CUDA(policy) = schedule.options.target else {
        return HashSet::new();
    };
    if policy.is_disabled() {
        return HashSet::new();
    }

    let Some(plan) = schedule.execution_plan.as_ref() else {
        return HashSet::new();
    };

    // Only treat an initializer as host-resident if the plan copies it into a GPU chunk.
    // This avoids misclassifying const outputs (Initializer -> Output transfers).
    let host_resident_values: HashSet<_> = plan
        .steps
        .iter()
        .filter_map(|s| match s {
            Step::Transfer(t)
                if t.context.device == crate::schedule::ir::Device::CUDA &&
                    matches!(t.dst.place, AllocPlace::Chunk(_)) =>
            {
                match t.src.place {
                    AllocPlace::Initializer(v) => Some(v),
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();
    schedule
        .initializers
        .iter()
        .enumerate()
        .filter(|(_, v)| host_resident_values.contains(v))
        .map(|(i, _)| i)
        .collect()
}

/// Host source for a `HostResident` weight, kept alive for the session and used
/// as the H2D source. External (file-backed) weights are read into a pinned host
/// buffer so the copy stream's `cudaMemcpyAsync` is truly async and overlaps
/// compute. Small inline weights stay owned (pageable; negligible).
enum HostWeight {
    Owned(StrictTensor),
    Pinned(PinnedHostBuffer),
}

impl HostWeight {
    fn as_ptr(&self) -> *const u8 {
        match self {
            HostWeight::Owned(t) => t.as_ptr(),
            HostWeight::Pinned(b) => b.ptr(),
        }
    }
}

pub struct SessionCUDA {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer_buffers: Vec<Arc<DeviceBuffer>>,
    host_resident_weights: HashMap<usize, HostWeight>,
    session_state_buffers: Vec<Arc<DeviceBuffer>>,

    #[allow(dead_code)]
    lib: libloading::Library,
    init_func: InitType,
    run_func: RunType,
    destroy_func: DestroyType,
    state: *mut std::ffi::c_void,
}

/// Compile the schedule's CUDA codegen output into a shared library and return
/// its path. Shared between SessionCUDA and SessionHybrid.
pub(super) fn compile_cuda_shared_lib(
    schedule: &Schedule,
    opt: &Options,
    build_dir: &Path,
) -> Result<PathBuf, SessionError> {
    let cuda_arch = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .map(|o| {
            let arch = String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("75")
                .replace('.', "");
            format!("sm_{}", arch)
        })
        .map_err(|e| {
            if e.kind() == ErrorKind::NotFound {
                SessionError::OtherError(
                    "nvidia-smi not found in PATH. Install the NVIDIA driver and ensure nvidia-smi is on PATH."
                        .to_string(),
                )
            } else {
                SessionError::OtherError(format!("nvidia-smi invocation failed: {e}"))
            }
        })?;
    let cuda_arch_num: u32 = cuda_arch
        .strip_prefix("sm_")
        .and_then(|s| s.parse().ok())
        .unwrap_or(75);

    let mut hostcode_gen = HostCodeGenerator::new(schedule, cuda_arch_num);
    let hostcode = hostcode_gen
        .generate(opt)
        .map_err(CodeGenError::CudaBuildError)
        .map_err(SessionError::CodeGenError)?;

    let main_file = build_dir.join("main.cu");
    let mut writer = std::fs::File::create(&main_file)
        .map_err(|e| SessionError::OtherError(format!("{:?}", e)))
        .map(BufWriter::new)?;
    hostcode
        .write(&mut writer)
        .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
    writer
        .flush()
        .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

    let kernel_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/codegen/cuda/cpp");
    let paths = vec![
        (main_file.clone(), main_file.with_extension("o")),
        (kernel_dir.join("common.cu"), build_dir.join("common.o")),
    ];
    info!("Generated");

    let shared_lib = build_dir.join("libmodel.so");

    info!("Compiling");

    let objs = paths
        .par_iter()
        .map(|(src, obj)| {
            Command::new("nvcc")
                .args([
                    src.to_str().unwrap(),
                    format!("-I{}", kernel_dir.to_str().unwrap()).as_str(),
                    "-std=c++17",
                    "-dc",
                    "-o",
                    obj.to_str().unwrap(),
                    "-lcudnn",
                    "-lcublas",
                    "-Xcompiler",
                    "-fPIC",
                    "-arch",
                    cuda_arch.as_str(),
                    "--expt-relaxed-constexpr",
                    "--diag-suppress=177", // unused variable
                ])
                .status()
                .map_err(nvcc_invocation_error)?;
            Ok::<PathBuf, SessionError>(obj.to_path_buf())
        })
        .collect::<Result<Vec<_>, SessionError>>()?;

    Command::new("nvcc")
        .args([
            "--shared",
            "-o",
            shared_lib.to_str().unwrap(),
            "-lcudnn",
            "-lcublas",
            "-Xcompiler",
            "-fPIC",
            "-arch",
            cuda_arch.as_str(),
            "--expt-relaxed-constexpr",
        ])
        .args(objs.iter().map(|p| p.to_str().unwrap()))
        .status()
        .map_err(nvcc_invocation_error)?;

    info!("Compiled");
    Ok(shared_lib)
}

impl SessionCUDA {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer_sources: Vec<InitializerSource>,
        initializer_names: Vec<String>,
        initializer_cache: Option<Arc<PersistentBuffers>>,
        session_state_buffers: Vec<Arc<DeviceBuffer>>,
        schedule: Schedule,
        opt: &Options,
        build_dir: &Path,
    ) -> Result<Self, SessionError> {
        let shared_lib = compile_cuda_shared_lib(&schedule, opt, build_dir)?;

        let _lock = cuda_lock();

        // HostResident initializers stay in host memory; everything else is
        // uploaded to VRAM as before. Derived from the plan (empty without a
        // prefetch policy, so the resident path is unchanged).
        let host_resident = host_resident_initializer_indices(&schedule);
        let mut host_resident_weights: HashMap<usize, HostWeight> = HashMap::new();

        let initializer_buffers: Vec<Arc<DeviceBuffer>> = initializer_sources
            .iter()
            .zip(initializer_names.iter())
            .enumerate()
            .map(
                |(i, (src, name))| -> Result<Arc<DeviceBuffer>, SessionError> {
                    if host_resident.contains(&i) {
                        // Keep the weight in host memory and stream it to a GPU
                        // staging chunk on demand. External weights are read into
                        // a pinned host buffer so the copy stream's H2D is truly
                        // async (overlaps compute); small inline weights stay
                        // owned (pageable; negligible). The device buffer is a
                        // 1-byte placeholder so model_init's pointer array stays
                        // valid; the host pointer is substituted in init_state.
                        let host = match src {
                            InitializerSource::External {
                                file,
                                offset,
                                length,
                                ..
                            } => {
                                let mut buf =
                                    PinnedHostBuffer::alloc(*length as usize).map_err(|e| {
                                        SessionError::OtherError(format!("cudaHostAlloc: {e}"))
                                    })?;
                                file.read_exact_at(unsafe { buf.as_mut_slice() }, *offset)
                                    .map_err(ModelLoadError::FileRead)
                                    .map_err(SessionError::ModelLoadError)?;
                                HostWeight::Pinned(buf)
                            }
                            InitializerSource::Inline(_) => HostWeight::Owned(
                                src.load_into_strict()
                                    .map_err(SessionError::ModelLoadError)?,
                            ),
                        };
                        host_resident_weights.insert(i, host);
                        let buf = DeviceBuffer::alloc_zeroed(1).map_err(|e| {
                            SessionError::OtherError(format!("cudaMalloc placeholder: {:?}", e))
                        })?;
                        return Ok(Arc::new(buf));
                    }
                    let upload = || -> Result<Arc<DeviceBuffer>, SessionError> {
                        let len = src.byte_len();
                        let buf =
                            Arc::new(DeviceBuffer::alloc_zeroed(len.max(1)).map_err(|e| {
                                SessionError::OtherError(format!("cudaMalloc: {:?}", e))
                            })?);
                        if 0 < len {
                            send_initializer_to_device(src, &buf)?;
                        }
                        Ok(buf)
                    };
                    match initializer_cache.as_ref() {
                        Some(cache) => cache.get_or_insert_with(name, upload),
                        None => upload(),
                    }
                },
            )
            .collect::<Result<Vec<_>, _>>()?;

        let lib = unsafe { libloading::Library::new(shared_lib.as_os_str()) }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        let init_func: libloading::Symbol<InitType> = unsafe { lib.get(b"model_init") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let run_func: libloading::Symbol<RunType> = unsafe { lib.get(b"model") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let destroy_func: libloading::Symbol<DestroyType> = unsafe { lib.get(b"model_destroy") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        let init_func = *init_func;
        let run_func = *run_func;
        let destroy_func = *destroy_func;

        info!("Loaded");

        Ok(Self {
            input_ty,
            output_ty,
            lib,
            init_func,
            run_func,
            destroy_func,
            state: std::ptr::null_mut(),
            initializer_buffers,
            host_resident_weights,
            session_state_buffers,
        })
    }

    fn init_state(&self) -> Result<*mut std::ffi::c_void, SessionError> {
        let initializer_ptrs: Vec<*const u8> = self
            .initializer_buffers
            .iter()
            .enumerate()
            .map(|(i, b)| match self.host_resident_weights.get(&i) {
                Some(h) => h.as_ptr(),
                None => b.ptr() as *const u8,
            })
            .collect();
        let session_state_ptrs: Vec<*mut std::ffi::c_void> =
            self.session_state_buffers.iter().map(|b| b.ptr()).collect();
        let state =
            unsafe { (self.init_func)(initializer_ptrs.as_ptr(), session_state_ptrs.as_ptr()) };
        Ok(state)
    }

    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        let _lock = cuda_lock();
        if self.state.is_null() {
            self.state = self.init_state()?;
        }
        let mut output_bufs = self
            .output_ty
            .iter()
            .map(|ty| StrictTensor::zeros(ty.elem_type, &ty.dims))
            .collect::<Vec<_>>();
        let output_ptrs = output_bufs
            .iter_mut()
            .map(|x| x.as_mut_ptr())
            .collect::<Vec<_>>();
        let input_bufs = inputs.iter().map(StrictTensor::from).collect::<Vec<_>>();
        let input_ptrs = input_bufs.iter().map(|t| t.as_ptr()).collect::<Vec<_>>();
        unsafe { (self.run_func)(self.state, output_ptrs.as_ptr(), input_ptrs.as_ptr()) };
        let outputs = zip_eq(self.output_ty.iter(), output_bufs)
            .map(|(ty, buf)| buf.into_tensor(ty.dims.clone(), ty.elem_type))
            .collect::<Vec<_>>();
        Ok(outputs)
    }
}

// SAFETY: FFI state is owned exclusively; transfer between threads is fine, but the raw pointer prevents shared access (no Sync).
unsafe impl Send for SessionCUDA {}

impl Drop for SessionCUDA {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe {
                (self.destroy_func)(self.state);
            }
        }
    }
}
