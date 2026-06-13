use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::OptimizationLevel;
use itertools::izip;
use itertools::Itertools;
use log::info;
use rayon::prelude::*;

use crate::codegen::cpu::get_kernel_name_or;
use crate::codegen::cpu::CodeGenContext as CpuCodeGenContext;
use crate::options::Options;
use crate::schedule::ir::AllocPlace;
use crate::schedule::ir::ArenaId;
use crate::schedule::ir::Device;
use crate::schedule::ir::ExecutionPlan;
use crate::schedule::ir::MemoryTier;
use crate::schedule::ir::Step;
use crate::schedule::ChunkId;
use crate::schedule::KernelId;
use crate::schedule::Schedule;
use crate::session::cpu::cpu_kernel_symbols;
use crate::session::cpu::CpuJitState;
use crate::session::cuda::compile_cuda_shared_lib;
use crate::session::cuda::cuda_lock;
use crate::session::send_initializer_to_device;
use crate::session::shared_lib::load_jit_runtime;
use crate::session::DeviceBuffer;
use crate::session::InitializerSource;
use crate::session::PersistentBuffers;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

// CPU kernels are emitted with one `*mut u8` per binding (outputs followed by
// inputs), so we dispatch by arity. Passing a single `*const *mut u8` would
// not match the ABI and corrupts the heap on call.
type KernelFn0 = unsafe extern "C" fn();
type KernelFn1 = unsafe extern "C" fn(*mut u8);
type KernelFn2 = unsafe extern "C" fn(*mut u8, *mut u8);
type KernelFn3 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8);
type KernelFn4 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8);
type KernelFn5 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8, *mut u8);
type KernelFn6 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8);
type KernelFn7 =
    unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8);
type KernelFn8 =
    unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8);

unsafe fn call_kernel(addr: u64, args: &[*mut u8]) {
    unsafe {
        match args.len() {
            0 => (std::mem::transmute::<u64, KernelFn0>(addr))(),
            1 => (std::mem::transmute::<u64, KernelFn1>(addr))(args[0]),
            2 => (std::mem::transmute::<u64, KernelFn2>(addr))(args[0], args[1]),
            3 => (std::mem::transmute::<u64, KernelFn3>(addr))(args[0], args[1], args[2]),
            4 => (std::mem::transmute::<u64, KernelFn4>(addr))(args[0], args[1], args[2], args[3]),
            5 => (std::mem::transmute::<u64, KernelFn5>(addr))(
                args[0], args[1], args[2], args[3], args[4],
            ),
            6 => (std::mem::transmute::<u64, KernelFn6>(addr))(
                args[0], args[1], args[2], args[3], args[4], args[5],
            ),
            7 => (std::mem::transmute::<u64, KernelFn7>(addr))(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6],
            ),
            8 => (std::mem::transmute::<u64, KernelFn8>(addr))(
                args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7],
            ),
            n => panic!("CPU kernel arity {n} not supported"),
        }
    }
}

type CudaInitFn =
    unsafe extern "C" fn(*const *const u8, *const *mut std::ffi::c_void) -> *mut std::ffi::c_void;
type CudaDestroyFn = unsafe extern "C" fn(*mut std::ffi::c_void);
type CudaStepFn = unsafe extern "C" fn(*mut std::ffi::c_void, *const *mut u8, *const *const u8);
type CudaSyncFn = unsafe extern "C" fn();
type CudaChunksFn = unsafe extern "C" fn(*mut std::ffi::c_void, *mut *mut std::ffi::c_void);
type CudaMemcpyFn =
    unsafe extern "C" fn(*mut std::ffi::c_void, *const std::ffi::c_void, usize, i32) -> i32;

const CUDA_MEMCPY_HOST_TO_HOST: i32 = 0;
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

struct CudaState {
    // Declaration order = drop order. initializer_buffers must drop BEFORE lib
    // so cudaFree runs while the model .so (and through it cudart) is still
    // mapped.
    #[allow(dead_code)]
    initializer_buffers: Vec<Arc<DeviceBuffer>>,
    kernel_fns: HashMap<KernelId, CudaStepFn>,
    #[allow(dead_code)]
    transfer_fns: HashMap<usize, CudaStepFn>,
    destroy_func: CudaDestroyFn,
    sync_func: CudaSyncFn,
    memcpy_func: CudaMemcpyFn,
    gpu_chunks: HashMap<ChunkId, *mut std::ffi::c_void>,
    state: *mut std::ffi::c_void,
    #[allow(dead_code)]
    lib: libloading::Library,
}

unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl CudaState {
    fn destroy(&mut self) {
        if !self.state.is_null() {
            unsafe { (self.sync_func)() };
            unsafe { (self.destroy_func)(self.state) };
            self.state = std::ptr::null_mut();
        }
    }
}

fn build_cuda_state(
    schedule: &Schedule,
    initializer_sources: &[InitializerSource],
    initializer_names: &[String],
    initializer_cache: Option<&Arc<PersistentBuffers>>,
    session_state_buffers: &[Arc<DeviceBuffer>],
    opt: &Options,
    build_dir: &Path,
) -> Result<CudaState, SessionError> {
    info!("Hybrid: compiling CUDA shared lib");
    let shared_lib = compile_cuda_shared_lib(schedule, opt, build_dir)?;

    // CUDA's model_init takes a pointer array indexed by global initializer
    // index, but for hybrid placements only initializers actually consumed
    // by CUDA kernels need device memory. Upload only those; the rest get
    // a 1-byte placeholder so model_init still receives a valid pointer.
    let gpu_initializer_indices = if let Some(plan) = schedule.execution_plan.as_ref() {
        plan.steps
            .iter()
            .flat_map(|step| match step {
                Step::Kernel(k) if k.context.device == Device::CUDA => {
                    k.bindings.iter().map(|b| b.place).collect::<Vec<_>>()
                }
                Step::Transfer(t) => vec![t.src.place, t.dst.place],
                _ => Vec::new(),
            })
            .filter_map(|place| match place {
                AllocPlace::Initializer(v) => Some(v),
                _ => None,
            })
            .filter_map(|v| schedule.initializers.iter().position(|x| *x == v))
            .collect::<HashSet<_>>()
    } else {
        HashSet::new()
    };

    let _lock = cuda_lock();
    let initializer_buffers = izip!(initializer_sources.iter(), initializer_names.iter())
        .enumerate()
        .map(
            |(i, (src, name))| -> Result<Arc<DeviceBuffer>, SessionError> {
                if !gpu_initializer_indices.contains(&i) {
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
                match initializer_cache {
                    Some(cache) => cache.get_or_insert_with(name, upload),
                    None => upload(),
                }
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    info!(
        "Hybrid: uploaded {} / {} initializers to GPU",
        gpu_initializer_indices.len(),
        initializer_sources.len(),
    );

    let lib = unsafe { libloading::Library::new(shared_lib.as_os_str()) }
        .map_err(|e| SessionError::OtherError(format!("dlopen: {:?}", e)))?;

    let init_func: CudaInitFn = unsafe {
        *lib.get::<CudaInitFn>(b"model_init")
            .map_err(|e| SessionError::OtherError(format!("dlsym model_init: {:?}", e)))?
    };
    let destroy_func: CudaDestroyFn = unsafe {
        *lib.get::<CudaDestroyFn>(b"model_destroy")
            .map_err(|e| SessionError::OtherError(format!("dlsym model_destroy: {:?}", e)))?
    };
    let sync_func: CudaSyncFn = unsafe {
        *lib.get::<CudaSyncFn>(b"model_device_sync")
            .map_err(|e| SessionError::OtherError(format!("dlsym model_device_sync: {:?}", e)))?
    };
    let memcpy_func: CudaMemcpyFn = unsafe {
        *lib.get::<CudaMemcpyFn>(b"model_memcpy")
            .map_err(|e| SessionError::OtherError(format!("dlsym model_memcpy: {:?}", e)))?
    };
    let chunks_func: CudaChunksFn = unsafe {
        *lib.get::<CudaChunksFn>(b"model_chunks")
            .map_err(|e| SessionError::OtherError(format!("dlsym model_chunks: {:?}", e)))?
    };

    let plan = schedule
        .execution_plan
        .as_ref()
        .ok_or_else(|| SessionError::OtherError("execution_plan missing".to_string()))?;
    let mut kernel_fns: HashMap<KernelId, CudaStepFn> = HashMap::new();
    let mut transfer_fns: HashMap<usize, CudaStepFn> = HashMap::new();
    for (idx, step) in plan.steps.iter().enumerate() {
        match step {
            Step::Kernel(k) if k.context.device == Device::CUDA => {
                let name = format!("model_step_kernel_{}\0", k.kernel.index());
                let f: CudaStepFn = unsafe {
                    *lib.get::<CudaStepFn>(name.as_bytes())
                        .map_err(|e| SessionError::OtherError(format!("dlsym {name}: {:?}", e)))?
                };
                kernel_fns.insert(k.kernel, f);
            }
            Step::Transfer(_) => {
                let name = format!("model_step_transfer_{idx}\0");
                let f: CudaStepFn = unsafe {
                    *lib.get::<CudaStepFn>(name.as_bytes())
                        .map_err(|e| SessionError::OtherError(format!("dlsym {name}: {:?}", e)))?
                };
                transfer_fns.insert(idx, f);
            }
            _ => {}
        }
    }

    let initializer_ptrs: Vec<*const u8> = initializer_buffers
        .iter()
        .map(|b| b.ptr() as *const u8)
        .collect();
    let session_state_ptrs: Vec<*mut std::ffi::c_void> =
        session_state_buffers.iter().map(|b| b.ptr()).collect();
    let state = unsafe { (init_func)(initializer_ptrs.as_ptr(), session_state_ptrs.as_ptr()) };

    let chunk_count = plan
        .chunks
        .iter()
        .map(|c| c.id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    let mut chunk_buf: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); chunk_count];
    unsafe { (chunks_func)(state, chunk_buf.as_mut_ptr()) };
    let gpu_chunks: HashMap<_, _> = plan
        .chunks
        .iter()
        .filter(|c| plan.arenas[c.arena].tier == MemoryTier::GpuArena)
        .map(|c| (c.id, chunk_buf[c.id]))
        .collect();

    Ok(CudaState {
        lib,
        kernel_fns,
        transfer_fns,
        destroy_func,
        sync_func,
        memcpy_func,
        gpu_chunks,
        state,
        initializer_buffers,
    })
}

pub struct SessionHybrid {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,

    schedule: Schedule,
    initializer: Vec<StrictTensor>,
    /// One buffer per session_state, each already on the device its kernels run
    /// on (`DeviceBuffer::is_host()` distinguishes host KV from VRAM KV).
    session_state_buffers: Vec<Arc<DeviceBuffer>>,

    cpu_kernel_fns: HashMap<KernelId, u64>,
    #[allow(dead_code)]
    cpu_jit: CpuJitState,

    host_arenas: HostArenas,
    cuda: Option<CudaState>,
}

struct HostArenas {
    #[allow(dead_code)]
    arenas: HashMap<ArenaId, Vec<u8>>,
    chunk2ptrs: HashMap<ChunkId, *mut u8>,
}

// SAFETY: chunk2ptrs is derived from `arenas` owned by Self and is only
// dereferenced from the thread that holds the SessionHybrid run lock.
unsafe impl Send for HostArenas {}
unsafe impl Sync for HostArenas {}

impl HostArenas {
    fn new(plan: &ExecutionPlan) -> Self {
        let arenas = plan
            .arenas
            .iter()
            .filter(|a| a.tier == MemoryTier::HostArena)
            .map(|a| (a.id, vec![0u8; a.size.max(1)]))
            .collect::<HashMap<_, _>>();

        let chunk2ptrs = plan
            .chunks
            .iter()
            .filter_map(|chunk| {
                arenas.get(&chunk.arena).map(|arena| {
                    let ptr = unsafe { (arena.as_ptr() as *mut u8).add(chunk.offset) };
                    (chunk.id, ptr)
                })
            })
            .collect::<HashMap<_, _>>();

        Self { arenas, chunk2ptrs }
    }
}

impl SessionHybrid {
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
        let initializer: Vec<StrictTensor> = initializer_sources
            .iter()
            .map(|src| src.load_into_strict())
            .collect::<Result<Vec<_>, _>>()
            .map_err(SessionError::ModelLoadError)?;

        info!("Hybrid: loading BLAS + OpenMP");
        let blas_backend = load_jit_runtime().map_err(SessionError::OtherError)?;

        let codegen_ctx =
            CpuCodeGenContext::new(schedule, blas_backend).map_err(SessionError::CodeGenError)?;
        let kernel_ids = codegen_ctx.all_necessary_kernels();

        let mut contexts = kernel_ids.iter().map(|_| Context::create()).collect_vec();

        info!("Hybrid: compiling {} CPU kernels", kernel_ids.len());
        let codegens = kernel_ids
            .iter()
            .zip(contexts.iter())
            .map(|(&id, ctx)| codegen_ctx.new_codegen_for_kernel(id, ctx))
            .collect::<Result<Vec<_>, _>>()
            .map_err(SessionError::CodeGenError)?;
        codegens
            .par_iter()
            .try_for_each(|cg| -> Result<(), String> {
                cg.compile().map_err(|e| format!("{e:?}"))?;
                cg.run_opt_aggressive().map_err(|e| format!("{e:?}"))?;
                Ok(())
            })
            .map_err(SessionError::OtherError)?;

        let kernel_modules: Vec<_> = codegens.into_iter().map(|cg| cg.into_module()).collect();

        let host_ctx = Context::create();
        let host_module = host_ctx.create_module("hybrid_host");
        let engine = host_module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|e| SessionError::OtherError(format!("JIT init: {:?}", e)))?;
        let engine: ExecutionEngine<'static> = unsafe { std::mem::transmute(engine) };

        for module in &kernel_modules {
            engine
                .add_module(unsafe {
                    std::mem::transmute::<
                        &inkwell::module::Module<'_>,
                        &inkwell::module::Module<'_>,
                    >(module)
                })
                .map_err(|()| SessionError::OtherError("add_module failed".to_string()))?;
        }

        for module in &kernel_modules {
            for (name, addr) in cpu_kernel_symbols() {
                if let Some(f) = module.get_function(name) {
                    engine.add_global_mapping(&f, addr);
                }
            }
        }

        let mut cpu_kernel_fns: HashMap<KernelId, u64> = HashMap::new();
        for &kid in &kernel_ids {
            let kernel = &codegen_ctx.schedule.kernels[kid];
            let name = get_kernel_name_or(kernel, kid);
            let addr = engine
                .get_function_address(&name)
                .map_err(|e| SessionError::OtherError(format!("get_function_address: {:?}", e)))?;
            cpu_kernel_fns.insert(kid, addr as u64);
        }

        drop(kernel_modules);
        drop(host_module);
        contexts.push(host_ctx);

        let CpuCodeGenContext { schedule, .. } = codegen_ctx;
        let host_arenas = {
            let plan = schedule
                .execution_plan
                .as_ref()
                .ok_or_else(|| SessionError::OtherError("execution_plan missing".to_string()))?;
            HostArenas::new(plan)
        };

        let needs_cuda = schedule
            .execution_plan
            .as_ref()
            .map(|plan| {
                plan.steps.iter().any(|step| match step {
                    Step::Kernel(k) => k.context.device == Device::CUDA,
                    Step::Transfer(_) => true,
                    _ => false,
                })
            })
            .unwrap_or(false);

        let cuda = if needs_cuda {
            Some(build_cuda_state(
                &schedule,
                &initializer_sources,
                &initializer_names,
                initializer_cache.as_ref(),
                &session_state_buffers,
                opt,
                build_dir,
            )?)
        } else {
            None
        };

        Ok(Self {
            input_ty,
            output_ty,
            schedule,
            initializer,
            session_state_buffers,
            cpu_kernel_fns,
            cpu_jit: CpuJitState {
                _engine: engine,
                _contexts: contexts,
            },
            host_arenas,
            cuda,
        })
    }

    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        let plan = self
            .schedule
            .execution_plan
            .as_ref()
            .ok_or_else(|| SessionError::OtherError("execution_plan missing".to_string()))?;

        let mut output_bufs = self
            .output_ty
            .iter()
            .map(|ty| StrictTensor::zeros(ty.elem_type, &ty.dims))
            .collect_vec();
        let input_bufs = inputs.iter().map(StrictTensor::from).collect_vec();

        let input_ptrs = input_bufs
            .iter()
            .map(|t| t.as_ptr() as *mut u8)
            .collect_vec();
        let output_ptrs = output_bufs.iter_mut().map(|t| t.as_mut_ptr()).collect_vec();

        let chunk_tiers: HashMap<_, _> = plan
            .chunks
            .iter()
            .map(|chunk| (chunk.id, plan.arenas[chunk.arena].tier))
            .collect();

        let resolve = |place: AllocPlace| -> Result<*mut u8, SessionError> {
            Ok(match place {
                AllocPlace::Chunk(cid) => self
                    .host_arenas
                    .chunk2ptrs
                    .get(&cid)
                    .copied()
                    .ok_or_else(|| SessionError::OtherError(format!("missing chunk {cid:?}")))?,
                AllocPlace::Input(v) => {
                    let idx = self
                        .schedule
                        .inputs
                        .iter()
                        .position(|x| *x == v)
                        .ok_or_else(|| {
                            SessionError::OtherError(format!("input {v:?} not found"))
                        })?;
                    input_ptrs[idx]
                }
                AllocPlace::Output(v) => {
                    let idx = self
                        .schedule
                        .outputs
                        .iter()
                        .position(|x| *x == v)
                        .ok_or_else(|| {
                            SessionError::OtherError(format!("output {v:?} not found"))
                        })?;
                    output_ptrs[idx]
                }
                AllocPlace::Initializer(v) => {
                    let idx = self
                        .schedule
                        .initializers
                        .iter()
                        .position(|x| *x == v)
                        .ok_or_else(|| {
                            SessionError::OtherError(format!("initializer {v:?} not found"))
                        })?;
                    self.initializer[idx].as_ptr() as *mut u8
                }
                AllocPlace::SessionState(v) => {
                    let idx = self
                        .schedule
                        .session_states
                        .iter()
                        .position(|x| *x == v)
                        .ok_or_else(|| {
                            SessionError::OtherError(format!("session_state {v:?} not found"))
                        })?;
                    let buf = &self.session_state_buffers[idx];
                    if !buf.is_host() {
                        return Err(SessionError::OtherError(format!(
                            "CPU kernel touches GPU-resident session_state {v:?} \
                             (mixed-device session state is not supported)"
                        )));
                    }
                    buf.ptr() as *mut u8
                }
            })
        };

        for step in plan.steps.iter() {
            match step {
                Step::Kernel(k) if k.context.device == Device::CPU => {
                    let kernel = &self.schedule.kernels[k.kernel];
                    let mut args = Vec::with_capacity(kernel.outputs.len() + kernel.inputs.len());
                    let bindings_by_value: HashMap<_, _> =
                        k.bindings.iter().map(|b| (b.value, b.place)).collect();
                    for &out in &kernel.outputs {
                        let place = *bindings_by_value.get(&out).ok_or_else(|| {
                            SessionError::OtherError(format!("output binding for {out:?} missing"))
                        })?;
                        args.push(resolve(place)?);
                    }
                    for input in kernel.inputs.iter().flatten() {
                        let place = *bindings_by_value.get(input).ok_or_else(|| {
                            SessionError::OtherError(format!("input binding for {input:?} missing"))
                        })?;
                        args.push(resolve(place)?);
                    }
                    // Identity / Reinterpret with aliased input/output chunk
                    // is elided by need_to_generate; skip silently here.
                    let Some(&addr) = self.cpu_kernel_fns.get(&k.kernel) else {
                        continue;
                    };
                    unsafe { call_kernel(addr, &args) };
                }
                Step::Kernel(k) => {
                    let cuda = self.cuda.as_ref().ok_or_else(|| {
                        SessionError::OtherError(
                            "CUDA kernel encountered but CudaState was not built".to_string(),
                        )
                    })?;
                    let f = cuda.kernel_fns.get(&k.kernel).ok_or_else(|| {
                        SessionError::OtherError(format!("CUDA kernel {:?} not loaded", k.kernel))
                    })?;
                    unsafe {
                        (*f)(cuda.state, output_ptrs.as_ptr(), input_ptrs.as_ptr() as _);
                    }
                }
                Step::Transfer(t) => {
                    let cuda = self.cuda.as_ref().ok_or_else(|| {
                        SessionError::OtherError(
                            "Transfer step encountered but CudaState was not built".to_string(),
                        )
                    })?;
                    let resolve_with_tier =
                        |place: AllocPlace| -> Result<(*mut u8, MemoryTier), SessionError> {
                            Ok(match place {
                                AllocPlace::Chunk(cid) => {
                                    let tier = *chunk_tiers.get(&cid).ok_or_else(|| {
                                        SessionError::OtherError(format!("unknown chunk {cid:?}"))
                                    })?;
                                    let p = match tier {
                                        MemoryTier::HostArena => {
                                            *self.host_arenas.chunk2ptrs.get(&cid).ok_or_else(
                                                || {
                                                    SessionError::OtherError(format!(
                                                        "missing host chunk {cid:?}"
                                                    ))
                                                },
                                            )?
                                        }
                                        MemoryTier::GpuArena => {
                                            *cuda.gpu_chunks.get(&cid).ok_or_else(|| {
                                                SessionError::OtherError(format!(
                                                    "missing gpu chunk {cid:?}"
                                                ))
                                            })?
                                                as *mut u8
                                        }
                                    };
                                    (p, tier)
                                }
                                AllocPlace::Input(v) => {
                                    let i = self
                                        .schedule
                                        .inputs
                                        .iter()
                                        .position(|x| *x == v)
                                        .ok_or_else(|| {
                                            SessionError::OtherError(format!("input {v:?}"))
                                        })?;
                                    (input_ptrs[i], MemoryTier::HostArena)
                                }
                                AllocPlace::Output(v) => {
                                    let i = self
                                        .schedule
                                        .outputs
                                        .iter()
                                        .position(|x| *x == v)
                                        .ok_or_else(|| {
                                            SessionError::OtherError(format!("output {v:?}"))
                                        })?;
                                    (output_ptrs[i], MemoryTier::HostArena)
                                }
                                AllocPlace::SessionState(v) => {
                                    let i = self
                                        .schedule
                                        .session_states
                                        .iter()
                                        .position(|x| *x == v)
                                        .ok_or_else(|| {
                                            SessionError::OtherError(format!(
                                                "session_state {v:?} not found"
                                            ))
                                        })?;
                                    let buf = &self.session_state_buffers[i];
                                    let tier = if buf.is_host() {
                                        MemoryTier::HostArena
                                    } else {
                                        MemoryTier::GpuArena
                                    };
                                    (buf.ptr() as *mut u8, tier)
                                }
                                AllocPlace::Initializer(v) => {
                                    let i = self
                                        .schedule
                                        .initializers
                                        .iter()
                                        .position(|x| *x == v)
                                        .ok_or_else(|| {
                                            SessionError::OtherError(format!(
                                                "initializer {v:?} not found"
                                            ))
                                        })?;
                                    (
                                        self.initializer[i].as_ptr() as *mut u8,
                                        MemoryTier::HostArena,
                                    )
                                }
                            })
                        };
                    let (src_ptr, src_tier) = resolve_with_tier(t.src.place)?;
                    let (dst_ptr, dst_tier) = resolve_with_tier(t.dst.place)?;
                    let size = self.schedule.value_byte_size(t.dst.value);
                    let kind = match (src_tier, dst_tier) {
                        (MemoryTier::HostArena, MemoryTier::HostArena) => CUDA_MEMCPY_HOST_TO_HOST,
                        (MemoryTier::HostArena, MemoryTier::GpuArena) => CUDA_MEMCPY_HOST_TO_DEVICE,
                        (MemoryTier::GpuArena, MemoryTier::HostArena) => CUDA_MEMCPY_DEVICE_TO_HOST,
                        (MemoryTier::GpuArena, MemoryTier::GpuArena) => {
                            CUDA_MEMCPY_DEVICE_TO_DEVICE
                        }
                    };
                    let rc = unsafe {
                        (cuda.memcpy_func)(
                            dst_ptr as *mut std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            size,
                            kind,
                        )
                    };
                    if rc != 0 {
                        return Err(SessionError::OtherError(format!(
                            "cudaMemcpy failed: rc={rc}"
                        )));
                    }
                }
                Step::SyncWait(_) => {}
            }
        }

        if let Some(cuda) = self.cuda.as_ref() {
            unsafe { (cuda.sync_func)() };
        }

        let outputs = output_bufs
            .into_iter()
            .zip(self.output_ty.iter())
            .map(|(buf, ty)| buf.into_tensor(ty.dims.clone(), ty.elem_type))
            .collect();
        Ok(outputs)
    }
}

impl Drop for SessionHybrid {
    fn drop(&mut self) {
        if let Some(cuda) = self.cuda.as_mut() {
            cuda.destroy();
        }
    }
}
