use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::OptimizationLevel;
use log::info;
use rayon::prelude::*;

use crate::codegen::cpu::blas;
use crate::codegen::cpu::get_kernel_name_or;
use crate::codegen::cpu::CodeGenContext as CpuCodeGenContext;
use crate::options::Options;
use crate::schedule::ir::AllocPlace;
use crate::schedule::ir::ArenaId;
use crate::schedule::ir::Device;
use crate::schedule::ir::MemoryTier;
use crate::schedule::ir::Step;
use crate::schedule::ChunkId;
use crate::schedule::KernelId;
use crate::schedule::Schedule;
use crate::session::cuda::compile_cuda_shared_lib;
use crate::session::cuda::cuda_lock;
use crate::session::send_initializer_to_device;
use crate::session::DeviceBuffer;
use crate::session::InitializerBuffers;
use crate::session::InitializerSource;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

type KernelFn = unsafe extern "C" fn(*const *mut u8);

type CudaInitFn =
    unsafe extern "C" fn(*const *const u8, *const *mut std::ffi::c_void) -> *mut std::ffi::c_void;
type CudaDestroyFn = unsafe extern "C" fn(*mut std::ffi::c_void);
type CudaStepFn = unsafe extern "C" fn(*mut std::ffi::c_void, *const *mut u8, *const *const u8);
type CudaSyncFn = unsafe extern "C" fn();

struct CpuJitState {
    _engine: ExecutionEngine<'static>,
    _contexts: Vec<Context>,
}

unsafe impl Send for CpuJitState {}
unsafe impl Sync for CpuJitState {}

struct CudaState {
    #[allow(dead_code)]
    lib: libloading::Library,
    kernel_fns: HashMap<KernelId, CudaStepFn>,
    transfer_fns: HashMap<usize, CudaStepFn>,
    destroy_func: CudaDestroyFn,
    sync_func: CudaSyncFn,
    state: *mut std::ffi::c_void,
    #[allow(dead_code)]
    initializer_buffers: Vec<Arc<DeviceBuffer>>,
}

unsafe impl Send for CudaState {}
unsafe impl Sync for CudaState {}

impl Drop for CudaState {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe { (self.destroy_func)(self.state) };
            self.state = std::ptr::null_mut();
        }
    }
}

pub struct SessionHybrid {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,

    schedule: Schedule,
    initializer: Vec<StrictTensor>,
    #[allow(dead_code)]
    session_state_buffers: Vec<Arc<DeviceBuffer>>,

    cpu_kernel_fns: HashMap<KernelId, KernelFn>,
    #[allow(dead_code)]
    cpu_jit: CpuJitState,

    cuda: CudaState,
}

impl SessionHybrid {
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer_sources: Vec<InitializerSource>,
        initializer_names: Vec<String>,
        initializer_cache: Option<Arc<InitializerBuffers>>,
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
        let blas_backend = blas::load_jit_runtime().map_err(SessionError::OtherError)?;

        let codegen_ctx =
            CpuCodeGenContext::new(schedule, blas_backend).map_err(SessionError::CodeGenError)?;
        let kernel_ids = codegen_ctx.all_necessary_kernels();

        let mut contexts: Vec<Context> = kernel_ids.iter().map(|_| Context::create()).collect();

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
                .add_module(unsafe { std::mem::transmute(module) })
                .map_err(|()| SessionError::OtherError("add_module failed".to_string()))?;
        }

        let mut cpu_kernel_fns: HashMap<KernelId, KernelFn> = HashMap::new();
        for &kid in &kernel_ids {
            let kernel = &codegen_ctx.schedule.kernels[kid];
            let name = get_kernel_name_or(kernel, kid);
            let addr = engine
                .get_function_address(&name)
                .map_err(|e| SessionError::OtherError(format!("get_function_address: {:?}", e)))?;
            let f: KernelFn = unsafe { std::mem::transmute(addr) };
            cpu_kernel_fns.insert(kid, f);
        }

        drop(kernel_modules);
        drop(host_module);
        contexts.push(host_ctx);

        let CpuCodeGenContext { schedule, .. } = codegen_ctx;

        info!("Hybrid: compiling CUDA shared lib");
        let shared_lib = compile_cuda_shared_lib(&schedule, opt, build_dir)?;

        // CUDA's model_init takes a pointer array indexed by global initializer
        // index, but for hybrid placements only initializers actually consumed
        // by CUDA kernels need device memory. Upload only those; the rest get
        // a 1-byte placeholder so model_init still receives a valid pointer.
        let mut gpu_initializer_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        if let Some(plan) = schedule.execution_plan.as_ref() {
            for step in &plan.steps {
                if let Step::Kernel(k) = step {
                    if k.context.device != Device::CUDA {
                        continue;
                    }
                    for b in &k.bindings {
                        if let AllocPlace::Initializer(v) = b.place {
                            if let Some(i) = schedule.initializers.iter().position(|x| *x == v) {
                                gpu_initializer_indices.insert(i);
                            }
                        }
                    }
                }
            }
        }

        let _lock = cuda_lock();
        let initializer_buffers: Vec<Arc<DeviceBuffer>> = initializer_sources
            .iter()
            .zip(initializer_names.iter())
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
                        if len > 0 {
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
            *lib.get::<CudaSyncFn>(b"model_device_sync").map_err(|e| {
                SessionError::OtherError(format!("dlsym model_device_sync: {:?}", e))
            })?
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
                        *lib.get::<CudaStepFn>(name.as_bytes()).map_err(|e| {
                            SessionError::OtherError(format!("dlsym {name}: {:?}", e))
                        })?
                    };
                    kernel_fns.insert(k.kernel, f);
                }
                Step::Transfer(_) => {
                    let name = format!("model_step_transfer_{idx}\0");
                    let f: CudaStepFn = unsafe {
                        *lib.get::<CudaStepFn>(name.as_bytes()).map_err(|e| {
                            SessionError::OtherError(format!("dlsym {name}: {:?}", e))
                        })?
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
            cuda: CudaState {
                lib,
                kernel_fns,
                transfer_fns,
                destroy_func,
                sync_func,
                state,
                initializer_buffers,
            },
        })
    }

    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        let plan = self
            .schedule
            .execution_plan
            .as_ref()
            .ok_or_else(|| SessionError::OtherError("execution_plan missing".to_string()))?;

        let mut output_bufs: Vec<StrictTensor> = self
            .output_ty
            .iter()
            .map(|ty| StrictTensor::zeros(ty.elem_type, &ty.dims))
            .collect();
        let input_bufs: Vec<StrictTensor> = inputs.iter().map(StrictTensor::from).collect();

        let input_ptrs: Vec<*mut u8> = input_bufs.iter().map(|t| t.as_ptr() as *mut u8).collect();
        let output_ptrs: Vec<*mut u8> = output_bufs.iter_mut().map(|t| t.as_mut_ptr()).collect();

        let mut host_arenas: HashMap<ArenaId, Vec<u8>> = HashMap::new();
        for arena in &plan.arenas {
            if arena.tier != MemoryTier::HostArena {
                continue;
            }
            // size.max(1) keeps as_mut_ptr non-dangling for empty arenas;
            // some chunks are referenced as parameters even when their
            // tensors are empty, and the kernel must still see a valid ptr.
            host_arenas.insert(arena.id, vec![0u8; arena.size.max(1)]);
        }

        let mut chunk_ptrs: HashMap<ChunkId, *mut u8> = HashMap::new();
        for chunk in &plan.chunks {
            let Some(arena) = host_arenas.get_mut(&chunk.arena) else {
                continue;
            };
            let ptr = unsafe { arena.as_mut_ptr().add(chunk.offset) };
            chunk_ptrs.insert(chunk.id, ptr);
        }

        let resolve = |place: AllocPlace,
                       chunk_ptrs: &HashMap<ChunkId, *mut u8>|
         -> Result<*mut u8, SessionError> {
            Ok(match place {
                AllocPlace::Chunk(cid) => *chunk_ptrs
                    .get(&cid)
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
                AllocPlace::SessionState(_) => {
                    return Err(SessionError::OtherError(
                        "SessionState not yet supported in hybrid runtime".to_string(),
                    ));
                }
            })
        };

        for (idx, step) in plan.steps.iter().enumerate() {
            match step {
                Step::Kernel(k) if k.context.device == Device::CPU => {
                    let kernel = &self.schedule.kernels[k.kernel];
                    let mut args: Vec<*mut u8> =
                        Vec::with_capacity(kernel.outputs.len() + kernel.inputs.len());
                    let bindings_by_value: HashMap<_, _> =
                        k.bindings.iter().map(|b| (b.value, b.place)).collect();
                    for &out in &kernel.outputs {
                        let place = *bindings_by_value.get(&out).ok_or_else(|| {
                            SessionError::OtherError(format!("output binding for {out:?} missing"))
                        })?;
                        args.push(resolve(place, &chunk_ptrs)?);
                    }
                    for input in kernel.inputs.iter().flatten().copied() {
                        let place = *bindings_by_value.get(&input).ok_or_else(|| {
                            SessionError::OtherError(format!("input binding for {input:?} missing"))
                        })?;
                        args.push(resolve(place, &chunk_ptrs)?);
                    }
                    // Identity / Reinterpret with aliased input/output chunk
                    // is elided by need_to_generate; skip silently here.
                    let Some(f) = self.cpu_kernel_fns.get(&k.kernel) else {
                        continue;
                    };
                    unsafe { f(args.as_ptr()) };
                }
                Step::Kernel(k) => {
                    let f = self.cuda.kernel_fns.get(&k.kernel).ok_or_else(|| {
                        SessionError::OtherError(format!("CUDA kernel {:?} not loaded", k.kernel))
                    })?;
                    unsafe {
                        (*f)(
                            self.cuda.state,
                            output_ptrs.as_ptr(),
                            input_ptrs.as_ptr() as _,
                        );
                    }
                }
                Step::Transfer(_) => {
                    let f = self.cuda.transfer_fns.get(&idx).ok_or_else(|| {
                        SessionError::OtherError(format!("Transfer step {idx} not loaded"))
                    })?;
                    unsafe {
                        (*f)(
                            self.cuda.state,
                            output_ptrs.as_ptr(),
                            input_ptrs.as_ptr() as _,
                        );
                    }
                }
                Step::SyncWait(_) => {}
            }
        }

        unsafe { (self.cuda.sync_func)() };

        let outputs = output_bufs
            .into_iter()
            .zip(self.output_ty.iter())
            .map(|(buf, ty)| buf.into_tensor(ty.dims.clone()))
            .collect();
        Ok(outputs)
    }
}
