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
use crate::session::DeviceBuffer;
use crate::session::InitializerSource;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

type KernelFn = unsafe extern "C" fn(*const *mut u8);

struct CpuJitState {
    _engine: ExecutionEngine<'static>,
    _contexts: Vec<Context>,
}

unsafe impl Send for CpuJitState {}
unsafe impl Sync for CpuJitState {}

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
}

impl SessionHybrid {
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer: Vec<InitializerSource>,
        session_state_buffers: Vec<Arc<DeviceBuffer>>,
        schedule: Schedule,
        opt: &Options,
        _build_dir: &Path,
    ) -> Result<Self, SessionError> {
        let _ = opt;
        let initializer: Vec<StrictTensor> = initializer
            .into_iter()
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

        for step in &plan.steps {
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
                    return Err(SessionError::OtherError(format!(
                        "non-CPU kernel dispatch not yet supported (device={:?})",
                        k.context.device
                    )));
                }
                Step::Transfer(_) => {
                    return Err(SessionError::OtherError(
                        "Transfer step dispatch not yet supported in hybrid runtime".to_string(),
                    ));
                }
                Step::SyncWait(_) => {}
            }
        }

        let outputs = output_bufs
            .into_iter()
            .zip(self.output_ty.iter())
            .map(|(buf, ty)| buf.into_tensor(ty.dims.clone()))
            .collect();
        Ok(outputs)
    }
}
