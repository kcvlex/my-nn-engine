use std::collections::HashMap;
use std::sync::Arc;

use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::OptimizationLevel;
use itertools::Itertools;
use log::info;
use rayon::prelude::*;

use crate::codegen::cpu::get_kernel_name_or;
use crate::codegen::cpu::CodeGenContext as CpuCodeGenContext;
use crate::schedule::ir::AllocPlace;
use crate::schedule::ir::ArenaId;
use crate::schedule::ir::Device;
use crate::schedule::ir::ExecutionPlan;
use crate::schedule::ir::MemoryTier;
use crate::schedule::ir::Step;
use crate::schedule::ChunkId;
use crate::schedule::KernelId;
use crate::schedule::Schedule;
use crate::session::cpu::CpuJitState;
use crate::session::shared_lib::load_jit_runtime;
use crate::session::DeviceBuffer;
use crate::session::InitializerSource;
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

pub struct SessionHybrid {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,

    schedule: Schedule,
    initializer: Vec<StrictTensor>,
    #[allow(dead_code)]
    session_state_buffers: Vec<Arc<DeviceBuffer>>,

    cpu_kernel_fns: HashMap<KernelId, u64>,
    #[allow(dead_code)]
    cpu_jit: CpuJitState,

    host_arenas: HostArenas,
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
        initializer: Vec<InitializerSource>,
        session_state_buffers: Vec<Arc<DeviceBuffer>>,
        schedule: Schedule,
    ) -> Result<Self, SessionError> {
        let initializer: Vec<StrictTensor> = initializer
            .into_iter()
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
