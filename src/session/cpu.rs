use std::path::Path;

use inkwell::context::Context;
use inkwell::execution_engine::ExecutionEngine;
use inkwell::support::load_library_permanently;
use inkwell::OptimizationLevel;
use itertools::zip_eq;
use log::info;
use rayon::prelude::*;

use crate::codegen::cpu::CodeGenContext;
use crate::options::Options;
use crate::schedule::Schedule;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8, *const *const u8);

/// SAFETY: After JIT compilation is complete, the engine is only used via a raw function pointer.
/// The engine and contexts are kept alive solely to prevent LLVM from deallocating the JIT code.
struct JitState {
    // engine must be dropped before contexts (field drop order guarantees this)
    _engine: ExecutionEngine<'static>,
    _contexts: Vec<Context>,
}

// SAFETY: Once JIT compilation is complete, the engine is not mutated and the compiled code
// is safe to call from any thread (it's just a function pointer into mmap'd memory).
unsafe impl Send for JitState {}
unsafe impl Sync for JitState {}

pub struct SessionCPU {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<StrictTensor>,

    #[allow(dead_code)]
    codegen_ctx: CodeGenContext,

    #[allow(dead_code)]
    jit: JitState,
    func: CodeType,
}

impl SessionCPU {
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer: Vec<StrictTensor>,
        schedule: Schedule,
        opt: &Options,
        build_dir: &Path,
    ) -> Result<Self, SessionError> {
        let codegen_ctx = CodeGenContext::new(schedule).map_err(SessionError::CodeGenError)?;
        let kernel_ids = codegen_ctx.all_necessary_kernels();

        let mut contexts: Vec<Context> = kernel_ids.iter().map(|_| Context::create()).collect();

        info!("Compiling");

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

        let kernel_modules: Vec<_> = codegens
            .into_iter()
            .enumerate()
            .map(|(i, cg)| {
                if opt.save_build_dir {
                    let ll_path = build_dir.join(format!("kernel_{}.ll", kernel_ids[i].index()));
                    cg.module().print_to_file(&ll_path).unwrap();
                }
                cg.into_module()
            })
            .collect();

        // Generate main module
        let main_context = Context::create();
        let main_codegen = codegen_ctx
            .new_codegen_for_main(&main_context, opt)
            .map_err(SessionError::CodeGenError)?;
        main_codegen.compile().map_err(SessionError::CodeGenError)?;
        main_codegen
            .run_opt_aggressive()
            .map_err(SessionError::CodeGenError)?;
        if opt.save_build_dir {
            let ll_path = build_dir.join("main.ll");
            main_codegen.module().print_to_file(&ll_path).unwrap();
        }
        let main_module = main_codegen.into_module();

        info!("Load external libraries");
        for lib in &["libopenblas.so", "libomp.so"] {
            load_library_permanently(Path::new(lib)).map_err(|e| {
                SessionError::OtherError(format!("Failed to load {}: {:?}", lib, e))
            })?;
        }

        info!("Linking and JIT compiling");
        let engine = main_module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(|e| {
                SessionError::OtherError(format!("Failed to create JIT engine: {:?}", e))
            })?;

        // Transmute engine to 'static before adding cross-context modules
        // SAFETY: _contexts outlives _engine by field drop order
        let engine: ExecutionEngine<'static> = unsafe { std::mem::transmute(engine) };

        // Add kernel modules (transmute lifetime to match engine's 'static)
        for module in kernel_modules.iter() {
            // SAFETY: corresponding context is stored in _contexts and outlives _engine
            engine
                .add_module(unsafe { std::mem::transmute(module) })
                .map_err(|()| {
                    SessionError::OtherError("Failed to add module to JIT engine".to_string())
                })?;
        }

        // Get function pointer
        let func_addr = engine.get_function_address("main").map_err(|e| {
            SessionError::OtherError(format!("Failed to get 'main' function: {:?}", e))
        })?;
        let func: CodeType = unsafe { std::mem::transmute(func_addr) };

        info!("JIT compiled and loaded");

        // Drop module objects to release borrows on contexts. The underlying LLVM modules are already owned by the engine (via create_jit_execution_engine and add_module).
        drop(kernel_modules);
        drop(main_module);
        contexts.push(main_context);

        Ok(Self {
            input_ty,
            output_ty,
            codegen_ctx,
            jit: JitState {
                _engine: engine,
                _contexts: contexts,
            },
            func,
            initializer,
        })
    }

    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
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
        let initializer_ptrs = self
            .initializer
            .iter()
            .map(|t| t.as_ptr())
            .collect::<Vec<_>>();
        unsafe {
            (self.func)(
                output_ptrs.as_ptr(),
                input_ptrs.as_ptr(),
                initializer_ptrs.as_ptr(),
            )
        };
        let outputs = zip_eq(self.output_ty.iter(), output_bufs)
            .map(|(ty, buf)| buf.into_tensor(ty.dims.clone()))
            .collect::<Vec<_>>();
        Ok(outputs)
    }
}
