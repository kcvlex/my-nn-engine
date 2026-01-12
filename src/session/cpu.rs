use std::process::Command;

use inkwell::context::Context;
use inkwell::targets::FileType;
use itertools::zip_eq;
use log::info;
use rayon::prelude::*;
use tempfile::TempDir;

use crate::codegen::cpu::CodeGenContext;
use crate::options::Options;
use crate::schedule::Schedule;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8, *const *const u8);

const DEBUG: bool = true;

pub struct SessionCPU {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<StrictTensor>,

    #[allow(dead_code)]
    codegen_ctx: CodeGenContext,

    #[allow(dead_code)]
    lib: libloading::Library,
    func: CodeType,
}

impl SessionCPU {
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer: Vec<StrictTensor>,
        schedule: Schedule,
        opt: &Options,
    ) -> Result<Self, SessionError> {
        let codegen_ctx = CodeGenContext::new(schedule).map_err(SessionError::CodeGenError)?;
        let (codegens, mut contexts): (Vec<_>, Vec<_>) = codegen_ctx
            .all_necessary_kernels()
            .iter()
            .copied()
            .map(|id| {
                let ll_ctx = Context::create();
                (id, ll_ctx)
            })
            .unzip();
        contexts.push(Context::create());
        let mut codegens = codegens
            .into_iter()
            .enumerate()
            .map(|(i, id)| {
                let ll_ctx = &contexts[i];
                codegen_ctx.new_codegen_for_kernel(id, ll_ctx)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(SessionError::CodeGenError)?;
        codegens.push({
            let ll_ctx = contexts.last().unwrap();
            codegen_ctx
                .new_codegen_for_main(ll_ctx)
                .map_err(SessionError::CodeGenError)?
        });

        let tmp_dir = TempDir::with_prefix("my_model_")
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let codegens = codegens
            .par_iter()
            .enumerate()
            .map(|(i, codegen)| {
                let path = tmp_dir.path().join(format!("model_{i}.o"));
                (path, codegen)
            })
            .collect::<Vec<_>>();

        dbg!(&tmp_dir);
        let shared_obj = tmp_dir.path().join("model.so");
        if opt.save_build_dir {
            let path = tmp_dir.into_path();
            info!("Build directory saved at {:?}", path);
        }

        info!("Compiling");
        let objs = codegens
            .into_par_iter()
            //.into_iter()
            .map(|(path, codegen)| {
                codegen.compile().unwrap();
                if !DEBUG {
                    codegen.run_opt_aggressive().unwrap();
                }
                if opt.save_build_dir {
                    let ll_path = path.with_extension("ll");
                    codegen.module().print_to_file(&ll_path).unwrap();
                }
                codegen.write_to_file(FileType::Object, &path).unwrap();
                path
            })
            .collect::<Vec<_>>();
        info!("Compiled");

        // TODO: args
        // TODO: remove -lm after llvm.tanh.* is available
        Command::new("clang")
            .args([
                "-shared",
                "-fPIC",
                "-fopenmp",
                "-I/usr/include/openblas",
                "-lopenblas",
                "-lm",
                "-o",
                shared_obj.to_str().unwrap(),
            ])
            .args(objs.iter().map(|p| p.to_str().unwrap()))
            .status()
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        info!("Generated");

        let lib = unsafe { libloading::Library::new(shared_obj.as_os_str()) }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func: libloading::Symbol<CodeType> = unsafe { lib.get(b"main") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func = *func;

        info!("Loaded");

        Ok(Self {
            input_ty,
            output_ty,
            codegen_ctx,
            lib,
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
