use crate::codegen::{CodeGenContext, CodeGenError};
use crate::onnx::load::*;
use crate::onnx::model::{Graph, Model, ValueId};
use crate::optimize::{
    batchnorm, gemm, identity, im2col, infer, normalize, omp,
    optimizer::{Optimizer, SimpleGraphModifier},
    reduce,
};
use crate::tensor::{
    dimensions::ResolvedTensorDims,
    types::{ResolvedTensorType, TypeError},
    Tensor,
};

use tempfile::TempDir;

use rayon::prelude::*;

use inkwell::context::Context;
use inkwell::targets::FileType;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use std::process::Command;

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8, *const *const u8);

const DEBUG: bool = true;

#[derive(Debug)]
pub enum SessionError {
    CodeGenError(CodeGenError),
    ModelLoadError(ModelLoadError),
    TypeError(TypeError),
    OtherError(String),
}

pub struct Session {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<Tensor>,

    #[allow(dead_code)]
    codegen_ctx: CodeGenContext,

    #[allow(dead_code)]
    tmp_dir: Option<TempDir>,

    #[allow(dead_code)]
    lib: libloading::Library,
    func: CodeType,
}

fn get_argument_types(
    graph: &Graph,
    values: &[ValueId],
) -> Result<Vec<ResolvedTensorType>, SessionError> {
    values
        .iter()
        .map(|&id| graph.get_resolved_tensor_type(id).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or(SessionError::TypeError(TypeError::UnresolvedInput))
}

impl Session {
    pub fn new<P: AsRef<Path>>(
        p: P,
        input_ty: Option<&[&ResolvedTensorDims]>,
        omp_threshold: usize,
    ) -> Result<Self, SessionError> {
        let mut model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        if let Some(input_ty) = input_ty {
            model
                .graph
                .resolve_input_types(input_ty)
                .map_err(SessionError::TypeError)?;
        }
        let mut optimizer = Optimizer::<SimpleGraphModifier>::new(String::from("optimizer"));

        optimizer
            .passes
            .push(Box::new(normalize::ContigousOutput::default()));
        optimizer
            .passes
            .push(Box::new(infer::ShapeInference::default()));
        optimizer
            .passes
            .push(Box::new(im2col::InsertIm2Col::default()));
        optimizer
            .passes
            .push(Box::new(batchnorm::DecomposeBatchNormalization::default()));
        optimizer
            .passes
            .push(Box::new(reduce::Reduce2ReduceMatrix::default()));
        optimizer
            .passes
            .push(Box::new(normalize::EliminateGlobalAvgPool::default()));
        optimizer
            .passes
            .push(Box::new(gemm::MatMul2Gemm::default()));
        optimizer
            .passes
            .push(Box::new(gemm::TransformBLASGemm::default()));
        optimizer
            .passes
            .push(Box::new(gemm::GemmTransComposition::default()));
        optimizer.passes.push(Box::new(omp::InnermostOMP {
            threshold: omp_threshold,
        }));
        optimizer
            .passes
            .push(Box::new(identity::Ops2Identity::default()));
        optimizer.run(&mut model.graph);

        // TODO: remove
        Self::_write_model(&model.graph, "model.dot");
        //panic!("a");

        let inputs_ty = get_argument_types(&model.graph, &model.graph.input_values())?;
        let outputs_ty = get_argument_types(&model.graph, &model.graph.output_values())?;
        let initializer: Vec<_> = model
            .graph
            .initializer
            .values()
            .cloned()
            .collect::<Vec<_>>();

        let codegen_ctx = CodeGenContext::new(model.graph).map_err(SessionError::CodeGenError)?;
        let (codegens, mut contexts): (Vec<_>, Vec<_>) = codegen_ctx
            .all_necessary_nodes()
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
                codegen_ctx.new_codegen_for_node(id, ll_ctx)
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
            .into_iter()
            .enumerate()
            .map(|(i, codegen)| {
                let path = tmp_dir.path().join(format!("model_{i}.o"));
                (path, codegen)
            })
            .collect::<Vec<_>>();

        println!("Compiling");
        let objs = codegens
            .into_par_iter()
            .map(|(path, codegen)| {
                codegen.compile().unwrap();
                if !DEBUG {
                    codegen.run_opt_aggressive().unwrap();
                }
                codegen.write_to_file(FileType::Object, &path).unwrap();
                if DEBUG {
                    let ll_path = path.with_extension("ll");
                    codegen.module().print_to_file(&ll_path).unwrap();
                }
                path
            })
            .collect::<Vec<_>>();
        println!("Compiled");

        let shared_obj = tmp_dir.path().join("model.so");

        // TODO: args
        Command::new("clang")
            .args([
                "-shared",
                "-fPIC",
                "-fopenmp",
                "-I/usr/include/openblas",
                "-lopenblas",
                "-o",
                shared_obj.to_str().unwrap(),
            ])
            .args(objs.iter().map(|p| p.to_str().unwrap()))
            .status()
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        println!("Generated");

        let lib = unsafe { libloading::Library::new(shared_obj.as_os_str()) }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func: libloading::Symbol<CodeType> = unsafe { lib.get(b"main") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func = *func;

        println!("Loaded");

        let tmp_dir = if DEBUG {
            let _ = tmp_dir.into_path();
            None
        } else {
            Some(tmp_dir)
        };

        Ok(Session {
            input_ty: inputs_ty,
            output_ty: outputs_ty,
            codegen_ctx,
            tmp_dir,
            lib,
            func,
            initializer,
        })
    }

    // TODO: Type check
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        let mut outputs = self
            .output_ty
            .iter()
            .map(|ty| Tensor::zeros(ty.elem_type, ty.dims.clone()))
            .collect::<Vec<_>>();
        let output_ptrs = outputs
            .iter_mut()
            .map(|t| t.data.as_mut_ptr())
            .collect::<Vec<_>>();
        let input_ptrs = inputs.iter().map(|t| t.data.as_ptr()).collect::<Vec<_>>();
        let initializer_ptrs = self
            .initializer
            .iter()
            .map(|t| t.data.as_ptr())
            .collect::<Vec<_>>();
        unsafe {
            (self.func)(
                output_ptrs.as_ptr(),
                input_ptrs.as_ptr(),
                initializer_ptrs.as_ptr(),
            )
        };
        Ok(outputs)
    }

    fn _write_model<P: AsRef<Path>>(graph: &Graph, p: P) {
        let mut file = File::create(p).unwrap();
        file.write_all(graph.to_dot().as_bytes()).unwrap();
    }

    pub fn write_model<P: AsRef<Path>>(&self, p: P) {
        Self::_write_model(&self.codegen_ctx.graph, p);
    }

    pub fn persistent(&mut self) -> Result<(), SessionError> {
        if let Some(tmp_dir) = self.tmp_dir.take() {
            let _ = tmp_dir.into_path();
        }
        Ok(())
    }
}
