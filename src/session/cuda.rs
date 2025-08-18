use crate::codegen::cuda::*;
use crate::codegen::*;
use crate::schedule::Schedule;
use crate::session::{SessionError, StrictTensor};
use crate::tensor::{types::ResolvedTensorType, Tensor};
use std::path::PathBuf;

use std::io::BufWriter;
use std::io::Write;
use tempfile::TempDir;

use itertools::zip_eq;

use std::process::Command;

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8, *const *const u8);

const PERSIST: bool = true;

pub struct SessionCUDA {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<StrictTensor>,

    #[allow(dead_code)]
    tmp_dir: Option<TempDir>,

    #[allow(dead_code)]
    lib: libloading::Library,
    func: CodeType,
}

impl SessionCUDA {
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer: Vec<StrictTensor>,
        schedule: Schedule,
    ) -> Result<Self, SessionError> {
        let mut hostcode_gen = HostCodeGenerator::new(&schedule);
        let hostcode = hostcode_gen
            .generate()
            .map_err(CodeGenError::CudaBuildError)
            .map_err(SessionError::CodeGenError)?;

        let tmp_dir = TempDir::with_prefix("my_model_")
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let main_file = tmp_dir.path().join("main.cu");
        let mut writer = std::fs::File::create(&main_file)
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))
            .map(BufWriter::new)?;
        hostcode
            .write(&mut writer)
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        writer
            .flush()
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        println!("Generated");

        let shared_obj = tmp_dir.path().join("libmodel.so");

        let kernel_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/codegen/cuda/kernels");

        Command::new("nvcc")
            .args([
                main_file.to_str().unwrap(),
                format!("-I{}", kernel_dir.to_str().unwrap()).as_str(),
                "--shared",
                "-o",
                shared_obj.to_str().unwrap(),
                "-lcudnn",
                "--compiler-options",
                "'-fPIC'",
            ])
            // .args(objs.iter().map(|p| p.to_str().unwrap()))
            .status()
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        dbg!(&tmp_dir);
        let tmp_dir = if PERSIST {
            let _ = tmp_dir.into_path();
            None
        } else {
            Some(tmp_dir)
        };

        println!("Compiled");

        let lib = unsafe { libloading::Library::new(shared_obj.as_os_str()) }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func: libloading::Symbol<CodeType> = unsafe { lib.get(b"model") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func = *func;

        println!("Loaded");

        Ok(Self {
            input_ty,
            output_ty,
            tmp_dir,
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
