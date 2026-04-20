use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use itertools::zip_eq;
use log::info;
use rayon::prelude::*;

use crate::codegen::cuda::*;
use crate::codegen::*;
use crate::options::Options;
use crate::schedule::Schedule;
use crate::session::SessionError;
use crate::session::StrictTensor;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::Tensor;

type InitType = unsafe extern "C" fn() -> *mut std::ffi::c_void;
type RunType = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const *mut u8,
    *const *const u8,
    *const *const u8,
);
type DestroyType = unsafe extern "C" fn(*mut std::ffi::c_void);

pub struct SessionCUDA {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<StrictTensor>,

    #[allow(dead_code)]
    lib: libloading::Library,
    init_func: InitType,
    run_func: RunType,
    destroy_func: DestroyType,
    state: *mut std::ffi::c_void,
}

impl SessionCUDA {
    pub(super) fn new(
        input_ty: Vec<ResolvedTensorType>,
        output_ty: Vec<ResolvedTensorType>,
        initializer: Vec<StrictTensor>,
        schedule: Schedule,
        opt: &Options,
        build_dir: &Path,
    ) -> Result<Self, SessionError> {
        let mut hostcode_gen = HostCodeGenerator::new(&schedule);
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
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

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
                    .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
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
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        info!("Compiled");

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
            initializer,
        })
    }

    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        if self.state.is_null() {
            self.state = unsafe { (self.init_func)() };
        }
        let initializer_ptrs = self
            .initializer
            .iter()
            .map(|t| t.as_ptr())
            .collect::<Vec<_>>();
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
        unsafe {
            (self.run_func)(
                self.state,
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

unsafe impl Send for SessionCUDA {}
// TODO: Sync is unsound — concurrent run() calls would race on state.
// Either protect with Mutex on the caller side, or remove Sync and use Mutex<Session> in my-onnx-ui.
unsafe impl Sync for SessionCUDA {}

impl Drop for SessionCUDA {
    fn drop(&mut self) {
        if !self.state.is_null() {
            unsafe {
                (self.destroy_func)(self.state);
            }
        }
    }
}
