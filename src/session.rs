mod cpu;
mod cuda;

use std::path::Path;
use std::path::PathBuf;

use log::info;
use tempfile::TempDir;

use crate::codegen::CodeGenError;
use crate::onnx::load::*;
use crate::onnx::model::Graph;
use crate::onnx::model::Model;
use crate::onnx::model::ValueId;
use crate::options::*;
use crate::schedule::create_schedule_passes;
use crate::schedule::Schedule;
use crate::session::cpu::SessionCPU;
use crate::session::cuda::SessionCUDA;
use crate::tensor::data::TensorData;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::TypeError;
use crate::tensor::types::UIntType;
use crate::tensor::Tensor;
use crate::transform::transform_graph;

enum StrictTensor {
    U8(Vec<u8>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

macro_rules! cast_vec {
    ($data: expr, $ty: ty) => {{
        $data.iter().map(|x| *x as $ty).collect::<Vec<_>>()
    }};
}

impl StrictTensor {
    fn zeros(ty: DataType, dims: &ResolvedTensorDims) -> Self {
        let sz = dims.size().max(1);
        match ty {
            DataType::Bool | DataType::UInt(UIntType::U8) => StrictTensor::U8(vec![0; sz]),
            DataType::SInt(SIntType::I32) => StrictTensor::I32(vec![0; sz]),
            DataType::SInt(SIntType::I64) => StrictTensor::I64(vec![0; sz]),
            DataType::UInt(UIntType::U64) => StrictTensor::U64(vec![0; sz]),
            DataType::Float(FloatType::F32) => StrictTensor::F32(vec![0.0; sz]),
            DataType::Float(FloatType::F64) => StrictTensor::F64(vec![0.0; sz]),
        }
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            StrictTensor::U8(v) => v.as_ptr() as *const u8,
            StrictTensor::I32(v) => v.as_ptr() as *const u8,
            StrictTensor::I64(v) => v.as_ptr() as *const u8,
            StrictTensor::U64(v) => v.as_ptr() as *const u8,
            StrictTensor::F32(v) => v.as_ptr() as *const u8,
            StrictTensor::F64(v) => v.as_ptr() as *const u8,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            StrictTensor::U8(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::I32(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::I64(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::U64(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::F32(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::F64(v) => v.as_mut_ptr() as *mut u8,
        }
    }

    fn into_tensor(self, dims: ResolvedTensorDims) -> Tensor {
        match self {
            StrictTensor::U8(v) => Tensor::new(
                dims,
                TensorData::Bool(v.iter().map(|&b| if b != 0 { 1u8 } else { 0u8 }).collect()),
            )
            .unwrap(),
            StrictTensor::I32(v) => {
                Tensor::new(dims, TensorData::SInt(SIntType::I32, cast_vec!(v, i64))).unwrap()
            }
            StrictTensor::I64(v) => Tensor::new(dims, TensorData::SInt(SIntType::I64, v)).unwrap(),
            StrictTensor::U64(v) => Tensor::new(dims, TensorData::UInt(UIntType::U64, v)).unwrap(),
            StrictTensor::F32(v) => {
                Tensor::new(dims, TensorData::Float(FloatType::F32, cast_vec!(v, f64))).unwrap()
            }
            StrictTensor::F64(v) => {
                Tensor::new(dims, TensorData::Float(FloatType::F64, v)).unwrap()
            }
        }
    }
}

impl From<&Tensor> for StrictTensor {
    fn from(t: &Tensor) -> Self {
        match &t.data {
            TensorData::Bool(v) => StrictTensor::U8(v.clone()),
            TensorData::UInt(UIntType::U8, v) => StrictTensor::U8(cast_vec!(v, u8)),
            TensorData::SInt(SIntType::I32, v) => StrictTensor::I32(cast_vec!(v, i32)),
            TensorData::SInt(SIntType::I64, v) => StrictTensor::I64(v.clone()),
            TensorData::UInt(UIntType::U64, v) => StrictTensor::U64(v.clone()),
            TensorData::Float(FloatType::F32, v) => StrictTensor::F32(cast_vec!(v, f32)),
            TensorData::Float(FloatType::F64, v) => StrictTensor::F64(v.clone()),
        }
    }
}

#[derive(Debug)]
pub enum SessionError {
    CodeGenError(CodeGenError),
    ModelLoadError(ModelLoadError),
    TypeError(TypeError),
    OtherError(String),
}

unsafe impl Send for SessionError {}

pub enum Session {
    CPU(SessionCPU),
    CUDA(SessionCUDA),
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
        input_ty: Option<&[ResolvedTensorType]>,
        options: &Options,
    ) -> Result<Self, SessionError> {
        let _ = env_logger::try_init();
        info!("Session starting");

        let mut model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        info!("Model loaded");

        if let Some(input_ty) = input_ty {
            model
                .graph
                .resolve_input_types(input_ty)
                .map_err(SessionError::TypeError)?;
        }

        transform_graph(&mut model.graph, options);
        info!("Transformed");

        let tmp_dir = TempDir::with_prefix("my_model_")
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        info!("Build directory: {:?}", tmp_dir.path());
        let build_dir = PathBuf::from(tmp_dir.path());

        if options.save_transformed_model {
            let path = build_dir.join("transformed.onnx");
            model
                .save_to_path(&path)
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
            info!("Transformed model saved to {:?}", path);
        }

        let inputs_ty = get_argument_types(&model.graph, &model.graph.input_values())?;
        let outputs_ty = get_argument_types(&model.graph, &model.graph.output_values())?;
        let initializer_ids = model.graph.initializer_ids();
        let initializer: Vec<_> = initializer_ids
            .iter()
            .map(|&id| {
                StrictTensor::from(
                    &model.graph.get_initializer(id).expect("initializer missing"),
                )
            })
            .collect::<Vec<_>>();

        let mut schedule = Schedule::new(model.graph, options.clone());
        let schedule_passes = create_schedule_passes(options);
        schedule_passes.run(&mut schedule);
        info!("Scheduled");

        if options.save_build_dir {
            let path = tmp_dir.keep();
            info!("Build directory saved at {:?}", path);
        }

        match options.target {
            Target::CPU => SessionCPU::new(
                inputs_ty,
                outputs_ty,
                initializer,
                schedule,
                options,
                &build_dir,
            )
            .map(Session::CPU),
            Target::CUDA => SessionCUDA::new(
                inputs_ty,
                outputs_ty,
                initializer,
                schedule,
                options,
                &build_dir,
            )
            .map(Session::CUDA),
        }
    }

    // TODO: Type check
    pub fn run(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        match self {
            Session::CPU(session) => session.run(inputs),
            Session::CUDA(session) => session.run(inputs),
        }
    }
}
