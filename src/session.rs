mod cpu;

use crate::codegen::CodeGenError;
use crate::onnx::load::*;
use crate::onnx::model::{Graph, Model, ValueId};
use crate::schedule::Schedule;
use crate::session::cpu::SessionCPU;
use crate::tensor::{
    data::TensorData,
    dimensions::ResolvedTensorDims,
    types::{DataType, FloatType, ResolvedTensorType, SIntType, TypeError, UIntType},
    Tensor,
};
use crate::transform::{transform_graph, Options};

use tempfile::TempDir;

use rayon::prelude::*;

use itertools::zip_eq;

use inkwell::context::Context;
use inkwell::targets::FileType;
use std::fs::File;
use std::io::Write;
use std::path::Path;

enum StrictTensor {
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
        match ty {
            DataType::SInt(SIntType::I32) => StrictTensor::I32(vec![0; dims.size()]),
            DataType::SInt(SIntType::I64) => StrictTensor::I64(vec![0; dims.size()]),
            DataType::UInt(UIntType::U64) => StrictTensor::U64(vec![0; dims.size()]),
            DataType::Float(FloatType::F32) => StrictTensor::F32(vec![0.0; dims.size()]),
            DataType::Float(FloatType::F64) => StrictTensor::F64(vec![0.0; dims.size()]),
        }
    }

    fn as_ptr(&self) -> *const u8 {
        match self {
            StrictTensor::I32(v) => v.as_ptr() as *const u8,
            StrictTensor::I64(v) => v.as_ptr() as *const u8,
            StrictTensor::U64(v) => v.as_ptr() as *const u8,
            StrictTensor::F32(v) => v.as_ptr() as *const u8,
            StrictTensor::F64(v) => v.as_ptr() as *const u8,
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            StrictTensor::I32(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::I64(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::U64(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::F32(v) => v.as_mut_ptr() as *mut u8,
            StrictTensor::F64(v) => v.as_mut_ptr() as *mut u8,
        }
    }

    fn into_tensor(self, dims: ResolvedTensorDims) -> Tensor {
        match self {
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

#[derive(Clone, Copy)]
pub enum Target {
    CPU,
}

pub enum Session {
    CPU(SessionCPU),
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
        target: Target,
    ) -> Result<Self, SessionError> {
        let mut model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        if let Some(input_ty) = input_ty {
            model
                .graph
                .resolve_input_types(input_ty)
                .map_err(SessionError::TypeError)?;
        }

        transform_graph(&mut model.graph, options);

        // TODO: remove
        // Self::_write_model(&model.graph, "model.dot");
        // panic!("a");

        if false {
            model
                .save_to_path("model.onnx")
                .map_err(|e| SessionError::OtherError(format!("Failed to save model: {:?}", e)))?;
        }

        let inputs_ty = get_argument_types(&model.graph, &model.graph.input_values())?;
        let outputs_ty = get_argument_types(&model.graph, &model.graph.output_values())?;
        let initializer: Vec<_> = model
            .graph
            .initializer
            .values()
            .map(StrictTensor::from)
            .collect::<Vec<_>>();

        let mut schedule = Schedule::new(model.graph);
        schedule.assign_mem();
        schedule.annotate_omp(options.omp_threshold); // TODO: Move to SessionCPU

        match target {
            Target::CPU => {
                SessionCPU::new(inputs_ty, outputs_ty, initializer, schedule).map(Session::CPU)
            }
        }
    }

    // TODO: Type check
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        match self {
            Session::CPU(session) => session.run(inputs),
        }
    }
}
