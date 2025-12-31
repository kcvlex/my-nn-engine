mod cpu;
mod cuda;

use std::path::Path;

use crate::codegen::CodeGenError;
use crate::onnx::load::*;
use crate::onnx::model::Graph;
use crate::onnx::model::Model;
use crate::onnx::model::ValueId;
use crate::options::*;
use crate::schedule::Schedule;
use crate::session::cpu::SessionCPU;
use crate::session::cuda::SessionCUDA;
use crate::tensor::data::TensorData;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::TypeError;
use crate::tensor::types::UIntType;
use crate::tensor::Tensor;
use crate::transform::transform_graph;

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

        let mut schedule = Schedule::new(model.graph, options.clone());
        schedule.assign_mem();
        schedule.annotate_omp(options.omp_threshold); // TODO: Move to SessionCPU

        match options.target {
            Target::CPU => {
                SessionCPU::new(inputs_ty, outputs_ty, initializer, schedule).map(Session::CPU)
            }
            Target::CUDA => {
                SessionCUDA::new(inputs_ty, outputs_ty, initializer, schedule).map(Session::CUDA)
            }
        }
    }

    // TODO: Type check
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        match self {
            Session::CPU(session) => session.run(inputs),
            Session::CUDA(session) => session.run(inputs),
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::izip;

    use super::*;
    use crate::session::Session;
    use crate::tensor::data::CompPolicy;
    use crate::tensor::Tensor;

    macro_rules! make_tensor {
        ($ty: ty, $($expr: expr,)*) => {{
            let orig: ndarray::Array<$ty, _> = ndarray::array!($($expr,)*);
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .into_dyn()
                .try_into()
                .map(|t| (t, orig.clone()))
                .map_err(SessionError::TypeError);
            res
        }};
    }

    macro_rules! make_tensor_3x2x4 {
        () => {{
            make_tensor!(
                f32,
                [
                    [-0.7736, 1.1965, 0.6127, 1.7081],
                    [-0.1194, 0.2656, -0.3478, 0.0629],
                ],
                [
                    [0.1489, -0.4435, -0.9640, -1.7148],
                    [0.8480, 0.5366, -0.0574, -0.5479],
                ],
                [
                    [0.5928, -1.7610, 1.4378, 0.0],
                    [0.2030, 0.0264, 1.3788, f32::INFINITY],
                ],
            )
        }};
    }

    macro_rules! make_tensor_3x2x4_pos {
        () => {{
            make_tensor!(
                f32,
                [
                    [0.7736, 1.1965, 0.6127, 1.7081],
                    [0.1194, 0.2656, 0.3478, 0.0629],
                ],
                [
                    [0.1489, 0.4435, 0.9640, 1.7148],
                    [0.8480, 0.5366, 0.0574, 0.5479],
                ],
                [
                    [0.5928, 1.7610, 1.4378, 1.8061],
                    [0.2030, 0.0264, 1.3788, f32::INFINITY],
                ],
            )
        }};
    }

    macro_rules! make_range_tensor {
        ($ty: ty, $($dim: expr),*) => {{
            let len = [$($dim),*].iter().product();
            let orig = ndarray::Array::from_iter((0..len).map(|x| x as $ty))
                .into_shape_with_order(($($dim),*))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .into_dyn()
                .try_into()
                .map(|t| (t, orig.clone()))
                .map_err(SessionError::TypeError);
            res
        }};
    }

    macro_rules! tensor_assert_eq {
        ($left: expr, $right: expr) => {{
            let right = Tensor::try_from($right).map_err(SessionError::TypeError)?;
            assert_eq!($left, right);
        }};
    }

    macro_rules! assert_eq_epsilon {
        ($left: expr, $right: expr, $epsilon: expr) => {{
            let res = $left.eq_with_epsilon(&$right, $epsilon, CompPolicy::Either);
            if !res {
                // For pretty print
                assert_eq!($left, $right);
            }
        }};
    }

    fn with_session<P, F>(p: P, targets: &[Target], f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: Fn(Session) -> TestResult,
    {
        use std::path::PathBuf;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/operator")
            .join(p);
        for target in targets.iter().copied() {
            let opt = match target {
                Target::CPU => Options::builder().target(target).omp_threshold(10).build(),
                Target::CUDA => Options::builder().target(target).build(),
            };
            let session = Session::new(&path, None, &opt)?;
            f(session)?;
        }
        Ok(())
    }

    fn with_session_and_tensors<P, F>(dir: P, targets: &[Target], f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: Fn(Session, (Tensor, Tensor)) -> TestResult,
    {
        use std::path::PathBuf;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/operator")
            .join(dir);
        let input = Tensor::load_from_path(dir.join("input.pb"))
            .map_err(|e| SessionError::OtherError(format!("Failed to load input: {:?}", e)))?;
        let output = Tensor::load_from_path(dir.join("output.pb"))
            .map_err(|e| SessionError::OtherError(format!("Failed to load output: {:?}", e)))?;

        for target in targets.iter().copied() {
            let opt = match target {
                Target::CPU => Options::builder().target(target).omp_threshold(10).build(),
                Target::CUDA => Options::builder().target(target).build(),
            };
            let session = Session::new(dir.join("model.onnx"), None, &opt)?;
            f(session, (input.clone(), output.clone()))?;
        }
        Ok(())
    }

    fn with_cpu_session<P, F>(p: P, f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: Fn(Session) -> TestResult,
    {
        with_session(p, &[Target::CPU], f)
    }

    fn with_all_sessions<P, F>(p: P, f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: Fn(Session) -> TestResult,
    {
        with_session(p, &[Target::CPU, Target::CUDA], f)
    }

    fn with_all_sessions_and_tensors<P, F>(p: P, f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: Fn(Session, (Tensor, Tensor)) -> TestResult,
    {
        with_session_and_tensors(p, &[Target::CPU, Target::CUDA], f)
    }

    type TestResult = Result<(), SessionError>;

    trait Sigmoid {
        fn sigmoid(self) -> Self;
    }

    impl Sigmoid for f32 {
        fn sigmoid(self) -> Self {
            1.0 / (1.0 + (-self).exp())
        }
    }

    #[test]
    fn add() -> TestResult {
        with_all_sessions("add.onnx", |session| {
            let (input0, orig0) = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],)?;
            let (input1, orig1) = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0],)?;
            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], (orig0 + orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn add_large() -> TestResult {
        with_all_sessions("add_large.onnx", |session| {
            let (input0, orig0) = make_tensor!(
                f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
            )?;
            let (input1, orig1) = make_tensor!(
                f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
            )?;
            let outputs = session.run(&[input0, input1])?;
            tensor_assert_eq!(outputs[0], (orig0 + orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn add_broadcast() -> TestResult {
        with_all_sessions("add_broadcast.onnx", |session| {
            // (1 x 4 x 5)
            let (input0, orig0) = make_tensor!(
                f32,
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                    [4.0, 5.0, 6.0, 7.0, 8.0],
                ],
            )?;

            // (2 x 3 x 1 x 1)
            let (input1, orig1) = make_tensor!(
                f32,
                [[[1.0]], [[2.0]], [[3.0]]],
                [[[1.0]], [[2.0]], [[3.0]]],
            )?;

            // (4 x 5)
            let (input2, orig2) = make_tensor!(
                f32,
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
                [40.0, 41.0, 42.0, 43.0, 44.0],
            )?;

            let output = session.run(&[input0, input1, input2])?;
            tensor_assert_eq!(output[0], (orig0 + orig1 + orig2).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn relu() -> TestResult {
        with_all_sessions("relu.onnx", |session| {
            let (input, orig) =
                make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],)?;
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], orig.mapv(|x| x.max(0.0)).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn transpose() -> TestResult {
        with_cpu_session("transpose.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 1, 7, 5, 1)?;
            let output = session.run(&[input])?;
            let expected = orig
                .view()
                .permuted_axes([2, 3, 1, 0])
                .to_owned()
                .into_dyn();
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn matmul() -> TestResult {
        with_all_sessions("matmul.onnx", |session| {
            let (input0, orig0) = make_tensor!(
                f32,
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            )?;
            let (input1, orig1) = make_tensor!(f32, [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],)?;
            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], orig0.dot(&orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn matmul_a_x_tb() -> TestResult {
        with_all_sessions("matmul_a_x_tb.onnx", |session| {
            let (input0, orig0) = make_range_tensor!(f32, 5, 7)?;
            let (input1, orig1) = make_range_tensor!(f32, 6, 7)?;

            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], orig0.dot(&orig1.t()).into_dyn());
            Ok(())
        })
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-32
    #[test]
    fn conv() -> TestResult {
        with_all_sessions("conv.onnx", |session| {
            // (1 x 1 x 5 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 5 x 5)
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [12.0, 21.0, 27.0, 33.0, 24.0],
                    [33.0, 54.0, 63.0, 72.0, 51.0],
                    [63.0, 99.0, 108.0, 117.0, 81.0],
                    [93.0, 144.0, 153.0, 162.0, 111.0],
                    [72.0, 111.0, 117.0, 123.0, 84.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_strides0() -> TestResult {
        with_all_sessions("conv_with_strides0.onnx", |session| {
            // (1 x 1 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 4 x 3)
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [12.0, 27.0, 24.0],
                    [63.0, 108.0, 81.0],
                    [123.0, 198.0, 141.0],
                    [112.0, 177.0, 124.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_strides1() -> TestResult {
        with_all_sessions("conv_with_strides1.onnx", |session| {
            // (1 x 1 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 3 x 2)
            let (expected, _) =
                make_tensor!(f32, [[[54.0, 72.0], [144.0, 162.0], [234.0, 252.0],]],)?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_strides2() -> TestResult {
        with_all_sessions("conv_with_strides2.onnx", |session| {
            // (1 x 1 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 4 x 2)
            let (expected, _) = make_tensor!(
                f32,
                [[[21.0, 33.0], [99.0, 117.0], [189.0, 207.0], [171.0, 183.0],]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_channels() -> TestResult {
        with_all_sessions("conv_channels.onnx", |session| {
            // (1 x 2 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0, 29.0],
                        [30.0, 31.0, 32.0, 33.0, 34.0],
                    ],
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0, 25.0],
                        [26.0, 27.0, 28.0, 29.0, 30.0],
                        [31.0, 32.0, 33.0, 34.0, 35.0],
                    ]
                ],
            )?;

            // (1 x 2 x 3 x 3)
            let (input1, _) = make_tensor!(
                f32,
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                ],
            )?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 4 x 3)
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [44.0, 93.0, 80.0],
                    [201.0, 342.0, 255.0],
                    [381.0, 612.0, 435.0],
                    [344.0, 543.0, 380.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_autopad_same() -> TestResult {
        with_all_sessions("conv_with_autopad_same.onnx", |session| {
            // (1 x 1 x 5 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;

            let output = session.run(&[input0, input1])?;

            let (expected, _) = make_tensor!(
                f32,
                [[[12.0, 27.0, 24.0], [63.0, 108.0, 81.0], [72.0, 117.0, 84.0],]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_bias() -> TestResult {
        with_all_sessions_and_tensors("conv_bias", |session, (input, output)| {
            let outputs = session.run(&[input])?;
            assert_eq_epsilon!(outputs[0], output, 1e-4);
            Ok(())
        })
    }

    #[test]
    fn maxpool() -> TestResult {
        with_all_sessions("maxpool.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 1, 3, 8, 8)?;
            let output = session.run(&[input])?;

            let expected = orig
                .windows((1, 1, 2, 2))
                .into_iter()
                .map(|w| w.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                .collect::<ndarray::Array<f32, _>>()
                .to_shape((1, 3, 7, 7))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn()
                .to_owned();
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn reducemax() -> TestResult {
        with_cpu_session("reducemax.onnx", |session| {
            let (input, orig) = make_tensor!(
                f32,
                [
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
            )?;
            let expected = orig
                .fold_axis(ndarray::Axis(2), f32::NEG_INFINITY, |&a, &b| a.max(b))
                .into_shape_with_order((3, 2))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn large_global_avg() -> TestResult {
        with_all_sessions_and_tensors("large_global_avg", |session, (input, output)| {
            let outputs = session.run(&[input])?;
            assert_eq_epsilon!(outputs[0], output, 1e-2);
            Ok(())
        })
    }

    #[test]
    fn global_avg_non_pow2() -> TestResult {
        with_all_sessions_and_tensors("global_avg_non_pow2", |session, (input, output)| {
            let outputs = session.run(&[input])?;
            assert_eq_epsilon!(outputs[0], output, 1e-2);
            Ok(())
        })
    }

    #[test]
    fn batchnorm() -> TestResult {
        with_all_sessions("batchnorm.onnx", |session| {
            let (input, _) = make_tensor!(
                f32,
                [
                    [[-0.7736, 1.1965], [0.6127, 1.7081]],
                    [[-1.2942, -0.1194], [0.2656, -0.3478]],
                    [[0.0629, 0.6267], [1.0625, -1.0402]],
                    [[0.9405, 0.8907], [-0.0534, -1.2017]]
                ],
                [
                    [[0.1489, -0.4435], [-0.9640, -1.7148]],
                    [[0.7103, 0.8480], [0.5366, -0.0574]],
                    [[-0.5479, -0.6636], [-0.8631, 1.0075]],
                    [[-0.3005, 0.8960], [0.9123, -0.5381]]
                ],
                [
                    [[0.5928, -1.7610], [1.4378, -1.8061]],
                    [[0.1554, 0.2030], [0.0264, 1.3788]],
                    [[0.0953, 2.1523], [-1.2667, 0.7831]],
                    [[-0.2522, 0.3387], [0.3128, 1.2057]]
                ],
                [
                    [[1.1783, 2.0076], [0.2719, 0.9309]],
                    [[0.2006, 0.3776], [0.7505, 0.2893]],
                    [[-0.3285, 2.2465], [1.1477, 1.3187]],
                    [[-0.4947, -0.3022], [-0.8595, -0.1885]]
                ],
            )?;

            let (expected, _) = make_tensor!(
                f32,
                [
                    [[2.6118e-01, 9.3978e-01], [7.3869e-01, 1.1160e+00]],
                    [[1.8126e-01, 7.8516e-01], [9.8307e-01, 6.6777e-01]],
                    [[6.2633e-01, 1.1128e+00], [1.4888e+00, -3.2538e-01]],
                    [[7.1125e-01, 6.9593e-01], [4.0556e-01, 5.2360e-02]]
                ],
                [
                    [[5.7894e-01, 3.7488e-01], [1.9563e-01, -6.2990e-02]],
                    [[1.2117e+00, 1.2825e+00], [1.1224e+00, 8.1704e-01]],
                    [[9.9338e-02, -5.0442e-04], [-1.7260e-01, 1.4413e+00]],
                    [[3.2954e-01, 6.9757e-01], [7.0256e-01, 2.5648e-01]]
                ],
                [
                    [[7.3184e-01, -7.8910e-02], [1.0229e+00, -9.4441e-02]],
                    [[9.2643e-01, 9.5088e-01], [8.6011e-01, 1.5553e+00]],
                    [[6.5432e-01, 2.4291e+00], [-5.2078e-01, 1.2477e+00]],
                    [[3.4440e-01, 5.2614e-01], [5.1817e-01, 7.9281e-01]]
                ],
                [
                    [[9.3350e-01, 1.2192e+00], [6.2131e-01, 8.4831e-01]],
                    [[9.4964e-01, 1.0406e+00], [1.2323e+00, 9.9528e-01]],
                    [[2.8862e-01, 2.5103e+00], [1.5623e+00, 1.7098e+00]],
                    [[2.6984e-01, 3.2904e-01], [1.5763e-01, 3.6400e-01]]
                ],
            )?;

            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 0.001);
            Ok(())
        })
    }

    #[test]
    fn leaky_relu() -> TestResult {
        with_all_sessions("leakyrelu.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4!()?;
            let alpha = 0.42;
            let expected = orig.map(|x| if *x < 0.0 { *x * alpha } else { *x });
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected.into_dyn());
            Ok(())
        })
    }

    #[test]
    fn exp() -> TestResult {
        with_all_sessions("exp.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4!()?;
            let expected = orig
                .map(|x| x.exp())
                .into_dyn()
                .try_into()
                .map_err(SessionError::TypeError)?;
            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 1e-6);
            Ok(())
        })
    }

    #[test]
    fn log() -> TestResult {
        with_all_sessions("log.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4_pos!()?;
            let expected = orig
                .map(|x| x.ln())
                .into_dyn()
                .try_into()
                .map_err(SessionError::TypeError)?;
            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 1e-6);
            Ok(())
        })
    }

    #[test]
    fn tanh() -> TestResult {
        with_all_sessions("tanh.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4!()?;
            let expected: Tensor = orig
                .map(|x| x.tanh())
                .into_dyn()
                .try_into()
                .map_err(SessionError::TypeError)?;
            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 1e-6);
            Ok(())
        })
    }

    #[test]
    fn sigmoid() -> TestResult {
        with_all_sessions("sigmoid.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4!()?;
            let expected: Tensor = orig
                .map(|x| x.sigmoid())
                .into_dyn()
                .try_into()
                .map_err(SessionError::TypeError)?;
            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 1e-6);
            Ok(())
        })
    }

    #[test]
    fn resize_downsample_sizes_nearest() -> TestResult {
        with_cpu_session("resize_downsample_sizes_nearest.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],]],)?;
            let (expected, _) = make_tensor!(f32, [[[1.0, 2.0, 4.0]]],)?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_scales_nearest() -> TestResult {
        with_cpu_session("resize_upsample_scales_nearest.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0], [3.0, 4.0],]],)?;
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                ]],
            )?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_scales_nearest_axes_2_3() -> TestResult {
        with_cpu_session("resize_upsample_scales_nearest_axes_2_3.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0], [3.0, 4.0],]],)?;
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                ]],
            )?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_scales_nearest_axes_3_2() -> TestResult {
        with_cpu_session("resize_upsample_scales_nearest_axes_3_2.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0], [3.0, 4.0],]],)?;
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                ]],
            )?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_sizes_nearest_axes_2_3() -> TestResult {
        with_cpu_session("resize_upsample_sizes_nearest_axes_2_3.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0], [3.0, 4.0],]],)?;
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                ]],
            )?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_sizes_nearest_axes_3_2() -> TestResult {
        with_cpu_session("resize_upsample_sizes_nearest_axes_3_2.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0], [3.0, 4.0],]],)?;
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                    [3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0],
                ]],
            )?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_sizes_nearest_ceil_half_pixel() -> TestResult {
        with_cpu_session(
            "resize_upsample_sizes_nearest_ceil_half_pixel.onnx",
            |session| {
                let (input, _) = make_tensor!(
                    f32,
                    [[
                        [1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0],
                        [9.0, 10.0, 11.0, 12.0],
                        [13.0, 14.0, 15.0, 16.0],
                    ]],
                )?;
                let (expected, _) = make_tensor!(
                    f32,
                    [[
                        [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0],
                        [5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0],
                        [5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 8.0],
                        [9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0],
                        [9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 12.0],
                        [13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0],
                        [13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0],
                        [13.0, 14.0, 14.0, 15.0, 15.0, 16.0, 16.0, 16.0],
                    ]],
                )?;
                let output = session.run(&[input])?;
                assert_eq!(output[0], expected);
                Ok(())
            },
        )
    }

    #[test]
    fn split_axis_2() -> TestResult {
        with_all_sessions("split_axis_2.onnx", |session| {
            let (input, _) = make_tensor!(
                f32,
                [[
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]],
            )?;
            let (expected0, _) =
                make_tensor!(f32, [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],]],)?;
            let (expected1, _) = make_tensor!(f32, [[[9.0, 10.0, 11.0, 12.0],]],)?;
            let (expected2, _) = make_tensor!(f32, [[[13.0, 14.0, 15.0, 16.0],]],)?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected0);
            assert_eq!(output[1], expected1);
            assert_eq!(output[2], expected2);
            Ok(())
        })
    }

    #[test]
    fn split_axis_3() -> TestResult {
        with_all_sessions("split_axis_3.onnx", |session| {
            let (input, _) = make_tensor!(
                f32,
                [[
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]],
            )?;
            let (expected0, _) =
                make_tensor!(f32, [[[1.0, 2.0], [5.0, 6.0], [9.0, 10.0], [13.0, 14.0],]],)?;
            let (expected1, _) = make_tensor!(f32, [[[3.0], [7.0], [11.0], [15.0],]],)?;
            let (expected2, _) = make_tensor!(f32, [[[4.0], [8.0], [12.0], [16.0],]],)?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected0);
            assert_eq!(output[1], expected1);
            assert_eq!(output[2], expected2);
            Ok(())
        })
    }

    #[test]
    fn concat_axis_2() -> TestResult {
        with_all_sessions("concat_axis_2.onnx", |session| {
            let (input0, _) = make_tensor!(f32, [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],]],)?;
            let (input1, _) = make_tensor!(f32, [[[9.0, 10.0, 11.0, 12.0],]],)?;
            let (input2, _) = make_tensor!(f32, [[[13.0, 14.0, 15.0, 16.0],]],)?;
            let output = session.run(&[input0, input1, input2])?;
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn bias_gemm() -> TestResult {
        with_all_sessions("bias_gemm.onnx", |session| {
            let (input0, orig0) = make_range_tensor!(f32, 4, 7)?;
            let (input1, orig1) = make_range_tensor!(f32, 7, 2)?;
            let bias = ndarray::array![[0.42, 0.63]];
            let mut expected = orig0.dot(&orig1);
            expected.scaled_add(0.5, &bias);
            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], expected.into_dyn());
            Ok(())
        })
    }

    #[test]
    fn squeeze() -> TestResult {
        with_all_sessions("squeeze.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 1, 2, 1, 3, 4)?;
            let expected = orig
                .into_shape_with_order((1, 2, 3, 4))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn squeeze_opt() -> TestResult {
        with_all_sessions("squeeze_opt.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 1, 2, 1, 3, 4)?;
            let expected = orig
                .into_shape_with_order((2, 3, 4))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn unsqueeze() -> TestResult {
        with_all_sessions("unsqueeze.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 2, 3, 4)?;
            let expected = orig
                .into_shape_with_order((1, 2, 3, 4, 1))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn reciprocal() -> TestResult {
        with_all_sessions("reciprocal.onnx", |session| {
            let (input, orig) = make_tensor!(f32, [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],)?;
            let expected = orig.map(|x| 1.0 / x).into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn sqrt() -> TestResult {
        with_all_sessions("sqrt.onnx", |session| {
            let (input, orig) = make_tensor!(f32, [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],)?;
            let expected = orig.map(|x| x.sqrt()).into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn pow_types() -> TestResult {
        let (x_i32, _) = make_tensor!(i32, 1, 2, 3,)?;
        let (x_i64, _) = make_tensor!(i64, 1, 2, 3,)?;
        let (x_u64, _) = make_tensor!(u64, 1, 2, 3,)?;
        let (x_f32, _) = make_tensor!(f32, 1.0, 2.0, 3.0,)?;
        let (x_f64, _) = make_tensor!(f64, 1.0, 2.0, 3.0,)?;

        let (y_i32, _) = make_tensor!(i32, 4, 5, 6,)?;
        let (y_i64, _) = make_tensor!(i64, 4, 5, 6,)?;
        let (y_u64, _) = make_tensor!(u64, 4, 5, 6,)?;
        let (y_f32, _) = make_tensor!(f32, 4.0, 5.0, 6.0,)?;
        let (y_f64, _) = make_tensor!(f64, 4.0, 5.0, 6.0,)?;

        let (z_i32, _) = make_tensor!(i32, 1, 32, 729,)?;
        let (z_i64, _) = make_tensor!(i64, 1, 32, 729,)?;
        let (z_u64, _) = make_tensor!(u64, 1, 32, 729,)?;
        let (z_f32, _) = make_tensor!(f32, 1.0, 32.0, 729.0,)?;
        let (z_f64, _) = make_tensor!(f64, 1.0, 32.0, 729.0,)?;

        let xv = [x_i32, x_i64, x_u64, x_f32, x_f64];
        let yv = [y_i32, y_i64, y_u64, y_f32, y_f64];
        let zv = [z_i32, z_i64, z_u64, z_f32, z_f64];
        let ty_lit = ["i32", "i64", "u64", "f32", "f64"];

        for (x, z, lty) in izip!(xv, zv, ty_lit) {
            for (y, rty) in izip!(yv.iter(), ty_lit) {
                let onnx = format!("pow_{}_{}.onnx", lty, rty);

                // TODO: Support integer types for CUDA.
                if lty.starts_with('f') && rty.starts_with('f') {
                    with_all_sessions(onnx, |session| {
                        let output = session.run(&[x.clone(), y.clone()])?;
                        assert_eq_epsilon!(output[0], z.clone(), 1e-6);
                        Ok(())
                    })?;
                } else {
                    with_cpu_session(onnx, |session| {
                        let output = session.run(&[x.clone(), y.clone()])?;
                        assert_eq!(output[0], z.clone());
                        Ok(())
                    })?;
                }
            }
        }

        Ok(())
    }

    #[test]
    fn sub() -> TestResult {
        with_all_sessions("sub.onnx", |session| {
            // (1 x 4 x 5)
            let (input0, orig0) = make_tensor!(
                f32,
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                    [4.0, 5.0, 6.0, 7.0, 8.0],
                ],
            )?;

            // (2 x 3 x 1 x 1)
            let (input1, orig1) = make_tensor!(
                f32,
                [[[1.0]], [[2.0]], [[3.0]]],
                [[[1.0]], [[2.0]], [[3.0]]],
            )?;

            // (4 x 5)
            let (input2, orig2) = make_tensor!(
                f32,
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
                [40.0, 41.0, 42.0, 43.0, 44.0],
            )?;

            let output = session.run(&[input0, input1, input2])?;
            tensor_assert_eq!(output[0], (orig0 - orig1 - orig2).into_dyn());
            Ok(())
        })
    }
}
