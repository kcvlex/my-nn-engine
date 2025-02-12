use crate::codegen::{CodeGenContext, CodeGenError};
use crate::onnx::load::*;
use crate::onnx::model::{Graph, Model, ValueId};
use crate::tensor::{
    data::TensorData,
    dimensions::ResolvedTensorDims,
    types::{DataType, FloatType, ResolvedTensorType, SIntType, TypeError, UIntType},
    Tensor,
};
use crate::transform::transform_graph;

use tempfile::TempDir;

use rayon::prelude::*;

use itertools::zip_eq;

use inkwell::context::Context;
use inkwell::targets::FileType;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use std::process::Command;

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8, *const *const u8);

const WRITE_LL: bool = true;
const DEBUG: bool = true;

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

pub struct Session {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<StrictTensor>,

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
        input_ty: Option<&[&ResolvedTensorType]>,
        omp_threshold: usize,
    ) -> Result<Self, SessionError> {
        let mut model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        if let Some(input_ty) = input_ty {
            model
                .graph
                .resolve_input_types(input_ty)
                .map_err(SessionError::TypeError)?;
        }

        transform_graph(&mut model.graph, omp_threshold);

        // TODO: remove
        //Self::_write_model(&model.graph, "model.dot");
        //panic!("a");

        let inputs_ty = get_argument_types(&model.graph, &model.graph.input_values())?;
        let outputs_ty = get_argument_types(&model.graph, &model.graph.output_values())?;
        let initializer: Vec<_> = model
            .graph
            .initializer
            .values()
            .map(StrictTensor::from)
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
            //.into_iter()
            .map(|(path, codegen)| {
                codegen.compile().unwrap();
                if !DEBUG {
                    codegen.run_opt_aggressive().unwrap();
                }
                if WRITE_LL {
                    let ll_path = path.with_extension("ll");
                    codegen.module().print_to_file(&ll_path).unwrap();
                }
                codegen.write_to_file(FileType::Object, &path).unwrap();
                path
            })
            .collect::<Vec<_>>();
        println!("Compiled");

        let shared_obj = tmp_dir.path().join("model.so");

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

        println!("Generated");

        let lib = unsafe { libloading::Library::new(shared_obj.as_os_str()) }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func: libloading::Symbol<CodeType> = unsafe { lib.get(b"main") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func = *func;

        println!("Loaded");

        let tmp_dir = if WRITE_LL {
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

#[cfg(test)]
mod test {
    use super::*;
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

    fn with_session<P, F>(path: P, f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: FnOnce(Session) -> TestResult,
    {
        use std::path::PathBuf;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/operator")
            .join(path);
        let session = Session::new(path, None, 10)?;
        f(session)?;
        Ok(())
    }

    type TestResult = Result<(), SessionError>;

    #[test]
    fn add() -> TestResult {
        with_session("add.onnx", |session| {
            let (input0, orig0) = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],)?;
            let (input1, orig1) = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0],)?;
            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], (orig0 + orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn add_large() -> TestResult {
        with_session("add_large.onnx", |session| {
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
        with_session("add_broadcast.onnx", |session| {
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
        with_session("relu.onnx", |session| {
            let (input, orig) =
                make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],)?;
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], orig.mapv(|x| x.max(0.0)).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn transpose() -> TestResult {
        with_session("transpose.onnx", |session| {
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
        with_session("matmul.onnx", |session| {
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
        with_session("matmul_a_x_tb.onnx", |session| {
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
        with_session("conv.onnx", |session| {
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
        with_session("conv_with_strides0.onnx", |session| {
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
        with_session("conv_with_strides1.onnx", |session| {
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
        with_session("conv_with_strides2.onnx", |session| {
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
        with_session("conv_channels.onnx", |session| {
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
        with_session("conv_with_autopad_same.onnx", |session| {
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
    fn maxpool() -> TestResult {
        with_session("maxpool.onnx", |session| {
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
        with_session("reducemax.onnx", |session| {
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
    fn batchnorm() -> TestResult {
        with_session("batchnorm.onnx", |session| {
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
        with_session("leakyrelu.onnx", |session| {
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
        with_session("exp.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4!()?;
            let expected = orig.map(|x| x.exp());
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected.into_dyn());
            Ok(())
        })
    }

    #[test]
    fn log() -> TestResult {
        with_session("log.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4_pos!()?;
            let expected = orig.map(|x| x.ln());
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected.into_dyn());
            Ok(())
        })
    }

    #[test]
    fn tanh() -> TestResult {
        with_session("tanh.onnx", |session| {
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
        with_session("sigmoid.onnx", |session| {
            let (input, orig) = make_tensor_3x2x4!()?;
            let expected: Tensor = orig
                .map(|x| {
                    let den = 1.0 + (-x).exp();
                    1.0 / den
                })
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
        with_session("resize_downsample_sizes_nearest.onnx", |session| {
            let (input, _) = make_tensor!(f32, [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],]],)?;
            let (expected, _) = make_tensor!(f32, [[[1.0, 2.0, 4.0]]],)?;
            let output = session.run(&[input])?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn resize_upsample_scales_nearest() -> TestResult {
        with_session("resize_upsample_scales_nearest.onnx", |session| {
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
        with_session("resize_upsample_scales_nearest_axes_2_3.onnx", |session| {
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
        with_session("resize_upsample_scales_nearest_axes_3_2.onnx", |session| {
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
        with_session("resize_upsample_sizes_nearest_axes_2_3.onnx", |session| {
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
        with_session("resize_upsample_sizes_nearest_axes_3_2.onnx", |session| {
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
        with_session(
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
        with_session("split_axis_2.onnx", |session| {
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
        with_session("split_axis_3.onnx", |session| {
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
        with_session("concat_axis_2.onnx", |session| {
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
        with_session("bias_gemm.onnx", |session| {
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
    fn transpose_conv2d() -> TestResult {
        with_session("transpose_conv2d.onnx", |session| {
            let (input, _) = make_tensor!(
                f32,
                [
                    [
                        [-3.2095e-01, -9.0422e-01, 3.1488e-01],
                        [-4.1095e-01, 7.6820e-01, -5.4613e-01],
                        [-2.3316e+00, 1.3857e+00, -7.5740e-01],
                        [2.8953e-01, 5.5962e-01, 3.2604e-01],
                        [-1.8044e-01, -7.4038e-01, 4.5303e-01]
                    ],
                    [
                        [2.4566e-01, -4.7064e-01, 3.9107e-01],
                        [-5.2738e-01, 6.9576e-01, 1.8397e+00],
                        [3.0494e-02, -1.1986e+00, 1.5420e+00],
                        [-4.8796e-01, 2.9630e-01, 1.6593e-01],
                        [-6.5490e-01, -4.0534e-01, -1.2149e+00]
                    ],
                    [
                        [1.4243e-01, -3.9638e-01, -1.7716e-01],
                        [1.9789e-03, -2.1054e+00, 2.6275e-01],
                        [2.2293e+00, 5.0398e-01, 1.2817e+00],
                        [8.5369e-01, 1.4235e+00, -5.2593e-01],
                        [-1.7966e+00, 1.6608e-01, 7.8946e-01]
                    ],
                    [
                        [3.6933e-01, -4.0664e-01, 5.9512e-01],
                        [-7.6952e-01, 1.1163e+00, -7.5076e-01],
                        [8.3995e-01, 3.9402e-01, 1.0629e+00],
                        [-1.1408e+00, -1.5816e+00, 6.3469e-01],
                        [-1.3892e+00, 5.5235e-01, 1.2791e-01]
                    ],
                    [
                        [5.6378e-01, -8.1214e-01, 2.7919e-01],
                        [2.0313e+00, 2.2484e+00, -4.7000e-01],
                        [6.4730e-02, -7.3608e-01, -8.2928e-01],
                        [-6.5936e-02, -2.3405e-01, -2.2370e+00],
                        [-9.8767e-01, -1.1777e+00, 3.7454e-01]
                    ]
                ],
                [
                    [
                        [4.9751e-01, 7.1751e-01, -3.1558e-01],
                        [3.1777e-01, -2.8072e-01, 7.0707e-01],
                        [-1.0877e+00, 1.2852e+00, -9.0810e-01],
                        [3.3413e-01, -4.7907e-01, -4.3840e-01],
                        [8.4484e-01, 9.9543e-01, 1.6256e+00]
                    ],
                    [
                        [5.6686e-01, 3.0577e-01, 1.4562e+00],
                        [1.0841e+00, -1.2680e+00, 2.1317e-01],
                        [-4.2257e-01, -1.5525e-01, -5.2615e-01],
                        [-1.0399e-02, -5.5959e-01, 5.0131e-01],
                        [1.1527e+00, 4.7749e-01, -9.3658e-01]
                    ],
                    [
                        [-9.2837e-02, 1.1496e-01, -9.7269e-01],
                        [-3.2708e-01, -2.0952e-02, 1.2174e+00],
                        [-1.5918e-01, -1.1992e-01, -2.7089e-01],
                        [-2.1716e-01, 7.2892e-01, 1.8506e-01],
                        [-2.7482e-01, -4.8260e-01, 5.6799e-01]
                    ],
                    [
                        [3.8295e-01, -9.8605e-01, -1.6701e-01],
                        [-4.7528e-01, -2.8737e-01, 2.6311e-01],
                        [-1.3966e+00, -1.3744e-01, 1.2726e+00],
                        [9.4526e-01, -1.5784e-01, -7.5309e-02],
                        [5.2631e-01, -3.1619e-01, 8.1557e-01]
                    ],
                    [
                        [2.1601e+00, 1.2314e+00, 1.8563e+00],
                        [8.4770e-01, 6.6654e-01, -9.8649e-02],
                        [9.2145e-02, 6.9004e-01, 7.7792e-03],
                        [8.2298e-01, -1.6009e+00, 3.1958e-01],
                        [1.7781e-01, -6.2230e-01, -3.2928e-01]
                    ]
                ],
            )?;
            let (expected, _) = make_tensor!(
                f32,
                [
                    [
                        [1.2381, -0.3433, 2.9396, -1.1015, 0.1276],
                        [-0.1335, 3.4467, 1.0562, 1.7610, 0.6045],
                        [1.3233, 3.4808, 4.5411, 2.7590, -1.3307],
                        [0.9636, 3.8556, 4.2921, -1.7481, -2.4406],
                        [2.9172, 2.0155, -1.5236, -2.5708, -3.8743]
                    ],
                    [
                        [-0.4767, -0.3071, 3.5954, 0.8265, -0.2037],
                        [0.4039, -1.4063, 3.2840, 2.2514, -0.8034],
                        [1.2593, 0.7444, 1.0667, 1.9431, 0.9907],
                        [0.6482, 7.7251, 2.6724, -2.0182, -4.6465],
                        [-0.3277, 2.8817, 1.7560, -1.8446, -3.3270]
                    ],
                    [
                        [1.4881, 1.0122, -0.5001, -2.4404, -0.0590],
                        [-0.4767, 0.8704, 2.5492, 1.4005, 1.9730],
                        [1.2901, 5.6284, 3.2765, 1.6170, -2.5378],
                        [3.4639, 4.4173, 2.7507, -0.6875, -1.7202],
                        [2.6701, 3.5191, 0.8889, -2.0664, -2.9624]
                    ],
                    [
                        [1.3659, 1.0598, 2.0609, -2.4204, 0.5957],
                        [1.1998, 2.9946, 2.5059, -0.6114, -0.0827],
                        [-1.5035, 3.9187, 2.6502, 2.0603, -0.1057],
                        [3.4743, 1.8747, -0.2402, 1.1975, -1.9844],
                        [3.2686, -0.0866, -0.1654, -2.2634, -2.0722]
                    ]
                ],
                [
                    [
                        [2.0635, 1.2183, -0.2066, 0.2045, 2.9659],
                        [3.3722, 1.7662, 0.6346, 0.9094, 3.0658],
                        [0.8943, 1.8979, -0.0464, 3.0577, 1.6243],
                        [1.4976, 4.8482, 2.6099, 0.0704, 1.2125],
                        [3.9615, 2.3500, 2.1069, 0.5022, 1.8891]
                    ],
                    [
                        [2.9516, 2.6835, -0.3297, 1.2063, 3.4254],
                        [1.9415, -0.6996, 0.5595, 1.3854, 1.0756],
                        [1.8959, -0.7900, -1.4683, 3.7450, 1.0177],
                        [3.8113, 4.7156, 0.7358, -0.4768, 1.0152],
                        [2.5536, 3.2853, 3.0557, 0.4126, -0.3446]
                    ],
                    [
                        [2.9442, 0.6257, 0.3769, 2.1093, 1.0043],
                        [2.4078, 2.1314, 0.0973, 2.2085, 3.4364],
                        [1.8316, -0.3404, 2.3759, -0.3242, 2.7339],
                        [4.0471, 2.6532, 1.7495, 1.7647, 1.3583],
                        [3.1823, 4.1501, 1.1978, 2.8213, 2.1236]
                    ],
                    [
                        [1.5752, 3.0735, -1.1559, 0.7527, 2.1573],
                        [0.6783, 1.2307, 1.0525, -0.2140, 1.1706],
                        [-0.3423, 0.0221, 2.0947, 1.2715, 0.9885],
                        [2.0386, 4.4832, 2.1149, -0.1863, 1.2452],
                        [2.9609, 3.6723, -0.3828, -0.8913, 1.8270]
                    ]
                ],
            )?;
            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 1e-3);
            Ok(())
        })
    }
}
