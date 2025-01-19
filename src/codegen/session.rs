use crate::codegen::gen::{CodeGen, CodeGenError};
use crate::onnx::load::*;
use crate::onnx::model::{Graph, Model, ValueId};
use crate::optimize::{
    batchnorm, gemm, identity, im2col, infer, normalize,
    optimizer::{Optimizer, SimpleGraphModifier},
    reduce,
};
use crate::tensor::{
    resolved_dimensions::ResolvedTensorDims,
    tensor::{ResolvedTensorType, Tensor, TypeError},
};

use inkwell::context::Context;
use inkwell::targets::FileType;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use rand::distributions::{Alphanumeric, DistString};
use rand::rngs::SmallRng;
use rand::SeedableRng;

use std::path::PathBuf;
use std::process::Command;

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8, *const *const u8);

#[derive(Debug)]
pub enum SessionError {
    CodeGenError(CodeGenError),
    ModelLoadError(ModelLoadError),
    TypeError(TypeError),
    OtherError(String),
}

pub struct Session<'ctx> {
    #[allow(dead_code)]
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,
    initializer: Vec<Tensor>,

    #[allow(dead_code)]
    codegen: CodeGen<'ctx>,

    shared_obj: PathBuf,

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

impl<'ctx> Session<'ctx> {
    pub fn new<P: AsRef<Path>>(
        ctx: &'ctx Context,
        p: P,
        input_ty: Option<&[&ResolvedTensorDims]>,
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

        let mut codegen = CodeGen::new(ctx, model.graph).map_err(SessionError::CodeGenError)?;
        println!("Compiling");
        codegen
            .compile_default()
            //.compile_with_passes(&[])
            .map_err(SessionError::CodeGenError)?;
        println!("Compiled");

        //codegen.module().print_to_file("model.ll").unwrap();

        let mut rng = SmallRng::from_entropy();
        let id = Alphanumeric.sample_string(&mut rng, 16);

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tmp_obj = dir.join(format!("model_{}.o", id));
        let shared_obj = dir.join(format!("model_{}.so", id));

        codegen
            .target_machine()
            .write_to_file(codegen.module(), FileType::Object, tmp_obj.as_ref())
            .map_err(CodeGenError::LLVMError)
            .map_err(SessionError::CodeGenError)?;

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
                tmp_obj.to_str().unwrap(),
            ])
            .status()
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        Command::new("rm")
            .args(["-f", tmp_obj.to_str().unwrap()])
            .status()
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;

        let lib = unsafe { libloading::Library::new(shared_obj.as_os_str()) }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func: libloading::Symbol<CodeType> = unsafe { lib.get(b"main") }
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
        let func = *func;

        println!("Loaded");

        Ok(Session {
            input_ty: inputs_ty,
            output_ty: outputs_ty,
            codegen,
            shared_obj,
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
        Self::_write_model(self.codegen.graph(), p);
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        Command::new("rm")
            .args(["-f", self.shared_obj.to_str().unwrap()])
            .status()
            .unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tensor::tensor::Tensor;
    use inkwell::context::Context;

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
            let res = $left.eq_with_epsilon(&$right, $epsilon);
            if !res {
                // For pretty print
                assert_eq!($left, $right);
            }
        }};
    }

    fn make_session<P: AsRef<std::path::Path>>(
        ctx: &'_ Context,
        path: P,
    ) -> Result<Session<'_>, SessionError> {
        use std::path::PathBuf;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/operator")
            .join(path);
        Session::new(ctx, path, None)
    }

    fn with_session<P, F>(path: P, f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: FnOnce(Session) -> TestResult,
    {
        let context = Context::create();
        let session = make_session(&context, path)?;
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
}
