use crate::codegen::jit::{CodegenError, GraphCompiler, JIT};
use crate::model::{Graph, Model, ValueId};
use crate::optimize::optimizer::Optimizer;
use crate::tensor::tensor::{ResolvedTensorType, Tensor, TypeError};
use itertools::izip;
use std::path::Path;

type CodeType = fn(*const u8, *const u8);

#[derive(Debug)]
pub enum SessionError {
    ModelLoadError(crate::load::ModelLoadError),
    TypeError(TypeError),
    CodegenError(CodegenError),
    InvalidInputNumber {
        expected: usize,
        got: usize,
    },
    InvalidInputType {
        expected: ResolvedTensorType,
        got: ResolvedTensorType,
    },
    OtherError(String),
}

pub struct Session {
    inputs_ty: Vec<ResolvedTensorType>,
    outputs_ty: Vec<ResolvedTensorType>,
    code: CodeType,
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
    pub fn new<P: AsRef<Path>>(p: P, pass: Optimizer) -> Result<Self, SessionError> {
        let mut model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        model.graph.infer().map_err(SessionError::TypeError)?;
        pass.run(&mut model.graph);
        let inputs_ty = get_argument_types(&model.graph, &model.graph.inputs)?;
        let outputs_ty = get_argument_types(&model.graph, &model.graph.outputs)?;
        let mut jit = JIT::default();
        let code =
            GraphCompiler::compile(&mut jit, &model.graph).map_err(SessionError::CodegenError)?;
        let code: CodeType = unsafe { std::mem::transmute(code) };
        Ok(Session {
            inputs_ty,
            outputs_ty,
            code,
        })
    }

    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, SessionError> {
        if inputs.len() != self.inputs_ty.len() {
            return Err(SessionError::InvalidInputNumber {
                expected: inputs.len(),
                got: self.inputs_ty.len(),
            });
        }

        let inputs = izip!(inputs, &self.inputs_ty)
            .map(|(t, ty)| {
                if t.ty == *ty {
                    Ok(t.data.raw_vec())
                } else {
                    Err(SessionError::InvalidInputType {
                        expected: ty.clone(),
                        got: t.ty.clone(),
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let outputs = self
            .outputs_ty
            .iter()
            .map(|ty| ty.mem_size())
            .sum::<usize>();
        let mut raw_output = vec![0u8; outputs];
        (self.code)(inputs.as_ptr(), raw_output.as_mut_ptr());
        let mut outputs = Vec::new();
        let mut offset = 0;
        for ty in self.outputs_ty.iter() {
            let size = ty.mem_size();
            let output = Tensor::from_bytes(ty.clone(), &raw_output[offset..offset + size])
                .map_err(SessionError::TypeError)?;
            outputs.push(output);
            offset += size;
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod test {
    use crate::codegen::session::{Session, SessionError};
    use crate::optimize::{matmul_a_tb::MatMulAxTB, optimizer::Optimizer};
    use crate::tensor::tensor::Tensor;
    use std::path::PathBuf;

    macro_rules! make_tensor {
        ($ty: ty, $($expr: expr,)*) => {{
            let orig: ndarray::Array<$ty, _> = ndarray::array!($($expr,)*);
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .try_into()
                .map(|t| (t, orig.clone()))
                .map_err(SessionError::TypeError);
            res
        }};
    }

    fn make_session(path: &str) -> Result<Session, SessionError> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
        let mut optimizer = Optimizer::new(String::from("test pass"));
        optimizer.passes.push(Box::new(MatMulAxTB::default()));
        Session::new(path, optimizer)
    }

    type TestResult = Result<(), SessionError>;

    macro_rules! tensor_assert_eq {
        ($left: expr, $right: expr) => {{
            let right = Tensor::try_from($right).map_err(SessionError::TypeError)?;
            assert_eq!($left, right);
        }};
    }

    #[test]
    fn relu() -> TestResult {
        let session = make_session("models/test/relu.onnx")?;
        let (input, orig) =
            make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],)?;
        let output = session.run(&[input])?;
        tensor_assert_eq!(output[0], orig.mapv(|x| x.max(0.0)));
        Ok(())
    }

    #[test]
    fn add() -> TestResult {
        let session = make_session("models/test/add.onnx")?;
        let (input0, orig0) = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],)?;
        let (input1, orig1) = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0],)?;
        let output = session.run(&[input0, input1])?;
        tensor_assert_eq!(output[0], orig0 + orig1);
        Ok(())
    }

    #[test]
    fn add_broaccst() -> TestResult {
        let session = make_session("models/test/add_broadcast.onnx")?;
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
        tensor_assert_eq!(output[0], orig0 + orig1 + orig2);
        Ok(())
    }

    #[ignore]
    #[test]
    fn run_reshape() -> TestResult {
        let session = make_session("models/test/reshape.onnx")?;

        let (input, orig) = make_tensor!(
            f32,
            [[[1.0, 10.0]], [[2.0, 10.0]]],
            [[[1.1, 10.1]], [[3.0, 10.1]]],
            [[[1.2, 10.2]], [[4.0, 10.2]]],
            [[[1.3, 10.3]], [[5.0, 10.3]]],
            [[[1.4, 10.4]], [[6.0, 10.4]]],
            [[[1.5, 10.5]], [[7.0, 10.5]]],
        )?;
        let output = session.run(&[input])?;
        let expected = orig
            .to_shape((3, 1, 1, 2, 4))
            .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
            .to_owned();
        tensor_assert_eq!(output[0], expected);
        Ok(())
    }

    #[test]
    fn matmul() -> TestResult {
        let session = make_session("models/test/matmul.onnx")?;

        let (input0, orig0) = make_tensor!(
            f32,
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        )?;
        let (input1, orig1) = make_tensor!(f32, [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],)?;
        let output = session.run(&[input0, input1])?;
        tensor_assert_eq!(output[0], orig0.dot(&orig1));
        Ok(())
    }

    #[test]
    fn matmul_a_x_tb() -> TestResult {
        let session = make_session("models/test/matmul_a_x_tb.onnx")?;

        // (5 x 7)
        let (input0, orig0) = make_tensor!(
            f32,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
            [29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
        )?;

        // (6 x 7)
        let (input1, orig1) = make_tensor!(
            f32,
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
            [29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0],
            [36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0],
        )?;
        let output = session.run(&[input0, input1])?;
        tensor_assert_eq!(output[0], orig0.dot(&orig1.t()));
        Ok(())
    }
}
