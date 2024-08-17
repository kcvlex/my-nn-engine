use crate::codegen::jit::{CodegenError, GraphCompiler, JIT};
use crate::model::{Graph, Model, ValueId};
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
    pub fn new<P: AsRef<Path>>(p: P) -> Result<Self, SessionError> {
        let mut model = Model::load_from_path(p).map_err(SessionError::ModelLoadError)?;
        model.graph.infer().map_err(SessionError::TypeError)?;
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
        Session::new(path)
    }

    #[test]
    fn run_relu() -> Result<(), SessionError> {
        let session = make_session("models/test/relu.onnx")?;
        let (input, _) =
            make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],)?;
        let output = session.run(&[input])?;
        let (expected, _) =
            make_tensor!(f32, [[1.0, 0.0], [42.0, 4.0]], [[0.0, 6.0], [0.0, 0.0]],)?;
        assert_eq!(output[0], expected);
        Ok(())
    }

    #[test]
    fn run_add() -> Result<(), SessionError> {
        let session = make_session("models/test/add.onnx")?;
        let (input0, _) = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],)?;
        let (input1, _) = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0],)?;
        let output = session.run(&[input0, input1])?;
        let (expected, _) = make_tensor!(f32, [2.0, 4.0, 6.0], [0.0, 0.0, 0.0],)?;
        assert_eq!(output[0], expected);
        Ok(())
    }

    #[test]
    fn run_add_broaccst() -> Result<(), SessionError> {
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
        let expected = orig0 + orig1 + orig2;
        let exptected = Tensor::try_from(expected).map_err(SessionError::TypeError)?;
        assert_eq!(output[0], exptected);
        Ok(())
    }
}
