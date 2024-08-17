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
        ($ty: ty, $($expr: expr),*) => {{
            let res: ndarray::Array<$ty, _> = ndarray::array!($($expr),*);
            let res: Result<Tensor, _> = res.try_into().map_err(SessionError::TypeError);
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
        let input = make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]])?;
        let output = session.run(&[input])?;
        let expected = make_tensor!(f32, [[1.0, 0.0], [42.0, 4.0]], [[0.0, 6.0], [0.0, 0.0]])?;
        assert_eq!(output[0], expected);
        Ok(())
    }

    #[test]
    fn run_add() -> Result<(), SessionError> {
        let session = make_session("models/test/add.onnx")?;
        let input0 = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0])?;
        let input1 = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0])?;
        let output = session.run(&[input0, input1])?;
        let expected = make_tensor!(f32, [2.0, 4.0, 6.0], [0.0, 0.0, 0.0])?;
        assert_eq!(output[0], expected);
        Ok(())
    }
}
