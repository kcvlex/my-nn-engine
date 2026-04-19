use itertools::zip_eq;

use crate::onnx::model::Graph;
use crate::onnx::model::UnifyMode;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::TypeError;
use crate::transform::shape::infer_node_output;
use crate::transform::GraphOp;
use crate::transform::Pass;
use crate::transform::Target;

pub struct ShapeVerification {
    pub check_strides: bool,
    pub target: Target,
}

#[derive(Debug, Clone)]
pub enum VerifyShapeError {
    OutputContiguous(ValueId, ResolvedTensorType),
    InconsistentShape(ValueId, ResolvedTensorDims, ResolvedTensorDims),
    InconsistentStrides(ValueId, ResolvedTensorType, ResolvedTensorType),
    UnresolvedShape(ValueId),
    TypeError(TypeError),
    Other(String),
}

impl<T: GraphOp> Pass<T> for ShapeVerification {
    fn summary(&self) -> &'static str {
        "Verify shapes"
    }

    fn run(&self, graph: &mut Graph, _modifier: &mut T) {
        self.run_impl(graph).expect("Shape verification failed");
    }
}

impl ShapeVerification {
    fn run_impl(&self, graph: &Graph) -> Result<(), VerifyShapeError> {
        for (node_id, node) in graph.nodes.iter() {
            match node.op {
                Operator::Input(_) => (),
                Operator::Output(value) if self.check_strides => {
                    let resolved = graph
                        .get_resolved_tensor_type(value)
                        .ok_or(VerifyShapeError::UnresolvedShape(value))?;
                    if !resolved.is_contiguous() {
                        return Err(VerifyShapeError::OutputContiguous(value, resolved.clone()));
                    }
                }
                _ => {
                    let resolved = infer_node_output(
                        graph,
                        node_id,
                        if self.check_strides {
                            UnifyMode::CheckStrides
                        } else {
                            UnifyMode::IgnoreStrides
                        },
                        self.target,
                    )
                    .map_err(VerifyShapeError::TypeError)?;
                    for (value_id, inferred) in zip_eq(node.outputs.iter(), resolved.into_iter()) {
                        let cur = graph
                            .get_resolved_tensor_type(*value_id)
                            .ok_or(VerifyShapeError::UnresolvedShape(*value_id))?;
                        if cur.dims != inferred.dims {
                            return Err(VerifyShapeError::InconsistentShape(
                                *value_id,
                                cur.dims.clone(),
                                inferred.dims.clone(),
                            ));
                        }
                        if self.check_strides && cur.strides() != inferred.strides() {
                            return Err(VerifyShapeError::InconsistentStrides(
                                *value_id,
                                cur.clone(),
                                inferred.clone(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use crate::onnx::load::*;
    use crate::onnx::model::*;
    use crate::options::Options;
    use crate::tensor::types::FloatType;
    use crate::tensor::types::ResolvedTensorDims;
    use crate::tensor::types::ResolvedTensorType;
    use crate::transform::transform_graph;

    #[test]
    fn infer_yolov4() {
        let model = "yolov4";
        let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/validated")
            .join(model)
            .join(format!("{model}.onnx"));
        let model = Model::load_from_path(&model).unwrap();
        let mut graph = model.graph;
        graph
            .resolve_input_types(&[ResolvedTensorType::new(
                FloatType::F32.into(),
                ResolvedTensorDims::new(&[1, 416, 416, 3]),
            )])
            .unwrap();
        let options = Options::builder().build();
        transform_graph(&mut graph, &options);
    }

    #[cfg(feature = "local")]
    #[test]
    fn infer_tinyllama() {
        use crate::tensor::Tensor;
        use crate::transform::create_infer_passes;
        use crate::transform::modify::SimpleGraphOp;
        use crate::transform::PassManager;

        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/hf/tinyllama");
        let model = Model::load_from_path(dir.join("model.onnx")).unwrap();
        let mut graph = model.graph;

        let test_dir = dir.join("test_data_set_0");
        let mut input_tys = Vec::new();
        let mut i = 0;
        loop {
            let p = test_dir.join(format!("input_{}.pb", i));
            if !p.exists() {
                break;
            }
            let t = Tensor::load_from_path(&p).unwrap();
            input_tys.push(t.tensor_type());
            i += 1;
        }
        graph.resolve_input_types(&input_tys).unwrap();

        let options = Options::builder().build();
        let manager = create_infer_passes(&options);
        let mut modifier = SimpleGraphOp::new(&graph);
        manager.run(&mut graph, &mut modifier);
    }
}
