use itertools::zip_eq;

use crate::onnx::model::Graph;
use crate::onnx::model::UnifyMode;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::TypeError;
use crate::transform::shape::infer_node_output;
use crate::transform::GraphOp;
use crate::transform::Pass;
use crate::transform::Target;

pub struct VerifyShape {
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

impl<T: GraphOp> Pass<T> for VerifyShape {
    fn summary(&self) -> &'static str {
        "Verify shapes"
    }

    fn run(&self, graph: &mut Graph, _modifier: &mut T) {
        self.run_impl(graph).expect("Shape verification failed");
    }
}

impl VerifyShape {
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

    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::*;
    use crate::tensor::types::FloatType;
    use crate::transform::layout::strides::AssignStrides;
    use crate::transform::modify::SimpleGraphOp;
    use crate::transform::shape::infer::ShapeInference;
    use crate::transform::*;

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
        let mut modifier = SimpleGraphOp::new(&graph);
        let mut pass_manager = SimplePassManager::new("Shape".to_string());
        let target = Target::CPU;
        pass_manager.add_pass(Box::new(ShapeInference { target }));
        pass_manager.add_pass(Box::new(AssignStrides { target }));
        pass_manager.add_pass(Box::new(VerifyShape {
            target,
            check_strides: true,
        }));
        pass_manager.run(&mut graph, &mut modifier);
    }
}
