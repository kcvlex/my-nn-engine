use crate::onnx::model::{Graph, UnifyMode};
use crate::tensor::types::TypeError;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::util;
use crate::transform::shape::verify;
use crate::transform::utils::const_fold::fold_constant;
use crate::transform::SimplePassManager;
use crate::transform::{GraphOp, Pass, PassManager};
use itertools::zip_eq;

pub struct Config {
    pub unify_mode: UnifyMode,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            unify_mode: UnifyMode::CheckStrides,
        }
    }
}

#[derive(Default)]
pub struct ShapeInference {}

impl<T: GraphOp> Pass<T> for ShapeInference {
    fn summary(&self) -> &'static str {
        "Infer shape of each node"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        self.infer(graph, modifier).unwrap();
    }
}

impl ShapeInference {
    fn infer<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) -> Result<(), TypeError> {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| !node.is_dummy())
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        let unify_mode = UnifyMode::OverwriteStrides;
        for id in ids {
            let types = util::infer_node_output(graph, id, unify_mode)?;
            let outputs = graph.nodes[id].outputs.clone();
            for (value_id, inferred) in zip_eq(outputs.iter(), types.into_iter()) {
                graph.try_unify_type(*value_id, &inferred, unify_mode)?;
                // dbg!(&graph.values[*value_id]);
            }

            if let Some(constants) = fold_constant(graph, id) {
                // dbg!(id, &constants);
                for (old_value, tensor) in zip_eq(outputs.iter(), constants.into_iter()) {
                    let new_value = modifier.register_new_tensor(
                        graph,
                        tensor,
                        format!("folded_{}", graph.nodes[id].name),
                    );
                    modifier.replace_input_value(graph, *old_value, new_value);
                }
            }
        }

        Ok(())
    }
}

pub fn create_infer_passes(verify: bool) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Shape inference".to_string());
    manager.add_pass(Box::new(ShapeInference::default()));
    if verify {
        manager.add_pass(Box::new(verify::VerifyShape {
            check_strides: false,
        }));
    }
    manager
}
