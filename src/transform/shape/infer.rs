use itertools::izip;

use crate::onnx::model::Graph;
use crate::onnx::model::UnifyMode;
use crate::tensor::types::TypeError;
use crate::transform::optimize::const_fold::fold_constant;
use crate::transform::optimize::const_fold::prop_constant;
use crate::transform::shape::infer_node_output;
use crate::transform::GraphOp;
use crate::transform::Pass;
use crate::transform::Target;

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

pub struct ShapeInference {
    pub target: Target,
}

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
            prop_constant(graph, id, modifier);
            let types = infer_node_output(graph, id, unify_mode, self.target)?;
            let outputs = graph.nodes[id].outputs.clone();
            for (value_id, inferred) in izip!(outputs.iter(), types.iter()) {
                graph.try_unify_type(*value_id, inferred, unify_mode)?;
                // dbg!(&graph.values[*value_id]);
            }

            if let Some(constants) = fold_constant(graph, id) {
                // dbg!(id, &constants);
                for (old_value, tensor, ty) in
                    izip!(outputs.iter(), constants.into_iter(), types.into_iter())
                {
                    let tensor = tensor.reshape(&ty.dims);
                    let new_value = modifier.register_new_tensor(
                        graph,
                        tensor,
                        format!("folded_{}", graph.nodes[id].name),
                    );
                    modifier.replace_input_value_if_without_typecheck(
                        graph,
                        *old_value,
                        new_value,
                        |_, _| true,
                    );
                }
            }
        }

        Ok(())
    }
}
