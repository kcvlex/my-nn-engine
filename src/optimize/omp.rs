use crate::onnx::model::Graph;
use crate::optimize::optimizer::{GraphModifier, Pass};

#[derive(Default)]
pub struct InnermostOMP {}

const THRESHOLD: usize = 100;

impl<T: GraphModifier> Pass<T> for InnermostOMP {
    fn summary(&self) -> &'static str {
        "Annotate omp parallel and for to the innermost loop"
    }

    fn run(&self, graph: &mut Graph, _modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if !node.op.is_elementwise() {
                    return None;
                }

                let output = node.outputs[0];
                let dims = &graph.get_resolved_tensor_type(output).unwrap().dims;
                let annotate = dims.last().map(|x| THRESHOLD <= *x).unwrap_or(false);
                if !annotate {
                    return None;
                }
                Some((id, dims.ndim() - 1))
            })
            .collect::<Vec<_>>();

        for (id, dim) in ids.iter() {
            let node = &mut graph.nodes[*id];
            node.meta.omp_parallel = Some(*dim);
            node.meta.omp_for = Some(*dim);
        }
    }
}
