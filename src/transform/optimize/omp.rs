use crate::onnx::model::Graph;
use crate::transform::modify::GraphModifier;
use crate::transform::Pass;

pub struct InnermostOMP {
    pub threshold: usize,
}

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
                let ty = &graph.get_resolved_tensor_type(output).unwrap();
                let annotate = ty
                    .dims
                    .last()
                    .map(|x| self.threshold <= *x)
                    .unwrap_or(false);
                if !annotate {
                    return None;
                }
                let ndim = ty.dims.ndim();
                if ty.stride(ndim - 1) != 1 {
                    return None;
                }
                Some((id, ty.dims.ndim() - 1))
            })
            .collect::<Vec<_>>();

        for (id, dim) in ids.iter() {
            let node = &mut graph.nodes[*id];
            node.meta.omp_parallel = Some(*dim);
            node.meta.omp_for = Some(*dim);
        }
    }
}
