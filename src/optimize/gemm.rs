use crate::model::{Graph, Node, ValueId};
use crate::operator::*;
use crate::optimize::optimizer::{GraphModifier, Pass};

#[derive(Default)]
pub struct GemmTransComposition {}

impl<T: GraphModifier> Pass<T> for GemmTransComposition {
    fn summary(&self) -> &'static str {
        "Compose Gemm and Tranpose into Gemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let res = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                if matches!(node.op, Operator::Gemm(_)) {
                    Some(id)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for (index, id) in res.into_iter().enumerate() {
            let is_transposed = |id: ValueId| {
                let node_id = modifier.defined_node(id)?.0;
                let node = &graph.nodes[node_id];
                match node.op {
                    Operator::Transpose(_) => Some(node.inputs[0]),
                    _ => None,
                }
            };

            let a = graph.nodes[id].inputs[args::GEMM_A];
            let b = graph.nodes[id].inputs[args::GEMM_B];
            let old_output = graph.nodes[id].outputs[0];
            let trans_a = is_transposed(a);
            let trans_b = is_transposed(b);

            // TODO: When output is transposed
            if trans_a.is_some() || trans_b.is_some() {
                let mut gemm = match &graph.nodes[id].op {
                    Operator::Gemm(gemm) => gemm.clone(),
                    _ => unreachable!(),
                };
                gemm.trans_a ^= trans_a.is_some();
                gemm.trans_b ^= trans_b.is_some();
                let ty = graph
                    .get_resolved_tensor_type(graph.nodes[id].outputs[0])
                    .unwrap()
                    .clone();
                let new_output = modifier.register_new_value(
                    graph,
                    format!("GemmTransComposition_Output_{index}"),
                    ty,
                );
                let new_node = Node {
                    inputs: vec![trans_a.unwrap_or(a), trans_b.unwrap_or(b)],
                    outputs: vec![new_output],
                    name: format!("GemmTransComposition_{index}"),
                    op: Operator::Gemm(gemm),
                    mark_as_deleted: false,
                };
                println!("new_node: {:?}", new_node);
                modifier.register_new_node(graph, new_node);
                modifier.replace_input_value(graph, old_output, new_output);
            }
        }
    }
}

#[derive(Default)]
pub struct MatMul2Gemm {}

impl<T: GraphModifier> Pass<T> for MatMul2Gemm {
    fn summary(&self) -> &'static str {
        "Convert 2-D MatMul to Gemm"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let mut res = Vec::new();
        for (id, node) in graph.nodes.iter() {
            let (lhs, rhs) = if matches!(node.op, Operator::MatMul) {
                (node.inputs[args::MATMUL_LHS], node.inputs[args::MATMUL_RHS])
            } else {
                continue;
            };

            let ldim = graph.get_resolved_tensor_type(lhs).unwrap().dims.ndim();
            let rdim = graph.get_resolved_tensor_type(rhs).unwrap().dims.ndim();
            if ldim == 2 && rdim == 2 {
                res.push(id)
            }
        }
        for (index, id) in res.into_iter().enumerate() {
            let lhs = graph.nodes[id].inputs[args::MATMUL_LHS];
            let rhs = graph.nodes[id].inputs[args::MATMUL_RHS];
            let old_output = graph.nodes[id].outputs[0];
            let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
            let new_output =
                modifier.register_new_value(graph, format!("MatMul2Gemm_Output_{index}"), ty);
            let new_node = Node {
                inputs: vec![lhs, rhs],
                outputs: vec![new_output],
                name: format!("MatMul2Gemm_{index}"),
                op: Operator::Gemm(Gemm::default()),
                mark_as_deleted: false,
            };
            modifier.register_new_node(graph, new_node);
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}
