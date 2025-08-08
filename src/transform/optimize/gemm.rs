use crate::onnx::model::{Graph, Node, NodeMeta, ValueId};
use crate::onnx::operator::*;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

// #[derive(Default)]
// pub struct TransformBLASGemm {}
//
// impl<T: GraphModifier> Pass<T> for TransformBLASGemm {
//     fn summary(&self) -> &'static str {
//         "Transform ONNX Gemm to BLAS Gemm"
//     }
//
//     fn run(&self, graph: &mut Graph, modifier: &mut T) {
//         let res = graph
//             .nodes
//             .iter()
//             .filter_map(|(id, node)| {
//                 if let Operator::Gemm(gemm) = &node.op {
//                     Some((id, gemm.clone()))
//                 } else {
//                     None
//                 }
//             })
//             .collect::<Vec<_>>();
//         for (id, gemm) in res.iter() {
//             let a = graph.nodes[*id].inputs[args::GEMM_A];
//             let b = graph.nodes[*id].inputs[args::GEMM_B];
//             let c_opt = graph.nodes[*id].inputs.get(args::GEMM_C).cloned();
//             let old_output = graph.nodes[*id].outputs[0];
//             let ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
//
//             // TODO
//             let blas_gemm: BLASGemm = gemm.try_into().unwrap();
//             let new_output = modifier.register_new_value(
//                 graph,
//                 format!("TransformBLASGemm_Output_{:?}", id),
//                 ty.clone(),
//             );
//             modifier.register_new_node(
//                 graph,
//                 Node {
//                     inputs: vec![a, b],
//                     outputs: vec![new_output],
//                     name: format!("TransformBLASGemm_{:?}", id),
//                     op: Operator::BLASGemm(blas_gemm),
//                     meta: NodeMeta::default(),
//                 },
//             );
//
//             let new_output = if let Some(c) = c_opt {
//                 let add = modifier.register_new_value(
//                     graph,
//                     format!("TransformBLASGemm_Add_{:?}", id),
//                     ty,
//                 );
//                 modifier.register_new_node(
//                     graph,
//                     Node {
//                         inputs: vec![new_output, c],
//                         outputs: vec![add],
//                         name: format!("TransformBLASGemm_Add_{:?}", id),
//                         op: Operator::Add,
//                         meta: NodeMeta::default(),
//                     },
//                 );
//                 add
//             } else {
//                 new_output
//             };
//
//             modifier.replace_input_value(graph, old_output, new_output);
//         }
//     }
// }

#[derive(Default)]
pub struct GemmTransComposition {}

impl<T: GraphOp> Pass<T> for GemmTransComposition {
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
                    meta: NodeMeta::default(),
                };
                modifier.register_new_node(graph, new_node);
                modifier.replace_input_value(graph, old_output, new_output);
            }
        }
    }
}
