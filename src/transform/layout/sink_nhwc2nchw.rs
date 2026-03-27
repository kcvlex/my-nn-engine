use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct SinkNHWC2NCHW {}

impl<T: GraphOp> Pass<T> for SinkNHWC2NCHW {
    fn summary(&self) -> &'static str {
        "Sink NHWC2NCHW past elementwise ops toward Conv/MaxPool/Im2Col"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let topo: Vec<NodeId> = simple_topological_order(graph)
            .into_iter()
            .filter(|id| graph.nodes[*id].op.is_elementwise())
            .collect();

        for id in topo {
            self.try_sink_through(graph, modifier, id);
        }
    }
}

impl SinkNHWC2NCHW {
    fn try_sink_through<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T, elem_id: NodeId) {
        let node = &graph.nodes[elem_id];
        if !node.op.is_elementwise() {
            return;
        }

        let inputs = node.inputs.clone();
        let outputs = node.outputs.clone();
        let op = node.op.clone();
        let name = node.name.clone();

        // Classify each input:
        // - NHWC2NCHW: unwrap to get NHWC data
        // - 4D initializer: permute strides in-place
        // - 4D non-initializer: insert Reinterpret(Transpose [0,2,3,1]) to view as NHWC
        // - Otherwise: cannot sink
        let mut nhwc_inputs = Vec::new(); // (input_idx, nhwc_value)
        let mut init_permutable = Vec::new(); // (input_idx, value_id) - 4D initializers
        let mut reinterpretable = Vec::new(); // (input_idx, value_id) - 4D non-initializers
        let mut can_sink = true;

        for (i, input) in inputs.iter().enumerate() {
            let def = modifier.defined_node(*input);
            if let Some((def_id, _)) = def {
                if matches!(graph.nodes[def_id].op, Operator::NHWC2NCHW) {
                    let nhwc_val = graph.nodes[def_id].inputs[0];
                    nhwc_inputs.push((i, nhwc_val));
                    continue;
                }
            }
            let ty = graph.get_resolved_tensor_type(*input).unwrap();
            if ty.dims.ndim() != 4 {
                can_sink = false;
                break;
            }
            if graph.initializer.contains_key(input) {
                init_permutable.push((i, *input));
            } else {
                reinterpretable.push((i, *input));
            }
        }

        if !can_sink || nhwc_inputs.is_empty() {
            return;
        }

        // Permute strides of 4D initializer inputs: NCHW -> NHWC (perm [0,2,3,1])
        for &(_, v) in &init_permutable {
            let ty = graph
                .get_resolved_tensor_type(v)
                .unwrap()
                .transpose(&[0, 2, 3, 1]);
            modifier.replace_tensor_type(graph, v, ty);
        }

        // Insert Reinterpret(Transpose [0,2,3,1]) for 4D non-initializer inputs
        for &(i, v) in &reinterpretable {
            let ty = graph.get_resolved_tensor_type(v).unwrap().clone();
            let nhwc_ty = ty.transpose(&[0, 2, 3, 1]);
            let reinterpreted = modifier.register_new_value(
                graph,
                format!("SinkNHWC2NCHW_Reinterpret_{}", v.index()),
                nhwc_ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![v],
                    outputs: vec![reinterpreted],
                    name: format!("SinkNHWC2NCHW_Reinterpret_{}", v.index()),
                    op: Operator::Reinterpret(Reinterpret {
                        ops: vec![ReinterpretType::Transpose(Transpose {
                            perm: Some(vec![0, 2, 3, 1]),
                        })],
                    }),
                    meta: NodeMeta::default(),
                },
            );
            nhwc_inputs.push((i, reinterpreted));
        }

        // Build new elementwise node with NHWC inputs
        let nhwc_ty = graph
            .get_resolved_tensor_type(nhwc_inputs[0].1)
            .unwrap()
            .clone();
        let mut new_inputs = inputs.clone();
        for &(i, nhwc_val) in &nhwc_inputs {
            new_inputs[i] = nhwc_val;
        }

        let mut new_outputs = Vec::with_capacity(outputs.len());
        for (oi, _) in outputs.iter().enumerate() {
            let new_out = modifier.register_new_value(
                graph,
                format!("SinkNHWC2NCHW_Elem_{}_{}", elem_id.index(), oi),
                nhwc_ty.clone(),
            );
            new_outputs.push(new_out);
        }

        modifier.register_new_node(
            graph,
            Node {
                inputs: new_inputs,
                outputs: new_outputs.clone(),
                name: format!("SinkNHWC2NCHW_Elem_{}", name),
                op,
                meta: NodeMeta::default(),
            },
        );

        // For each output, insert NHWC2NCHW and replace uses
        for (oi, old_output) in outputs.iter().enumerate() {
            let old_ty = graph.get_resolved_tensor_type(*old_output).unwrap().clone();
            let new_nchw = modifier.register_new_value(
                graph,
                format!("SinkNHWC2NCHW_NCHW_{}_{}", elem_id.index(), oi),
                old_ty,
            );
            modifier.register_new_node(
                graph,
                Node {
                    inputs: vec![new_outputs[oi]],
                    outputs: vec![new_nchw],
                    name: format!("SinkNHWC2NCHW_NCHW_{}_{}", elem_id.index(), oi),
                    op: Operator::NHWC2NCHW,
                    meta: NodeMeta::default(),
                },
            );
            modifier.replace_input_value(graph, *old_output, new_nchw);
        }
    }
}
