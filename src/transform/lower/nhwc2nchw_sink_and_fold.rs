use std::collections::HashMap;
use std::collections::HashSet;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::tensor::types::TensorType;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct NHWC2NCHWSinkAndFold {}

impl<T: GraphOp> Pass<T> for NHWC2NCHWSinkAndFold {
    fn summary(&self) -> &'static str {
        "Sink NHWC2NCHW through elementwise, fold into Conv, revert unfoldable"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        // Phase 1: Sink NHWC2NCHW through elementwise ops toward Conv
        self.sink(graph, modifier);

        // Phase 2: Fold NHWC2NCHW -> Conv (set input_layout=NHWC)
        self.fold_into_conv_input(graph, modifier);

        // Phase 3: Revert remaining Conv(output_layout=NHWC) -> NHWC2NCHW
        // (set output_layout=NCHW, remove NHWC2NCHW)
        self.revert(graph, modifier);
    }
}

impl NHWC2NCHWSinkAndFold {
    fn sink<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
        let nodes_ids = simple_topological_order(graph);
        let mut marked = {
            let mut marker = SinkMarker::new(graph, modifier);
            marker.mark();
            marker.marked
        };
        for id in nodes_ids {
            self.do_sink(graph, modifier, &mut marked, id);
        }
    }

    fn do_sink<T: GraphOp>(
        &self,
        graph: &mut Graph,
        modifier: &mut T,
        marked: &mut HashSet<NodeId>,
        node_id: NodeId,
    ) {
        let mut cands = vec![None; graph.nodes[node_id].inputs.len()];
        for (i, input) in graph.nodes[node_id].inputs.iter().enumerate() {
            let Some(input) = input else { continue };
            if let Some((def_id, _)) = modifier.defined_node(*input) {
                if matches!(graph.nodes[def_id].op, Operator::NHWC2NCHW) {
                    if marked.contains(&def_id) {
                        cands[i] = Some(def_id);
                    } else {
                        return;
                    }
                }
            }
        }

        if cands.iter().all(|c| c.is_none()) {
            return;
        }

        match &graph.nodes[node_id].op {
            Operator::Conv(_) => {
                assert!(matches!(
                    graph.nodes[cands[0].unwrap()].op,
                    Operator::NHWC2NCHW
                ));
                let cur_input = graph.nodes[node_id].inputs[0].unwrap();
                let new_input = graph.nodes[cands[0].unwrap()].inputs[0].unwrap();
                let conv = match &mut graph.nodes[node_id].op {
                    Operator::Conv(ref mut conv) => conv,
                    _ => unreachable!(),
                };
                assert!(conv.input_layout == Layout::NCHW);
                conv.input_layout = Layout::NHWC;
                modifier.replace_input_value_if_without_typecheck(
                    graph,
                    cur_input,
                    new_input,
                    |id, _| id == node_id,
                );
            }

            Operator::MaxPool(_) => {
                assert!(matches!(
                    graph.nodes[cands[0].unwrap()].op,
                    Operator::NHWC2NCHW
                ));
                let cur_input = graph.nodes[node_id].inputs[0].unwrap();
                let new_input = graph.nodes[cands[0].unwrap()].inputs[0].unwrap();
                let pooling = match &mut graph.nodes[node_id].op {
                    Operator::MaxPool(ref mut pooling) => pooling,
                    _ => unreachable!(),
                };
                assert!(pooling.layout == Layout::NCHW);
                pooling.layout = Layout::NHWC;
                modifier.replace_input_value_if_without_typecheck(
                    graph,
                    cur_input,
                    new_input,
                    |id, _| id == node_id,
                );

                let output = graph.nodes[node_id].outputs[0];
                let old_ty = graph.get_resolved_tensor_type(output).unwrap().clone();
                let nhwc_ty = old_ty.transpose(&[0, 2, 3, 1]).contiguous();
                graph.values[output].ty = Some(TensorType::Resolved(nhwc_ty));

                let nchw_output = modifier.register_new_value(
                    graph,
                    format!("MaxPool_NHWC2NCHW_{}", output.index()),
                    old_ty,
                );
                let nhwc2nchw_id = modifier.register_new_node(
                    graph,
                    Node {
                        inputs: vec![Some(output)],
                        outputs: vec![nchw_output],
                        name: format!("MaxPool_NHWC2NCHW_{}", output.index()),
                        op: Operator::NHWC2NCHW,
                        meta: NodeMeta::default(),
                    },
                );
                marked.insert(nhwc2nchw_id);
                modifier.replace_input_value_if_without_typecheck(
                    graph,
                    output,
                    nchw_output,
                    |id, _| id != nhwc2nchw_id && id != node_id,
                );
            }

            Operator::NHWC2NCHW => {
                assert!(cands.is_empty());
            }

            op => {
                assert!(op.is_elementwise());
                for (i, c) in cands.iter().enumerate() {
                    let Some(v) = graph.nodes[node_id].inputs[i] else {
                        continue;
                    };
                    if let Some(def_id) = c {
                        assert!(matches!(graph.nodes[*def_id].op, Operator::NHWC2NCHW));
                        let nhwc_input = graph.nodes[*def_id].inputs[0].unwrap();
                        modifier.replace_input_value_if_without_typecheck(
                            graph,
                            v,
                            nhwc_input,
                            |id, _| id == node_id,
                        );
                    } else {
                        let v = v;
                        let nchw_ty = graph.get_resolved_tensor_type(v).unwrap();
                        let nhwc_ty = nchw_ty.transpose(&[0, 2, 3, 1]);
                        let reinterp_out = modifier.register_new_value(
                            graph,
                            format!("SinkNHWC2NCHW_Reinterp_{}", v.index()),
                            nhwc_ty,
                        );
                        modifier.register_new_node(
                            graph,
                            Node {
                                inputs: vec![Some(v)],
                                outputs: vec![reinterp_out],
                                name: format!("SinkNHWC2NCHW_Reinterp_{}", v.index()),
                                op: Operator::Reinterpret(Reinterpret {
                                    ops: vec![ReinterpretType::Transpose(Transpose {
                                        perm: Some(vec![0, 2, 3, 1]),
                                    })],
                                }),
                                meta: NodeMeta::default(),
                            },
                        );
                        modifier.replace_input_value_if_without_typecheck(
                            graph,
                            v,
                            reinterp_out,
                            |id, _| id == node_id,
                        );
                    }
                }

                // Update output types to NHWC and insert NHWC2NCHW after each output
                let outputs = graph.nodes[node_id].outputs.clone();
                for old_output in outputs {
                    let old_ty = graph.get_resolved_tensor_type(old_output).unwrap().clone();
                    let nhwc_ty = old_ty.transpose(&[0, 2, 3, 1]).contiguous();
                    graph.values[old_output].ty = Some(TensorType::Resolved(nhwc_ty));

                    let nchw_output = modifier.register_new_value(
                        graph,
                        format!("SinkNHWC2NCHW_NCHW_{}", old_output.index()),
                        old_ty,
                    );
                    let nhwc2nchw_id = modifier.register_new_node(
                        graph,
                        Node {
                            inputs: vec![Some(old_output)],
                            outputs: vec![nchw_output],
                            name: format!("SinkNHWC2NCHW_NCHW_{}", old_output.index()),
                            op: Operator::NHWC2NCHW,
                            meta: NodeMeta::default(),
                        },
                    );
                    marked.insert(nhwc2nchw_id);
                    modifier.replace_input_value_if_without_typecheck(
                        graph,
                        old_output,
                        nchw_output,
                        |id, _| id != nhwc2nchw_id && id != node_id,
                    );
                }
            }
        }
    }

    /// NHWC2NCHW -> Conv: set Conv.input_layout=NHWC, remove NHWC2NCHW
    fn fold_into_conv_input<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
        let targets: Vec<(NodeId, Vec<NodeId>)> = graph
            .nodes
            .iter()
            .filter(|(_, node)| matches!(node.op, Operator::NHWC2NCHW))
            .filter_map(|(nchw_id, nchw_node)| {
                let nchw_output = nchw_node.outputs[0];
                let users = modifier.used_node(nchw_output)?;
                let user_ids: Vec<_> = users
                    .iter()
                    .filter_map(|&(user_id, user_input_idx)| {
                        if user_input_idx != 0 {
                            return None;
                        }
                        match &graph.nodes[user_id].op {
                            Operator::Conv(_) => Some(user_id),
                            _ => None,
                        }
                    })
                    .collect();
                if user_ids.len() != users.len() {
                    return None;
                }
                Some((nchw_id, user_ids))
            })
            .collect();

        for (nchw_id, user_ids) in targets {
            let nchw_input = graph.nodes[nchw_id].inputs[0].unwrap();
            let nchw_output = graph.nodes[nchw_id].outputs[0];

            for &conv_id in &user_ids {
                let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                    unreachable!();
                };
                conv.input_layout = Layout::NHWC;
            }

            modifier.replace_input_value_if_without_typecheck(
                graph,
                nchw_output,
                nchw_input,
                |id, _| user_ids.contains(&id),
            );
        }
    }

    /// Revert Conv(output_layout=NHWC) when ALL users of Conv output are NHWC2NCHW:
    /// set Conv.output_layout=NCHW, update type, remove NHWC2NCHW nodes
    fn revert<T: GraphOp>(&self, graph: &mut Graph, modifier: &mut T) {
        let targets: Vec<(NodeId, Vec<NodeId>)> = graph
            .nodes
            .iter()
            .filter_map(|(conv_id, node)| {
                let Operator::Conv(ref conv) = node.op else {
                    return None;
                };
                if conv.output_layout != Layout::NHWC {
                    return None;
                }
                let output = node.outputs[0];
                let users = modifier.used_node(output)?;
                let nhwc2nchw_ids: Vec<NodeId> = users
                    .iter()
                    .filter_map(|&(user_id, _)| {
                        if matches!(graph.nodes[user_id].op, Operator::NHWC2NCHW) {
                            Some(user_id)
                        } else {
                            None
                        }
                    })
                    .collect();
                if nhwc2nchw_ids.len() != users.len() {
                    return None;
                }
                Some((conv_id, nhwc2nchw_ids))
            })
            .collect();

        for (conv_id, nhwc2nchw_ids) in targets {
            let conv_output = graph.nodes[conv_id].outputs[0];

            let Operator::Conv(ref mut conv) = graph.nodes[conv_id].op else {
                unreachable!();
            };
            conv.output_layout = Layout::NCHW;

            let nchw_ty = graph
                .get_resolved_tensor_type(graph.nodes[nhwc2nchw_ids[0]].outputs[0])
                .unwrap()
                .clone();
            graph.values[conv_output].ty = Some(TensorType::Resolved(nchw_ty));

            for nchw_id in nhwc2nchw_ids {
                let nchw_output = graph.nodes[nchw_id].outputs[0];
                modifier.replace_input_value_if_without_typecheck(
                    graph,
                    nchw_output,
                    conv_output,
                    |_, _| true,
                );
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SinkScore {
    Benefit,
    Neutral,
    Forbidden,
}

impl SinkScore {
    fn merge(&self, other: Self) -> Self {
        use SinkScore::*;
        match (self, other) {
            (Forbidden, _) | (_, Forbidden) => Forbidden,
            (Benefit, _) | (_, Benefit) => Benefit,
            (Neutral, Neutral) => Neutral,
        }
    }
}

struct SinkMarker<'a, T: GraphOp> {
    graph: &'a Graph,
    modifier: &'a T,
    memo: HashMap<NodeId, SinkScore>,
    marked: HashSet<NodeId>,
}

impl<'a, T: GraphOp> SinkMarker<'a, T> {
    fn new(graph: &'a Graph, modifier: &'a T) -> Self {
        Self {
            graph,
            modifier,
            memo: HashMap::new(),
            marked: HashSet::new(),
        }
    }

    fn visit(&mut self, node_id: NodeId) {
        if self.memo.contains_key(&node_id) {
            return;
        }

        match &self.graph.nodes[node_id].op {
            Operator::Conv(_) => {
                self.memo.insert(node_id, SinkScore::Benefit);
            }

            Operator::MaxPool(_) => {
                self.memo.insert(node_id, SinkScore::Benefit);
            }

            Operator::NHWC2NCHW => {
                let output = self.graph.nodes[node_id].outputs[0];
                let score = self.modifier.used_node(output).unwrap().iter().fold(
                    SinkScore::Neutral,
                    |acc, (user_id, _)| {
                        let rhs = *self.memo.get(user_id).unwrap_or(&SinkScore::Forbidden);
                        acc.merge(rhs)
                    },
                );
                self.memo.insert(node_id, score);
                if score == SinkScore::Benefit {
                    self.marked.insert(node_id);
                }
            }

            op => {
                if !op.is_elementwise() {
                    self.memo.insert(node_id, SinkScore::Forbidden);
                    return;
                }

                if !self.graph.nodes[node_id]
                    .inputs
                    .iter()
                    .filter_map(|x| x.as_ref())
                    .all(|input| {
                        self.graph
                            .get_resolved_tensor_type(*input)
                            .unwrap()
                            .dims
                            .ndim() ==
                            4
                    })
                {
                    self.memo.insert(node_id, SinkScore::Forbidden);
                    return;
                }

                let ok = self.graph.nodes[node_id].outputs.iter().fold(
                    SinkScore::Neutral,
                    |acc, output| {
                        let rhs = self.modifier.used_node(*output).unwrap().iter().fold(
                            SinkScore::Neutral,
                            |acc, (user_id, _)| {
                                let rhs = *self.memo.get(user_id).unwrap_or(&SinkScore::Forbidden);
                                acc.merge(rhs)
                            },
                        );
                        acc.merge(rhs)
                    },
                );
                self.memo.insert(node_id, ok);
            }
        }
    }

    fn mark(&mut self) {
        let nodes_ids = simple_topological_order(self.graph);
        for id in nodes_ids.iter().rev() {
            self.visit(*id);
        }
    }
}
