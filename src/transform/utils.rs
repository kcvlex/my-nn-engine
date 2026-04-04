use std::io::Error;
use std::io::Result;

use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::NodeMeta;
use crate::onnx::model::ValueId;
use crate::onnx::model::ValueInfo;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::tensor::types::ResolvedTensorDims;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct TransposeGenerator {
    input: Option<ValueId>,
    perm: Option<Vec<usize>>,
    node_name: Option<String>,
    value_name: Option<String>,
    contiguous: Option<bool>,
}

impl TransposeGenerator {
    pub fn set_input(mut self, input: ValueId) -> Self {
        self.input = Some(input);
        self
    }

    pub fn set_perm(mut self, perms: Vec<usize>) -> Self {
        self.perm = Some(perms);
        self
    }

    pub fn set_contiguous(mut self, contiguous: bool) -> Self {
        self.contiguous = Some(contiguous);
        self
    }

    pub fn set_node_name(mut self, node_name: String) -> Self {
        self.node_name = Some(node_name);
        self
    }

    pub fn set_value_name(mut self, value_name: String) -> Self {
        self.value_name = Some(value_name);
        self
    }

    pub fn generate<T: GraphOp>(self, graph: &mut Graph, graph_op: &mut T) -> Result<ValueId> {
        let input = self.input.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "input is required",
        ))?;
        let perm = self.perm.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "perms is required",
        ))?;
        let node_name = self
            .node_name
            .unwrap_or_else(|| format!("Transpose_{}", input.index()));
        let value_name = self
            .value_name
            .unwrap_or_else(|| format!("Transpose_{}", input.index()));
        let contiguous = self.contiguous.unwrap_or(false);

        let input_ty = graph.get_resolved_tensor_type(input).unwrap();
        if input_ty.dims.ndim() != perm.len() {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "perms length must be equal to input rank",
            ));
        }

        let new_ty = input_ty.transpose(&perm);
        let perm = Some(perm);
        let transposed = graph_op.register_new_value(graph, value_name.clone(), new_ty.clone());
        graph_op.register_new_node(
            graph,
            Node {
                inputs: vec![Some(input)],
                outputs: vec![transposed],
                op: Operator::Transpose(Transpose { perm }),
                name: node_name.clone(),
                meta: NodeMeta::default(),
            },
        );

        let new_value = if contiguous {
            let new_ty = new_ty.contiguous();
            let new_value =
                graph_op.register_new_value(graph, format!("{value_name}_Contiguous"), new_ty);
            graph_op.register_new_node(
                graph,
                Node {
                    inputs: vec![Some(transposed)],
                    outputs: vec![new_value],
                    op: Operator::Contiguous(Contiguous { ops: vec![] }),
                    name: format!("{node_name}_Contiguous"),
                    meta: NodeMeta::default(),
                },
            );
            new_value
        } else {
            transposed
        };
        Ok(new_value)
    }
}

#[derive(Default)]
pub struct ReshapeGenerator {
    input: Option<ValueId>,
    dims: Option<ResolvedTensorDims>,
    node_name: Option<String>,
    value_name: Option<String>,
    allow_contiguous: Option<bool>,
}

#[allow(dead_code)]
impl ReshapeGenerator {
    pub fn set_input(mut self, input: ValueId) -> Self {
        self.input = Some(input);
        self
    }

    pub fn set_dims(mut self, dims: &[usize]) -> Self {
        self.dims = Some(ResolvedTensorDims::new(dims));
        self
    }

    pub fn set_allow_contiguous(mut self, allow_contiguous: bool) -> Self {
        self.allow_contiguous = Some(allow_contiguous);
        self
    }

    pub fn set_node_name(mut self, node_name: String) -> Self {
        self.node_name = Some(node_name);
        self
    }

    pub fn set_value_name(mut self, value_name: String) -> Self {
        self.value_name = Some(value_name);
        self
    }

    pub fn generate<T: GraphOp>(self, graph: &mut Graph, graph_op: &mut T) -> Result<ValueId> {
        let allow_contiguous = self.allow_contiguous.unwrap_or(true);
        let input = self.input.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "input is required",
        ))?;
        let dims = self.dims.ok_or(Error::new(
            std::io::ErrorKind::InvalidInput,
            "dims is required",
        ))?;
        let node_name = self
            .node_name
            .unwrap_or_else(|| format!("Reshape_{}", input.index()));
        let value_name = self
            .value_name
            .unwrap_or_else(|| format!("Reshape_{}", input.index()));

        let input_ty = graph.get_resolved_tensor_type(input).unwrap().clone();
        if input_ty.dims.size() != dims.size() &&
            !(input_ty.dims.compatible_with_scalar() && dims.compatible_with_scalar())
        {
            return Err(Error::new(
                std::io::ErrorKind::InvalidInput,
                "dims size must be equal to input size",
            ));
        }
        let reshaped_ty = input_ty.try_reshape(&dims);
        let (input_value, reshaped_ty) = match reshaped_ty {
            Some(ty) => (input, ty),
            None => {
                if !allow_contiguous {
                    return Err(Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "cannot reshape because of uncontiguous area",
                    ));
                } else {
                    let new_ty = input_ty.contiguous();
                    let reshaped_ty = new_ty.try_reshape(&dims).unwrap();
                    let value_name = format!("{value_name}_Continguous");
                    let node_name = format!("{node_name}_Continguous");
                    let new_value = graph_op.register_new_value(graph, value_name, new_ty.clone());
                    graph_op.register_new_node(
                        graph,
                        Node {
                            inputs: vec![Some(input)],
                            outputs: vec![new_value],
                            op: Operator::Contiguous(Contiguous { ops: vec![] }),
                            name: node_name,
                            meta: NodeMeta::default(),
                        },
                    );
                    (new_value, reshaped_ty)
                }
            }
        };

        let shape_name = format!("{value_name}_Shape");
        let shape_input = graph_op.register_new_tensor(graph, dims.to_tensor(), shape_name.clone());
        let new_value = graph_op.register_new_value(graph, value_name, reshaped_ty);
        graph_op.register_new_node(
            graph,
            Node {
                inputs: vec![Some(input_value), Some(shape_input)],
                outputs: vec![new_value],
                op: Operator::Reshape,
                name: node_name,
                meta: NodeMeta::default(),
            },
        );
        Ok(new_value)
    }
}

#[allow(dead_code)]
#[derive(Default)]
pub struct ContiguousOutput {}

// This pass is assumed to be run before shape inference
impl<T: GraphOp> Pass<T> for ContiguousOutput {
    fn summary(&self) -> &'static str {
        "Insert contiguous before all Outputs"
    }

    fn run(&self, graph: &mut Graph, graph_op: &mut T) {
        let ids = graph.outputs.clone();
        for id in ids.iter() {
            let input = graph.nodes[*id].inputs[0].unwrap();
            let input_ty = graph.values[input].ty.clone();
            let new_value = graph.values.alloc(ValueInfo {
                name: format!("Contiguous_Output_{}", id.index()),
                ty: input_ty,
            });
            graph_op.register_new_node(
                graph,
                Node {
                    inputs: vec![Some(input)],
                    outputs: vec![new_value],
                    op: Operator::Contiguous(Contiguous { ops: vec![] }),
                    name: format!("Contiguous_Output_{}", id.index()),
                    meta: NodeMeta::default(),
                },
            );
            graph_op.replace_input_value_if_without_typecheck(
                graph,
                input,
                new_value,
                |_, node| matches!(node.op, Operator::Output(_)),
            );

            // Forget the dimension information of old output to make shape inference easier.
            // TODO: Maybe incorrect if the Input node is directly connected to the Output node.
            graph.values[input].ty = None;
        }
    }
}

#[derive(Default)]
pub struct ReinterpretConversion {}

fn bundle_reshape_and_transpose<T: GraphOp>(
    graph: &Graph,
    node_id: NodeId,
    modifier: &T,
) -> (Reinterpret, ValueId) {
    let input_value = graph.nodes[node_id].inputs[0].unwrap();
    let (source, chain) = modifier.walk_chain_backward(graph, input_value, |node| {
        matches!(node.op, Operator::Reshape | Operator::Transpose(_))
    });

    let all_nodes: Vec<NodeId> = chain
        .into_iter()
        .rev()
        .chain(std::iter::once(node_id))
        .collect();
    let ops = all_nodes
        .iter()
        .map(|&id| match &graph.nodes[id].op {
            Operator::Reshape => {
                let input_shape = graph
                    .get_resolved_tensor_type(graph.nodes[id].inputs[0].unwrap())
                    .unwrap();
                let output_shape = graph
                    .get_resolved_tensor_type(graph.nodes[id].outputs[0])
                    .unwrap();
                ReinterpretType::Reshape {
                    before: input_shape.dims.iter().copied().collect(),
                    after: output_shape.dims.iter().copied().collect(),
                }
            }
            Operator::Transpose(perm) => ReinterpretType::Transpose(perm.clone()),
            _ => unreachable!(),
        })
        .collect();

    (Reinterpret { ops }, source)
}

impl<T: GraphOp> Pass<T> for ReinterpretConversion {
    fn summary(&self) -> &'static str {
        "Convert Reshape/Transpose to Reinterpret"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = simple_topological_order(graph)
            .into_iter()
            .filter(|id| {
                matches!(
                    graph.nodes[*id].op,
                    Operator::Reshape | Operator::Transpose(_)
                )
            })
            .rev()
            .collect_vec();

        for id in ids.iter() {
            let old_output = graph.nodes[*id].outputs[0];
            let Some(to_bundle) = modifier.used_node(old_output) else {
                continue;
            };
            let to_bundle = to_bundle.iter().any(|(user_id, _)| {
                !matches!(
                    graph.nodes[*user_id].op,
                    Operator::Reshape | Operator::Transpose(_)
                )
            });
            if !to_bundle {
                continue;
            }

            let (re, input) = bundle_reshape_and_transpose(graph, *id, modifier);
            let new_output = modifier.register_new_value(
                graph,
                format!("Reinterpret_{:?}", old_output),
                graph.get_resolved_tensor_type(old_output).unwrap().clone(),
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    vec![Some(input)],
                    vec![new_output],
                    format!("Reinterpret_{:?}", id),
                    Operator::Reinterpret(re),
                ),
            );
            modifier.replace_input_value(graph, old_output, new_output);
        }
    }
}

#[derive(Default)]
pub struct ContiguousElimination {}

impl<T: GraphOp> Pass<T> for ContiguousElimination {
    fn summary(&self) -> &'static str {
        "Eliminate unnecessary Contiguous nodes"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph
            .nodes
            .iter()
            .filter(|(_, node)| match node.op {
                Operator::Contiguous(_) => {
                    let input_ty = graph
                        .get_resolved_tensor_type(node.inputs[0].unwrap())
                        .unwrap();
                    let output_ty = graph.get_resolved_tensor_type(node.outputs[0]).unwrap();
                    input_ty.is_contiguous() && input_ty.dims == output_ty.dims
                }
                _ => false,
            })
            .map(|(id, _node)| id)
            .collect::<Vec<_>>();

        for id in ids.iter() {
            let node = &graph.nodes[*id];
            let input = node.inputs[0].unwrap();
            let output = node.outputs[0];
            modifier.replace_input_value(graph, output, input);
        }
    }
}

fn is_reinterpret_or_contiguous(node: &Node) -> bool {
    matches!(node.op, Operator::Reinterpret(_) | Operator::Contiguous(_))
}

fn extract_ops(node: &Node) -> &[ReinterpretType] {
    match &node.op {
        Operator::Reinterpret(re) => &re.ops,
        Operator::Contiguous(cont) => &cont.ops,
        _ => &[],
    }
}

pub struct ContiguousFolding {
    pub fold_forward: bool,
}

impl Default for ContiguousFolding {
    fn default() -> Self {
        Self { fold_forward: true }
    }
}

impl ContiguousFolding {
    pub fn backward_only() -> Self {
        Self {
            fold_forward: false,
        }
    }
}

impl<T: GraphOp> Pass<T> for ContiguousFolding {
    fn summary(&self) -> &'static str {
        "Fold Contiguous/Reinterpret chains"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let topo = simple_topological_order(graph);

        let chain_starts: Vec<NodeId> = topo
            .into_iter()
            .filter(|id| {
                let node = &graph.nodes[*id];
                if !is_reinterpret_or_contiguous(node) {
                    return false;
                }
                let input = node.inputs[0].unwrap();
                let Some((def_id, _)) = modifier.defined_node(input) else {
                    return true;
                };
                if !is_reinterpret_or_contiguous(&graph.nodes[def_id]) {
                    return true;
                }
                modifier.used_node(input).is_none_or(|u| u.len() != 1)
            })
            .collect();

        for start_id in chain_starts {
            let (all_nodes, final_output) = if self.fold_forward {
                let start_output = graph.nodes[start_id].outputs[0];
                let (final_output, forward_chain) =
                    modifier.walk_chain_forward(graph, start_output, is_reinterpret_or_contiguous);
                let all_nodes: Vec<NodeId> =
                    std::iter::once(start_id).chain(forward_chain).collect();
                (all_nodes, final_output)
            } else {
                let output = graph.nodes[start_id].outputs[0];
                (vec![start_id], output)
            };

            let has_contiguous = all_nodes
                .iter()
                .any(|&id| matches!(graph.nodes[id].op, Operator::Contiguous(_)));
            if !has_contiguous {
                continue;
            }

            if all_nodes.len() == 1 {
                continue;
            }

            let all_ops: Vec<ReinterpretType> = all_nodes
                .iter()
                .flat_map(|&id| extract_ops(&graph.nodes[id]).iter().cloned())
                .collect();

            let true_input = graph.nodes[start_id].inputs[0].unwrap();
            let output_ty = graph
                .get_resolved_tensor_type(final_output)
                .unwrap()
                .clone();
            let new_output = modifier.register_new_value(
                graph,
                format!("FoldedContiguous_{:?}", start_id),
                output_ty,
            );
            modifier.register_new_node(
                graph,
                Node::create_node(
                    vec![Some(true_input)],
                    vec![new_output],
                    format!("FoldedContiguous_{:?}", start_id),
                    Operator::Contiguous(Contiguous { ops: all_ops }),
                ),
            );
            modifier.replace_input_value(graph, final_output, new_output);
        }
    }
}
