use itertools::Itertools;

use crate::onnx::model::Graph;
use crate::onnx::model::ValueId;
use crate::onnx::operator::args;
use crate::onnx::operator::Constant;
use crate::onnx::operator::Operator;
use crate::onnx::operator::Squeeze;
use crate::onnx::operator::TensorIndex;
use crate::onnx::operator::Unsqueeze;
use crate::tensor::data::TensorData;
use crate::tensor::types::SIntType;
use crate::tensor::Tensor;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

// Rewrites ops whose schema changed between opsets into a canonical
// attribute-based form so downstream passes don't need to care about the opset.
// Currently handles:
//   - Unsqueeze: axes moved from attribute (<=12) to input[1] (>=13).
//   - Squeeze:   axes moved from attribute (<=12) to input[1] (>=13).
//   - ReduceMax/ReduceMean/ReduceSum: axes moved to input[1] (>=13/18).
#[derive(Default)]
pub struct OpsetAdaptation {}

impl<T: GraphOp> Pass<T> for OpsetAdaptation {
    fn summary(&self) -> &'static str {
        "Adapt opset differences"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = graph.nodes.iter().map(|(id, _)| id).collect_vec();
        for id in ids {
            match &graph.nodes[id].op {
                Operator::Unsqueeze(_) => bake_unsqueeze_axes(graph, id, modifier),
                Operator::Squeeze(_) => bake_squeeze_axes(graph, id, modifier),
                Operator::ReduceMax(_) | Operator::ReduceMean(_) | Operator::ReduceSum(_) => {
                    bake_reduce_axes(graph, id, modifier)
                }
                _ => (),
            }
        }
    }
}

fn lookup_constant_tensor(graph: &Graph, value_id: ValueId) -> Option<Tensor> {
    if let Some(t) = graph.get_initializer(value_id) {
        return Some(t.clone());
    }
    for (_, node) in graph.nodes.iter() {
        if !node.outputs.contains(&value_id) {
            continue;
        }
        if let Operator::Constant(Constant { value }) = &node.op {
            return Some(value.clone());
        }
    }
    None
}

fn extract_axes(
    graph: &Graph,
    node_id: crate::onnx::model::NodeId,
    arg: usize,
) -> Option<Vec<TensorIndex>> {
    let value_id = *graph.nodes[node_id].inputs.get(arg)?.as_ref()?;
    let tensor = lookup_constant_tensor(graph, value_id)?;
    match &tensor.data {
        TensorData::SInt(SIntType::I64, v) => {
            Some(v.iter().map(|&x| TensorIndex::new(x as isize)).collect())
        }
        _ => None,
    }
}

fn bake_unsqueeze_axes<T: GraphOp>(
    graph: &mut Graph,
    node_id: crate::onnx::model::NodeId,
    modifier: &mut T,
) {
    let Some(axes) = extract_axes(graph, node_id, args::UNSQUEEZE_AXES) else {
        return;
    };
    if let Operator::Unsqueeze(Unsqueeze { axes: existing }) = &mut graph.nodes[node_id].op {
        if existing.is_empty() {
            *existing = axes;
        }
    }
    modifier.drop_node_input(graph, node_id, args::UNSQUEEZE_AXES);
}

fn bake_squeeze_axes<T: GraphOp>(
    graph: &mut Graph,
    node_id: crate::onnx::model::NodeId,
    modifier: &mut T,
) {
    let Some(axes) = extract_axes(graph, node_id, args::SQUEEZE_AXES) else {
        return;
    };
    if let Operator::Squeeze(Squeeze { axes: existing }) = &mut graph.nodes[node_id].op {
        if existing.is_none() {
            *existing = Some(axes);
        }
    }
    modifier.drop_node_input(graph, node_id, args::SQUEEZE_AXES);
}

fn bake_reduce_axes<T: GraphOp>(
    graph: &mut Graph,
    node_id: crate::onnx::model::NodeId,
    modifier: &mut T,
) {
    let Some(axes) = extract_axes(graph, node_id, args::REDUCE_AXES) else {
        return;
    };
    let axes_i64: Vec<i64> = axes.iter().map(|a| a.raw() as i64).collect();
    let reduce = match &mut graph.nodes[node_id].op {
        Operator::ReduceMax(r) | Operator::ReduceMean(r) | Operator::ReduceSum(r) => r,
        _ => return,
    };
    if reduce.axes.is_empty() {
        reduce.axes = axes_i64;
    }
    modifier.drop_node_input(graph, node_id, args::REDUCE_AXES);
}
