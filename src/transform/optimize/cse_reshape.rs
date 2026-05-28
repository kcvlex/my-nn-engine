use std::collections::HashMap;

use itertools::Itertools;

use crate::graph::operator::*;
use crate::graph::Graph;
use crate::graph::ValueId;
use crate::tensor::data::TensorData;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

#[derive(Default)]
pub struct CseReshape {}

impl<T: GraphOp> Pass<T> for CseReshape {
    fn summary(&self) -> &'static str {
        "CSE Reshape nodes with the same input and target shape"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let reshape_ids = graph
            .nodes
            .iter()
            .filter_map(|(id, n)| match n.op {
                Operator::Reshape => Some(id),
                _ => None,
            })
            .collect_vec();

        let mut seen: HashMap<(ValueId, Vec<i64>), ValueId> = HashMap::new();
        for id in reshape_ids {
            let node = &graph.nodes[id];
            let Some(input) = node.inputs.first().and_then(|x| *x) else {
                continue;
            };
            let Some(shape_input) = node.inputs.get(1).and_then(|x| *x) else {
                continue;
            };
            let Some(shape_tensor) = graph.get_initializer(shape_input) else {
                continue;
            };
            let TensorData::SInt(_, ref shape_vals) = shape_tensor.data else {
                continue;
            };
            let key = (input, shape_vals.clone());
            let output = node.outputs[0];
            match seen.get(&key) {
                Some(&canonical) if canonical != output => {
                    modifier.replace_input_value(graph, output, canonical);
                }
                _ => {
                    seen.insert(key, output);
                }
            }
        }
    }
}
