use crate::onnx::model::Graph;
use crate::onnx::model::Node;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::transform::modify::GraphOp;

pub struct PatternMatcher<'a, T: GraphOp> {
    graph: &'a Graph,
    modifier: &'a T,
    current_value: ValueId,
    last_node: NodeId,
}

impl<'a, T: GraphOp> PatternMatcher<'a, T> {
    pub fn new(graph: &'a Graph, modifier: &'a T, out: (NodeId, usize)) -> Self {
        let (node_id, idx) = out;
        let output = graph.nodes[node_id].outputs[idx];
        PatternMatcher {
            graph,
            modifier,
            last_node: node_id,
            current_value: output,
        }
    }

    pub fn then<F>(mut self, pred: F) -> Option<Self>
    where
        F: Fn((&Node, ValueId)) -> bool,
    {
        for user in self.modifier.used_node(self.current_value)? {
            let (node_id, _) = user;
            let node = &self.graph.nodes[*node_id];
            if pred((node, self.current_value)) {
                self.last_node = *node_id;
                self.current_value = node.outputs[0];
                return Some(self);
            }
        }

        None
    }

    pub fn last_node(&self) -> NodeId {
        self.last_node
    }

    pub fn capture_value(self, m: &mut Option<ValueId>) -> Self {
        *m = Some(self.current_value);
        self
    }

    pub fn capture_node(self, m: &mut Option<NodeId>) -> Self {
        *m = Some(self.last_node);
        self
    }
}

pub(crate) fn extract_other_binary_input(node: &Node, known_input: ValueId) -> Option<ValueId> {
    if node.inputs.len() != 2 {
        return None;
    }

    if node.inputs[0] == known_input {
        Some(node.inputs[1])
    } else if node.inputs[1] == known_input {
        Some(node.inputs[0])
    } else {
        None
    }
}
