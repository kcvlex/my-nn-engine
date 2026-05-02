use crate::graph::operator::Operator;
use crate::graph::Graph;
use crate::session::SessionConfig;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

pub struct SessionStateRewrite {
    config: SessionConfig,
}

impl SessionStateRewrite {
    pub fn new(config: SessionConfig) -> Self {
        Self { config }
    }
}

impl<T: GraphOp> Pass<T> for SessionStateRewrite {
    fn summary(&self) -> &'static str {
        "Convert configured graph inputs to SessionState"
    }

    fn run(&self, graph: &mut Graph, _modifier: &mut T) {
        for spec in &self.config.session_states {
            let Some(node_id) = graph.inputs.iter().copied().find(|&id| {
                let Operator::Input(value_id) = graph.nodes[id].op else {
                    return false;
                };
                graph.values[value_id].name == spec.name
            }) else {
                panic!("SessionStateRewrite: graph input {:?} not found", spec.name);
            };
            let Operator::Input(value_id) = graph.nodes[node_id].op else {
                unreachable!()
            };
            graph.nodes[node_id].op = Operator::SessionState(value_id);
        }
    }
}
