pub mod epilog;
pub mod lower;
pub mod modify;
pub mod optimize;
mod pattern;
pub mod shape;
pub mod utils;

pub use epilog::create_epilog_passes;
use log::info;
pub use lower::create_lower_passes;
use modify::GraphOp;
use modify::NodeDelete;
use modify::SimpleGraphOp;
pub use optimize::create_optimize_passes;
pub use shape::create_infer_passes;

use crate::onnx::model::Graph;
use crate::onnx::operator::Operator;
use crate::options::*;

pub trait Pass<T: GraphOp> {
    fn summary(&self) -> &str;
    fn run(&self, graph: &mut Graph, modifier: &mut T);
}

pub trait PassManager<T: GraphOp + NodeDelete> {
    fn add_pass(&mut self, pass: Box<dyn Pass<T>>);
    fn run(&self, graph: &mut Graph, modifier: &mut T);
    fn name(&self) -> &str;
}

pub struct SimplePassManager<T: GraphOp + NodeDelete> {
    name: String,
    passes: Vec<Box<dyn Pass<T>>>,
}

impl<T: GraphOp + NodeDelete> SimplePassManager<T> {
    pub fn new(name: String) -> Self {
        Self {
            name,
            passes: Vec::new(),
        }
    }
}

impl<T: GraphOp + NodeDelete> PassManager<T> for SimplePassManager<T> {
    fn add_pass(&mut self, pass: Box<dyn Pass<T>>) {
        self.passes.push(pass);
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        info!("SimplePassManager: {}", self.name());
        for opt in self.passes.iter() {
            info!("-- Running pass: {}", opt.summary());
            opt.run(graph, modifier);
            modifier.update_deleted_nodes(graph);

            // TODO: Move this assertion to somewhere else.
            for (id, _) in graph.values.inner().iter() {
                if let Some(shape) = graph.get_resolved_tensor_type(id) {
                    for (dim, stride) in shape.dims.iter().zip(shape.strides().iter()) {
                        if *dim == 1 {
                            assert_eq!(*stride, 0);
                        }
                    }
                }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub fn transform_graph(graph: &mut Graph, options: &Options) {
    let enable_nhwc = options.enable_nhwc_optimization.unwrap_or_else(|| {
        let has_conv = graph
            .nodes
            .iter()
            .any(|(_, node)| matches!(node.op, Operator::Conv(_)));
        let all_inputs_4d = graph.inputs.iter().all(|&id| {
            let crate::onnx::operator::Operator::Input(value_id) = graph.nodes[id].op else {
                return false;
            };
            graph
                .get_resolved_tensor_type(value_id)
                .map(|ty| ty.dims.ndim() == 4)
                .unwrap_or(false)
        });
        has_conv && all_inputs_4d
    });

    let managers = [
        create_infer_passes(options),
        create_optimize_passes(),
        create_lower_passes(options, enable_nhwc),
        create_epilog_passes(options),
    ];

    let mut modifier = SimpleGraphOp::new(graph);
    for manager in managers.iter() {
        manager.run(graph, &mut modifier);
    }

    graph.delete_nodes();
}
