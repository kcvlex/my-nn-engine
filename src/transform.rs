pub mod epilog;
pub mod layout;
pub mod lower;
pub mod modify;
pub mod optimize;
pub mod shape;
mod utils;

pub use epilog::create_epilog_passes;
pub use layout::create_layout_passes;
pub use lower::create_lower_passes;
use modify::GraphOp;
use modify::NodeDelete;
use modify::SimpleGraphOp;
pub use optimize::create_optimize_passes1;
pub use shape::create_infer_passes;

use crate::onnx::model::Graph;
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
        println!("SimplePassManager: {}", self.name());
        for opt in self.passes.iter() {
            println!("-- Running pass: {}", opt.summary());
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
    let managers = [
        create_infer_passes(options),
        create_lower_passes(),
        create_optimize_passes1(options),
        create_layout_passes(options),
        create_epilog_passes(options),
    ];

    let mut modifier = SimpleGraphOp::new(graph);
    for manager in managers.iter() {
        manager.run(graph, &mut modifier);
    }

    graph.delete_nodes();
}

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::onnx::load::*;
//     use crate::onnx::model::Model;
//     use std::path::PathBuf;
//
//     #[ignore]
//     #[test]
//     fn test_save() {
//         let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/test/optimize");
//         let input_name = "elementwise_complex0";
//         let input = dir.join(format!("{}.onnx", input_name));
//         let mut model = Model::load_from_path(&input).unwrap();
//         let graph = &mut model.graph;
//         let mut modifier = SimpleGraphOp::new(graph);
//         let infer = create_infer_passes(true);
//         let fusion = FuseElementwiseOps::default();
//         infer.run(graph, &mut modifier);
//         fusion.run(graph, &mut modifier);
//         modifier.update_deleted_nodes(graph);
//         graph.delete_nodes();
//         let output = dir.join(format!("{}.out.onnx", input_name));
//         model.save_to_path(&output).unwrap();
//     }
// }
