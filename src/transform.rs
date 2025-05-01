pub mod epilog;
pub mod lower;
pub mod modify;
pub mod optimize;
pub mod shape;
mod utils;

use crate::onnx::model::Graph;
use modify::{GraphModifier, NodeDelete, SimpleGraphModifier};

pub use epilog::create_epilog_passes;
pub use lower::create_lower_passes;
pub use optimize::create_optimize_passes;
pub use shape::infer::create_infer_passes;
pub use shape::strides::create_strides_passes;

pub trait Pass<T: GraphModifier> {
    fn summary(&self) -> &str;
    fn run(&self, graph: &mut Graph, modifier: &mut T);
}

pub trait PassManager<T: GraphModifier + NodeDelete> {
    fn add_pass(&mut self, pass: Box<dyn Pass<T>>);
    fn run(&self, graph: &mut Graph, modifier: &mut T);
    fn name(&self) -> &str;
}

pub struct SimplePassManager<T: GraphModifier + NodeDelete> {
    name: String,
    passes: Vec<Box<dyn Pass<T>>>,
}

impl<T: GraphModifier + NodeDelete> SimplePassManager<T> {
    pub fn new(name: String) -> Self {
        Self {
            name,
            passes: Vec::new(),
        }
    }
}

impl<T: GraphModifier + NodeDelete> PassManager<T> for SimplePassManager<T> {
    fn add_pass(&mut self, pass: Box<dyn Pass<T>>) {
        self.passes.push(pass);
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        println!("SimplePassManager: {}", self.name());
        for opt in self.passes.iter() {
            println!("-- Running pass: {}", opt.summary());
            opt.run(graph, modifier);
            modifier.update_deleted_nodes(graph);
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub fn transform_graph(graph: &mut Graph, omp_threshold: usize) {
    let verify = true;
    // TODO: Make optimization passes before lowering
    let managers = [
        Box::new(create_infer_passes(verify)),
        Box::new(create_lower_passes()),
        Box::new(create_optimize_passes(omp_threshold)),
        Box::new(create_strides_passes(verify)),
        Box::new(create_epilog_passes()),
    ];
    let mut modifier = SimpleGraphModifier::new(graph);
    for manager in managers.iter() {
        manager.run(graph, &mut modifier);
    }

    graph.delete_nodes();
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::Model;
    use crate::transform::optimize::elementwise_fuse::FuseElementwiseOps;
    use std::path::PathBuf;

    #[ignore]
    #[test]
    fn test_save() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/test/optimize");
        let input_name = "elementwise_complex0";
        let input = dir.join(format!("{}.onnx", input_name));
        let mut model = Model::load_from_path(&input).unwrap();
        let graph = &mut model.graph;
        let mut modifier = SimpleGraphModifier::new(graph);
        let infer = create_infer_passes(true);
        let fusion = FuseElementwiseOps::default();
        infer.run(graph, &mut modifier);
        fusion.run(graph, &mut modifier);
        modifier.update_deleted_nodes(graph);
        graph.delete_nodes();
        let output = dir.join(format!("{}.out.onnx", input_name));
        model.save_to_path(&output).unwrap();
    }
}
