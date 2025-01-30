pub mod epilog;
pub mod infer;
pub mod lower;
pub mod modify;
pub mod optimize;
mod utils;

use crate::onnx::model::Graph;
use modify::{GraphModifier, NodeDelete, SimpleGraphModifier};

pub use epilog::create_epilog_passes;
pub use infer::create_infer_passes;
pub use lower::create_lower_passes;
pub use optimize::create_optimize_passes;

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
    // // TODO: Make optimization passes before lowering
    // let managers = [
    //     Box::new(create_infer_passes()),
    //     Box::new(create_lower_passes()),
    //     Box::new(create_optimize_passes(omp_threshold)),
    //     Box::new(create_epilog_passes()),
    // ];
    // let mut modifier = SimpleGraphModifier::new(graph);
    // for manager in managers.iter() {
    //     manager.run(graph, &mut modifier);
    // }

    // TODO: ResNet fails when the order of InsertIm2Col and DecomposeBatchNormalization is swapped
    let mut modifier = SimpleGraphModifier::new(graph);
    let passes: &[Box<dyn Pass<SimpleGraphModifier>>] = &[
        Box::new(utils::tensor::ContigousOutput::default()),
        Box::new(infer::ShapeInference::default()),
        Box::new(optimize::im2col::InsertIm2Col::default()),
        Box::new(lower::DecomposeBatchNormalization::default()),
        Box::new(lower::Reduce2ReduceMatrix::default()),
        Box::new(lower::EliminateGlobalAvgPool::default()),
        Box::new(lower::MatMul2Gemm::default()),
        Box::new(optimize::gemm::TransformBLASGemm::default()),
        Box::new(optimize::gemm::GemmTransComposition::default()),
        Box::new(optimize::omp::InnermostOMP {
            threshold: omp_threshold,
        }),
        Box::new(epilog::Ops2Identity::default()),
    ];
    for pass in passes.iter() {
        pass.run(graph, &mut modifier);
        modifier.update_deleted_nodes(graph);
    }

    graph.delete_nodes();
}
