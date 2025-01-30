pub mod gemm;
pub mod im2col;
pub mod omp;

use crate::transform::modify::SimpleGraphModifier;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes(omp_threshold: usize) -> SimplePassManager<SimpleGraphModifier> {
    let mut pass_manager = SimplePassManager::new("Optimize".to_string());
    pass_manager.add_pass(Box::new(im2col::InsertIm2Col::default()));
    pass_manager.add_pass(Box::new(omp::InnermostOMP {
        threshold: omp_threshold,
    }));
    pass_manager.add_pass(Box::new(gemm::TransformBLASGemm::default()));
    pass_manager.add_pass(Box::new(gemm::GemmTransComposition::default()));
    pass_manager
}
