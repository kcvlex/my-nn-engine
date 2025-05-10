pub mod elementwise_fuse;
pub mod gemm;
pub mod im2col;
pub mod omp;

use crate::transform::modify::SimpleGraphModifier;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes0(enable_fuse: bool) -> SimplePassManager<SimpleGraphModifier> {
    let mut pass_manager = SimplePassManager::new("Optimization before Lowering".to_string());
    if enable_fuse {
        pass_manager.add_pass(Box::new(elementwise_fuse::FuseElementwiseOps::default()));
    }
    pass_manager
}

pub fn create_optimize_passes1() -> SimplePassManager<SimpleGraphModifier> {
    let mut pass_manager =
        SimplePassManager::new("Optimization between Lowering and Strides".to_string());
    pass_manager.add_pass(Box::new(im2col::InsertIm2Col::default()));
    pass_manager.add_pass(Box::new(gemm::GemmTransComposition::default()));
    pass_manager
}

pub fn create_optimize_passes2(omp_threshold: usize) -> SimplePassManager<SimpleGraphModifier> {
    let mut pass_manager = SimplePassManager::new("Optimization after Strides".to_string());
    pass_manager.add_pass(Box::new(omp::InnermostOMP {
        threshold: omp_threshold,
    }));
    pass_manager
}
