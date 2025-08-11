pub mod gemm;
pub mod im2col;

use crate::transform::modify::SimpleGraphOp;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes0() -> SimplePassManager<SimpleGraphOp> {
    let pass_manager = SimplePassManager::new("Optimization before Lowering".to_string());
    // if enable_fuse {
    //     pass_manager.add_pass(Box::new(elementwise_fuse::FuseElementwiseOps::default()));
    // }
    pass_manager
}

pub fn create_optimize_passes1() -> SimplePassManager<SimpleGraphOp> {
    let mut pass_manager =
        SimplePassManager::new("Optimization between Lowering and Strides".to_string());
    pass_manager.add_pass(Box::new(im2col::InsertIm2Col::default()));
    pass_manager.add_pass(Box::new(gemm::GemmTransComposition::default()));
    pass_manager
}
