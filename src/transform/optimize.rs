pub mod const_fold;
pub mod const_prop;
pub mod gemm_add_fusion;
pub mod gemm_transpose_fusion;
pub mod im2col;

use crate::options::*;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes1(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut pass_manager =
        SimplePassManager::new("Optimization between Lowering and Strides".to_string());
    // pass_manager.add_pass(Box::new(const_prop::ConstProp::default()));
    if matches!(opt.target, Target::CPU) {
        pass_manager.add_pass(Box::new(im2col::InsertIm2Col::default()));
    }
    pass_manager.add_pass(Box::new(
        gemm_transpose_fusion::GemmTransposeFusion::default(),
    ));
    pass_manager.add_pass(Box::new(gemm_add_fusion::GemmAddFusion::default()));
    pass_manager
}
