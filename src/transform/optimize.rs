pub mod attention_fusion;
pub mod canonicalize;
pub mod const_fold;
pub mod elim_identity;
pub mod fast_gelu_fusion;
pub mod gemm_add_fusion;
pub mod gemm_transpose_fusion;
pub mod im2col;
pub mod layer_norm_fusion;
pub mod reorder_nodes;
pub mod transpose_fusion;

use crate::options::*;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes0(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut pass_manager = SimplePassManager::new("Optimization before Lowering".to_string());
    pass_manager.add_pass(Box::new(elim_identity::EliminateIdentity::default()));
    pass_manager.add_pass(Box::new(canonicalize::Canonicalize::default()));
    pass_manager.add_pass(Box::new(const_fold::ConstantFold {
        check_strides: false,
    }));
    pass_manager.add_pass(Box::new(fast_gelu_fusion::FastGeLUFusion::default()));
    pass_manager.add_pass(Box::new(layer_norm_fusion::LayerNormFusion::default()));
    pass_manager.add_pass(Box::new(attention_fusion::AttentionFusion::default()));
    pass_manager.add_pass(Box::new(reorder_nodes::ReorderNodes::default()));
    pass_manager.add_pass(Box::new(transpose_fusion::TransposeFusion {
        check_strides: false,
    }));
    pass_manager
}

pub fn create_optimize_passes1(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut pass_manager =
        SimplePassManager::new("Optimization between Lowering and Strides".to_string());
    pass_manager.add_pass(Box::new(canonicalize::Canonicalize::default()));
    if matches!(opt.target, Target::CPU) {
        pass_manager.add_pass(Box::new(im2col::InsertIm2Col::default()));
    }
    pass_manager.add_pass(Box::new(elim_identity::EliminateIdentity::default()));
    pass_manager.add_pass(Box::new(
        gemm_transpose_fusion::GemmTransposeFusion::default(),
    ));
    pass_manager.add_pass(Box::new(gemm_add_fusion::GemmAddFusion::default()));
    pass_manager
}
