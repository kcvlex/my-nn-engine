pub mod attention_fusion;
pub mod canonicalize;
pub mod const_fold;
pub mod conv_bn_fusion;
pub mod elim_identity;
pub mod fast_gelu_fusion;
pub mod gemm_add_fusion;
pub mod gemm_transpose_fusion;
pub mod layer_norm_fusion;
pub mod reorder_nodes;
pub mod transpose_fusion;

use crate::transform::modify::SimpleGraphOp;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes() -> SimplePassManager<SimpleGraphOp> {
    let mut pass_manager = SimplePassManager::new("Optimization".to_string());
    pass_manager.add_pass(Box::new(elim_identity::EliminateIdentity::default()));
    pass_manager.add_pass(Box::new(canonicalize::Canonicalize::default()));
    pass_manager.add_pass(Box::new(const_fold::ConstantFold {
        check_strides: false,
    }));
    pass_manager.add_pass(Box::new(conv_bn_fusion::ConvBNFusion::default()));
    pass_manager.add_pass(Box::new(fast_gelu_fusion::FastGeLUFusion::default()));
    pass_manager.add_pass(Box::new(layer_norm_fusion::LayerNormFusion::default()));
    pass_manager.add_pass(Box::new(attention_fusion::AttentionFusion::default()));
    pass_manager.add_pass(Box::new(canonicalize::MatMul2BatchedGemm::default()));
    pass_manager.add_pass(Box::new(reorder_nodes::ReorderNodes::default()));
    pass_manager.add_pass(Box::new(transpose_fusion::TransposeFusion {
        check_strides: false,
    }));
    pass_manager.add_pass(Box::new(canonicalize::Canonicalize::default()));
    pass_manager.add_pass(Box::new(elim_identity::EliminateIdentity::default()));
    pass_manager.add_pass(Box::new(
        gemm_transpose_fusion::GemmTransposeFusion::default(),
    ));
    pass_manager.add_pass(Box::new(gemm_add_fusion::GemmAddFusion::default()));
    pass_manager.add_pass(Box::new(const_fold::ConstantFold {
        check_strides: true,
    }));
    pass_manager
}
