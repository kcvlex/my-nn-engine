pub mod attention_fusion;
pub mod canonicalization;
pub mod const_folding;
pub mod conv_activation_fusion;
pub mod conv_bn_fusion;
pub mod cse_reshape;
pub mod dequant_gemm_fusion;
pub mod fast_gelu_fusion;
pub mod gemm_add_fusion;
pub mod gemm_transpose_fusion;
pub mod identity_elimination;
pub mod layer_norm_fusion;
pub mod nodes_reorder;
pub mod quantize_activations;
pub mod rms_norm_fusion;
pub mod transpose_fusion;

use crate::options::Options;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::optimize::attention_fusion::AttentionFusion;
use crate::transform::optimize::canonicalization::Canonicalization;
use crate::transform::optimize::canonicalization::MatMul2BatchedGemm;
use crate::transform::optimize::const_folding::ConstantFolding;
use crate::transform::optimize::conv_activation_fusion::ConvActivationFusion;
use crate::transform::optimize::conv_bn_fusion::ConvBNFusion;
use crate::transform::optimize::cse_reshape::CseReshape;
use crate::transform::optimize::dequant_gemm_fusion::DequantGemmFusion;
use crate::transform::optimize::fast_gelu_fusion::FastGeLUFusion;
use crate::transform::optimize::gemm_add_fusion::GemmAddFusion;
use crate::transform::optimize::gemm_transpose_fusion::GemmTransposeFusion;
use crate::transform::optimize::identity_elimination::IdentityElimination;
use crate::transform::optimize::layer_norm_fusion::LayerNormFusion;
use crate::transform::optimize::nodes_reorder::ReorderNodes;
use crate::transform::optimize::quantize_activations::QuantizeActivations;
use crate::transform::optimize::rms_norm_fusion::RMSNormFusion;
use crate::transform::optimize::transpose_fusion::TransposeFusion;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_optimize_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut pass_manager = SimplePassManager::new("Optimization".to_string());
    pass_manager.add_pass(Box::new(IdentityElimination::default()));
    pass_manager.add_pass(Box::new(Canonicalization::default()));
    pass_manager.add_pass(Box::new(ConstantFolding {
        check_strides: false,
    }));
    pass_manager.add_pass(Box::new(ConvBNFusion::default()));
    if opt.target.codegen_device() == crate::schedule::ir::Device::CUDA {
        pass_manager.add_pass(Box::new(ConvActivationFusion::default()));
    }
    pass_manager.add_pass(Box::new(FastGeLUFusion::default()));
    pass_manager.add_pass(Box::new(LayerNormFusion::default()));
    pass_manager.add_pass(Box::new(RMSNormFusion::default()));
    pass_manager.add_pass(Box::new(AttentionFusion::default()));
    pass_manager.add_pass(Box::new(MatMul2BatchedGemm::default()));
    pass_manager.add_pass(Box::new(ReorderNodes::default()));
    pass_manager.add_pass(Box::new(TransposeFusion {
        check_strides: false,
    }));
    pass_manager.add_pass(Box::new(Canonicalization::default()));
    pass_manager.add_pass(Box::new(IdentityElimination::default()));
    pass_manager.add_pass(Box::new(GemmTransposeFusion::default()));
    pass_manager.add_pass(Box::new(GemmAddFusion::default()));
    pass_manager.add_pass(Box::new(DequantGemmFusion::default()));
    if opt.quantize_activations {
        pass_manager.add_pass(Box::new(CseReshape::default()));
        pass_manager.add_pass(Box::new(QuantizeActivations::default()));
    }
    pass_manager.add_pass(Box::new(ConstantFolding {
        check_strides: true,
    }));
    pass_manager
}
