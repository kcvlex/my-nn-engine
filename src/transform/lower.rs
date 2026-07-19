mod contiguous_insertion;
mod decomposition;
mod nhwc2nchw_detection;
mod nhwc2nchw_inertion;
mod nhwc2nchw_sink_and_fold;
pub mod strides;

use crate::options::*;
use crate::transform::lower::contiguous_insertion::ContiguousInsertion;
use crate::transform::lower::decomposition::GlobalAvgPoolDecomposition;
use crate::transform::lower::decomposition::ReduceDecomposition;
use crate::transform::lower::nhwc2nchw_detection::NHWC2NCHWDetection;
use crate::transform::lower::nhwc2nchw_inertion::NHWC2NCHWInsertion;
use crate::transform::lower::nhwc2nchw_sink_and_fold::NHWC2NCHWSinkAndFold;
use crate::transform::lower::strides::AssignStrides;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::verify::ShapeVerification;
use crate::transform::utils::ContiguousElimination;
use crate::transform::utils::ContiguousFolding;
use crate::transform::utils::ReinterpretConversion;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_lower_passes(_opt: &Options, enable_nhwc: bool) -> SimplePassManager<SimpleGraphOp> {
    let mut passes = SimplePassManager::new("Lowering".to_string());
    passes.add_pass(Box::new(ReduceDecomposition::default()));
    passes.add_pass(Box::new(GlobalAvgPoolDecomposition::default()));
    passes.add_pass(Box::new(ContiguousInsertion::default()));
    if enable_nhwc {
        passes.add_pass(Box::new(NHWC2NCHWInsertion::default()));
    }
    passes.add_pass(Box::new(AssignStrides));
    passes.add_pass(Box::new(ShapeVerification {
        check_strides: true,
    }));

    // NHWC optimization
    if enable_nhwc {
        passes.add_pass(Box::new(ReinterpretConversion::default()));
        passes.add_pass(Box::new(ContiguousFolding::backward_only()));
        passes.add_pass(Box::new(ContiguousElimination::default()));
        passes.add_pass(Box::new(ShapeVerification {
            check_strides: true,
        }));
        passes.add_pass(Box::new(NHWC2NCHWDetection::default()));
        passes.add_pass(Box::new(ShapeVerification {
            check_strides: true,
        }));
        passes.add_pass(Box::new(NHWC2NCHWSinkAndFold::default()));
        passes.add_pass(Box::new(ShapeVerification {
            check_strides: true,
        }));
    }

    passes
}
