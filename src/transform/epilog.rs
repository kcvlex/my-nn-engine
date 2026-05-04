mod cleanup;
mod lower_nhwc2nchw;

use crate::options::*;
use crate::transform::epilog::cleanup::CleanupTensors;
use crate::transform::epilog::lower_nhwc2nchw::NHWC2NCHWLowering;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::utils::ContiguousElimination;
use crate::transform::utils::ContiguousFolding;
use crate::transform::utils::ReinterpretConversion;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_epilog_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Epilog".to_string());
    manager.add_pass(Box::new(NHWC2NCHWLowering::default()));
    manager.add_pass(Box::new(ReinterpretConversion::default()));
    if matches!(opt.target, Target::CUDA) {
        manager.add_pass(Box::new(ContiguousElimination::default()));
    }
    manager.add_pass(Box::new(ContiguousFolding::default()));
    manager.add_pass(Box::new(CleanupTensors::default()));
    manager
}
