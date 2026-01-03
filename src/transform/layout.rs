pub mod strides;

use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::verify;
use crate::transform::Options;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_layout_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Layout".to_string());
    manager.add_pass(Box::new(strides::AssignStrides { target: opt.target }));
    if opt.verify_after_strides {
        manager.add_pass(Box::new(verify::VerifyShape {
            target: opt.target,
            check_strides: true,
        }));
    }
    manager
}
