pub mod fold_cont;
pub mod insert_cont;
pub mod ops2reinterpret;
pub mod strides;

use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::verify;
use crate::transform::Options;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_layout_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Layout".to_string());
    manager.add_pass(Box::new(insert_cont::InsertContiguous::default()));
    manager.add_pass(Box::new(strides::AssignStrides { target: opt.target }));
    if opt.verify_after_strides {
        manager.add_pass(Box::new(verify::VerifyShape {
            target: opt.target,
            check_strides: true,
        }));
    }
    manager.add_pass(Box::new(ops2reinterpret::Ops2Reinterpret::default()));
    manager.add_pass(Box::new(fold_cont::FoldContiguous::backward_only()));
    manager
}
