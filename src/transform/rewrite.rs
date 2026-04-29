pub mod session_state;

use crate::session::SessionConfig;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::rewrite::session_state::SessionStateRewrite;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;

pub fn create_rewrite_passes(config: &SessionConfig) -> SimplePassManager<SimpleGraphOp> {
    let mut passes = SimplePassManager::new("Rewrite".to_string());
    if !config.session_states.is_empty() {
        passes.add_pass(Box::new(SessionStateRewrite::new(config.clone())));
    }
    passes
}
