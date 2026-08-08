//! Process-global communicator used by collective kernels (AllReduce).
//!
//! Tensor parallelism runs one process per rank; the session's generated code
//! reaches the communicator through this global because CPU kernel signatures
//! carry no context argument.

use std::sync::Arc;
use std::sync::RwLock;

use my_nn_engine_comm::Communicator;

static COMMUNICATOR: RwLock<Option<Arc<dyn Communicator>>> = RwLock::new(None);

pub fn set_communicator(comm: Arc<dyn Communicator>) {
    *COMMUNICATOR.write().unwrap() = Some(comm);
}

pub(crate) fn communicator() -> Option<Arc<dyn Communicator>> {
    COMMUNICATOR.read().unwrap().clone()
}
