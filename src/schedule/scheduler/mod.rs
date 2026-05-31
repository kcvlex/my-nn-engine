//! Schedule-pass schedulers. The default [`MemoryAwareSchedulePass`] keeps every
//! weight resident; variants share planning infrastructure in [`common`].

mod common;
mod memory_aware;
pub mod prefetch;

pub use common::PlacementStrategy;
pub use memory_aware::MemoryAwareSchedulePass;
pub use prefetch::PrefetchPolicy;
