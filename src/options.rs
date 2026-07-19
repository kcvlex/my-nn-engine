use typed_builder::TypedBuilder;

use crate::schedule::ir::Device;
pub use crate::schedule::scheduler::PlacementStrategy;
pub use crate::schedule::scheduler::PrefetchPolicy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    CPU,
    /// All compute on the GPU. Weights not kept resident by the policy are
    /// streamed from host on demand (`PrefetchPolicy::Disabled` = fully
    /// VRAM-resident).
    CUDA(PrefetchPolicy),
    /// Kernels split across CPU and GPU by the placement; runs on the hybrid
    /// runtime.
    Hybrid(PlacementStrategy),
}

impl Target {
    /// Device the graph is transformed/lowered for. A hybrid target follows
    /// its placement; GPU-primary strategies lower as CUDA.
    pub fn codegen_device(&self) -> Device {
        match self {
            Target::CPU => Device::CPU,
            Target::CUDA(_) => Device::CUDA,
            Target::Hybrid(PlacementStrategy::Uniform(d)) => *d,
            Target::Hybrid(_) => Device::CUDA,
        }
    }

    /// Placement used by the memory-aware scheduler.
    pub fn placement_strategy(&self) -> PlacementStrategy {
        match self {
            Target::CPU => PlacementStrategy::Uniform(Device::CPU),
            Target::CUDA(_) => PlacementStrategy::Uniform(Device::CUDA),
            Target::Hybrid(s) => *s,
        }
    }
}

#[derive(Clone, Debug, TypedBuilder)]
pub struct Options {
    #[builder(default = 100000)]
    pub omp_elementwise_threshold: usize,

    #[builder(default = 64)]
    pub omp_softmax_threshold: usize,

    #[builder(default = true)]
    pub enable_fuse_ops: bool,

    #[builder(default = Target::CPU)]
    pub target: Target,

    #[builder(default = 16)]
    pub num_cuda_streams: usize,

    #[builder(default)]
    pub enable_nhwc_optimization: Option<bool>,

    #[builder(default = false)]
    pub profile: bool,

    #[builder(default = std::env::var("MY_ONNX_SAVE_BUILD_DIR").is_ok())]
    pub save_build_dir: bool,

    #[builder(default = std::env::var("MY_ONNX_SAVE_BUILD_DIR").is_ok())]
    pub save_transformed_model: bool,

    /// If true, rewrite every `DequantMatMul` so its activation input is first
    /// passed through a `DynamicQuantizeLinear` (symmetric per-row int8) and
    /// then consumed by a `QuantizedMatMul` instead. Requires K % 32 == 0
    /// (CUDA INT8 mma kernel constraint); nodes that don't qualify are left
    /// alone. CUDA only -- on CPU this option has no effect yet.
    #[builder(default = false)]
    pub quantize_activations: bool,
}
