use typed_builder::TypedBuilder;

use crate::schedule::scheduler::PlacementStrategy;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    CPU,
    CUDA,
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

    #[builder(default)]
    pub placement_strategy: Option<PlacementStrategy>,

    /// If true, rewrite every `DequantMatMul` so its activation input is first
    /// passed through a `DynamicQuantizeLinear` (symmetric per-row int8) and
    /// then consumed by a `QuantizedMatMul` instead. Requires K % 32 == 0
    /// (CUDA INT8 mma kernel constraint); nodes that don't qualify are left
    /// alone. CUDA only -- on CPU this option has no effect yet.
    #[builder(default = false)]
    pub quantize_activations: bool,
}
