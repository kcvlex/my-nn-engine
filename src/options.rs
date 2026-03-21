use typed_builder::TypedBuilder;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    CPU,
    CUDA,
}

#[derive(Clone, Debug, TypedBuilder)]
pub struct Options {
    #[builder(default = 100000)]
    pub omp_elementwise_threshold: usize,

    #[builder(default = 10000)]
    pub omp_attention_threshold: usize,

    #[builder(default = true)]
    pub enable_fuse_ops: bool,

    #[builder(default = true)]
    pub verify_after_inference: bool,

    #[builder(default = true)]
    pub verify_after_strides: bool,

    #[builder(default = Target::CPU)]
    pub target: Target,

    #[builder(default = 16)]
    pub num_cuda_streams: usize,

    #[builder(default = false)]
    pub profile: bool,

    #[builder(default = true)]
    pub save_build_dir: bool,

    #[builder(default = true)]
    pub save_transformed_model: bool,
}
