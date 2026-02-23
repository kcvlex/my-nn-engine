use typed_builder::TypedBuilder;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Target {
    CPU,
    CUDA,
}

#[derive(Clone, Debug, TypedBuilder)]
pub struct Options {
    #[builder(default = 100)]
    pub omp_threshold: usize,

    #[builder(default = true)]
    pub enable_fuse_ops: bool,

    #[builder(default = true)]
    pub verify_after_inference: bool,

    #[builder(default = true)]
    pub verify_after_strides: bool,

    #[builder(default = Target::CPU)]
    pub target: Target,

    #[builder(default = false)]
    pub profile: bool,

    #[builder(default = true)]
    pub save_build_dir: bool,
}
