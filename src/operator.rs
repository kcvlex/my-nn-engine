use crate::model::ValueId;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use std::ops::Index;
//use strum_macros::EnumString;

#[derive(Debug, Clone)]
pub enum Operator {
    Add,
    BatchNormalization(BatchNormalization),
    Conv(Conv),
    Gemm(Gemm),
    GlobalAveragePool,
    Identity,
    ReLU,
    Reshape,
    MatMul,
    MaxPool(Pooling),
    ReduceMax(Reduce),
    ReduceMean(Reduce),
    ReduceSum(Reduce),
    Sigmoid,
    Transpose(Vec<usize>),

    // Custom
    BatchNormalizationPerChannel(BatchNormalization),
    Im2Col(Im2Col),
    ReduceMatrix(ReduceOp),

    // For debug
    ForceReshape,

    // Dummy
    Input(ValueId),
    Output(ValueId),
}

#[derive(Debug, Clone)]
pub struct BatchNormalization {
    pub epsilon: f32,
    pub momentum: f32,
}

#[derive(Debug, Clone)]
pub struct OptionalVec<T: Copy + Clone> {
    vec: Option<Vec<T>>,
    default: T,
}

impl<T: Clone + Copy> OptionalVec<T> {
    pub fn new(vec: Option<Vec<T>>, default: T) -> Self {
        Self { vec, default }
    }
}

impl<T: Clone + Copy> Index<usize> for OptionalVec<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        self.vec.as_ref().map_or(&self.default, |v| &v[i])
    }
}

#[derive(Debug, Clone)]
pub enum ConvPad {
    NotSet(OptionalVec<(usize, usize)>),
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Debug, Clone)]
pub struct Conv {
    pub pad: ConvPad,
    pub dilations: OptionalVec<usize>,
    pub groups: usize,
    pub kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

// TODO: storage_order
#[derive(Debug, Clone)]
pub struct Pooling {
    pub pad: ConvPad,
    pub ceil_mode: bool,
    pub dilations: OptionalVec<usize>,
    pub kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

#[derive(Debug, Clone)]
pub struct Reduce {
    pub axes: Vec<i64>,
    pub keepdims: bool,
}

impl Reduce {
    pub fn normalize_axes(&self, rank: usize) -> Option<Vec<usize>> {
        let res: Vec<_> = self
            .axes
            .iter()
            .map(|&a| if a < 0 { rank as i64 + a } else { a } as usize)
            .collect();
        for i in res.iter() {
            if rank <= *i {
                return None;
            }
        }
        Some(res)
    }
}

#[derive(Debug, Clone)]
pub struct Gemm {
    pub alpha: f64,
    pub beta: f64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub trans_c: bool,
}

impl Default for Gemm {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
            trans_c: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Channel {
    Meld(usize),  // Conv
    Split(usize), // MaxPool
}

impl Channel {
    pub fn inner(&self) -> usize {
        match self {
            Channel::Meld(v) => *v,
            Channel::Split(v) => *v,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Im2Col {
    pub nbatch: usize,
    pub one_fm_shape: ResolvedTensorDims, // convolution of one image and one kernel (feature map)
    pub pad: ConvPad,
    pub channel: Channel,
    pub dilations: OptionalVec<usize>,
    pub one_kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

impl Im2Col {
    pub fn padded_len(&self, dim: usize) -> usize {
        let unit = self.dilations[dim] * (self.one_kernel_shape[dim] - 1) + 1;
        self.strides[dim] * (self.one_fm_shape[dim] - 1) + unit
    }
}

impl Operator {
    pub fn name(&self) -> &str {
        match self {
            Operator::Add => "Add",
            Operator::BatchNormalization(_) => "BatchNormalization",
            Operator::Conv(_) => "Conv",
            Operator::Gemm(_) => "Gemm",
            Operator::GlobalAveragePool => "GlobalAveragePool",
            Operator::Identity => "Identity",
            Operator::ReLU => "ReLU",
            Operator::Reshape => "Reshape",
            Operator::MatMul => "MatMul",
            Operator::MaxPool(_) => "MaxPool",
            Operator::ReduceMax(_) => "ReduceMax",
            Operator::ReduceMean(_) => "ReduceMean",
            Operator::ReduceSum(_) => "ReduceSum",
            Operator::Sigmoid => "Sigmoid",
            Operator::Transpose(_) => "Transpose",

            // Custom
            // Operator::MatMulRightTransposed => "MatMulRightTransposed (Custom)",
            Operator::BatchNormalizationPerChannel(_) => "BatchNormalizationPerChannel (Custom)",
            Operator::Im2Col(_) => "Im2Col (Custom)",
            Operator::ReduceMatrix(_) => "ReduceMatrix (Custom)",

            Operator::ForceReshape => "ForceReshape (For Debug)",

            // Dummy
            Operator::Input(_) => "Input",
            Operator::Output(_) => "Output",
        }
    }

    pub fn is_elementwise(&self) -> bool {
        matches!(self, Operator::Add | Operator::ReLU | Operator::Sigmoid)
    }

    pub fn is_identity(&self) -> bool {
        matches!(self, Operator::Identity)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Max,
    Mean,
    Variance,
    Sum,
}

pub mod args {
    pub const ADD_LHS: usize = 0;
    pub const ADD_RHS: usize = 1;

    pub const RESHAPE_DATA: usize = 0;
    pub const RESHAPE_SHAPE: usize = 1;

    pub const CONV_DATA: usize = 0;
    pub const CONV_WEIGHT: usize = 1;
    pub const CONV_BIAS: usize = 2;

    pub const RELU_DATA: usize = 0;

    pub const MATMUL_LHS: usize = 0;
    pub const MATMUL_RHS: usize = 1;

    pub const MAXPOOL_DATA: usize = 0;

    pub const TRANSPOSE_DATA: usize = 0;

    pub const GEMM_A: usize = 0;
    pub const GEMM_B: usize = 1;

    pub const BATCHNORM_DATA: usize = 0;
    pub const BATCHNORM_SCALE: usize = 1;
    pub const BATCHNORM_BIAS: usize = 2;
    pub const BATCHNORM_MEAN: usize = 3;
    pub const BATCHNORM_VAR: usize = 4;
}

//#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
//#[derive(Debug, Clone, EnumString, PartialEq)]
//#[test]
//fn test_enum_string() {
//    assert_eq!(AutoPad::try_from("NOT_SET"), Ok(AutoPad::NotSet));
//}
