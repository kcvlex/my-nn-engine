use crate::model::ValueId;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use std::ops::Index;
//use strum_macros::EnumString;

#[derive(Debug, Clone)]
pub enum Operator {
    Add,
    Conv(Conv),
    ReLU,
    MatMul,
    MaxPool(MaxPool),
    Reshape,
    Transpose(Vec<usize>),

    // Custom
    MatMulRightTransposed,

    // Dummy
    Input(ValueId),
    Output(ValueId),
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
pub struct MaxPool {
    pub pad: ConvPad,
    pub ceil_mode: bool,
    pub dilations: OptionalVec<usize>,
    pub kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

#[derive(Debug, Clone, Copy)]
pub enum Channel {
    Meld(usize),  // Conv
    Split(usize), // MaxPool
}

#[derive(Debug, Clone)]
pub struct Im2Col {
    pub result_shape: ResolvedTensorDims, // convolution of one image and one kernel
    pub pad: ConvPad,
    pub channel: Channel,
    pub dilations: OptionalVec<usize>,
    pub kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

impl Operator {
    pub fn name(&self) -> &str {
        match self {
            Operator::Add => "Add",
            Operator::Conv(_) => "Conv",
            Operator::ReLU => "ReLU",
            Operator::MatMul => "MatMul",
            Operator::MaxPool(_) => "MaxPool",
            Operator::Reshape => "Reshape",
            Operator::Transpose(_) => "Transpose",

            // Custom
            Operator::MatMulRightTransposed => "MatMulRightTransposed (Custom)",

            // Dummy
            Operator::Input(_) => "Input",
            Operator::Output(_) => "Output",
        }
    }
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
}

//#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
//#[derive(Debug, Clone, EnumString, PartialEq)]
//#[test]
//fn test_enum_string() {
//    assert_eq!(AutoPad::try_from("NOT_SET"), Ok(AutoPad::NotSet));
//}
