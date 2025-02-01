use crate::onnx::model::{Graph, NodeId, ValueId};
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::DataType;
use itertools::{izip, zip_eq};
use std::ops::Index;
//use strum_macros::EnumString;

#[derive(Debug, Clone, Copy)]
pub struct TensorIndex(isize);

impl TensorIndex {
    pub fn new(i: isize) -> Self {
        Self(i)
    }

    pub fn index(&self, rank: usize) -> usize {
        if self.0 < 0 {
            (rank as isize + self.0) as usize
        } else {
            self.0 as usize
        }
    }

    pub fn raw(&self) -> isize {
        self.0
    }
}

#[derive(Debug, Clone)]
pub enum Operator {
    Add,
    BatchNormalization(BatchNormalization),
    Cast(Cast),
    Concat(Concat),
    Conv(Conv),
    Exp,
    Gather(Gather),
    Gemm(Gemm),
    GlobalAveragePool,
    Identity,
    LeakyReLU(LeakyReLU),
    Log,
    ReLU,
    Reshape,
    Resize(Resize),
    MatMul,
    MaxPool(Pooling),
    Mul,
    ReduceMax(Reduce),
    ReduceMean(Reduce),
    ReduceSum(Reduce),
    Shape(Shape),
    Slice,
    Split(Split),
    Sigmoid,
    Tanh,
    Transpose(Transpose),

    // Custom
    BatchNormalizationPerChannel(BatchNormalization),
    BLASGemm(BLASGemm),
    Contiguous,
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
pub struct Cast {
    pub to: DataType,
}

#[derive(Debug, Clone)]
pub struct Concat {
    pub axis: TensorIndex,
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

#[derive(Debug, Clone)]
pub struct Gather {
    pub axis: TensorIndex,
}

#[derive(Debug, Clone)]
pub struct LeakyReLU {
    pub alpha: f64,
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

#[derive(Debug, Clone, Copy)]
pub enum ResizeNearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

#[derive(Debug, Clone, Copy)]
pub enum ResizeCoordinateTransformationMode {
    HalfPixel,
}

#[derive(Debug, Clone, Copy)]
pub enum ResizeKeepAspectRatioPolicy {
    Stretch,
    NotLarger,
    NotSmaller,
}

#[derive(Debug, Clone, Copy)]
pub enum ResizeMode {
    Nearest(ResizeNearestMode),
}

#[derive(Debug, Clone)]
pub struct Resize {
    pub axes: Option<Vec<TensorIndex>>,
    pub coordinate_transformation_mode: ResizeCoordinateTransformationMode,
    pub keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
    pub mode: ResizeMode,
}

#[derive(Debug, Clone)]
pub struct Shape {
    pub start: TensorIndex,
    pub end: Option<TensorIndex>,
}

#[derive(Debug, Clone)]
pub struct Slice {
    pub start: TensorIndex,
    pub end: TensorIndex,
    pub axis: TensorIndex,
    pub step: i64,
}

#[derive(Debug, Clone)]
pub struct Transpose {
    pub perm: Option<Vec<usize>>,
}

impl Slice {
    pub fn collect_slices(graph: &Graph, node_id: NodeId) -> Option<Vec<Self>> {
        let node = &graph.nodes[node_id];
        let starts = graph
            .initializer
            .get(&node.inputs[args::SLICE_STARTS])?
            .to_indices()?;
        let ends = graph
            .initializer
            .get(&node.inputs[args::SLICE_ENDS])?
            .to_indices()?;
        let axes = node
            .inputs
            .get(args::SLICE_AXES)
            .and_then(|x| graph.initializer.get(x))
            .and_then(|x| x.to_indices())
            .unwrap_or(
                (0..(starts.len() as isize))
                    .map(TensorIndex::new)
                    .collect::<Vec<_>>(),
            );
        let steps = if node.inputs.get(args::SLICE_STEPS).is_some() {
            unimplemented!()
        } else {
            vec![1; starts.len()]
        };

        // TODO: Check length of each vector
        Some(
            izip!(
                starts.into_iter(),
                ends.into_iter(),
                axes.into_iter(),
                steps.into_iter()
            )
            .map(|(start, end, axis, step)| Slice {
                start,
                end,
                axis,
                step,
            })
            .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone)]
pub enum SplitOutputs {
    NumOutputs(usize),
    Split(Vec<usize>),
}

#[derive(Debug, Clone)]
pub struct Split {
    pub axis: TensorIndex,
    pub outputs: SplitOutputs,
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
}

impl Default for Gemm {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BLASGemm {
    pub alpha: f64,
    pub beta: f64,
    pub trans_a: bool,
    pub trans_b: bool,
    pub trans_c: bool,
}

impl Default for BLASGemm {
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

impl TryInto<BLASGemm> for &Gemm {
    type Error = String;

    fn try_into(self) -> Result<BLASGemm, Self::Error> {
        if self.alpha != self.beta {
            return Err("alpha and beta must be the same".to_string());
        }
        Ok(BLASGemm {
            alpha: self.alpha,
            beta: 0.0,
            trans_a: self.trans_a,
            trans_b: self.trans_b,
            trans_c: false,
        })
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

impl Resize {
    pub fn resized_shape(&self, graph: &Graph, node_id: NodeId) -> Option<ResolvedTensorDims> {
        let node = &graph.nodes[node_id];
        assert!(matches!(node.op, Operator::Resize(_)));

        let mut dims = graph.get_resolved_tensor_type(node.inputs[0])?.dims.clone();
        if let Some(roi) = &node
            .inputs
            .get(args::RESIZE_ROI)
            .and_then(|x| graph.initializer.get(x))
        {
            if !roi.dims.is_scalar() {
                unimplemented!()
            }
        }

        let axes: Vec<_> = match self.axes {
            Some(ref axes) => axes.iter().map(|x| x.index(dims.ndim())).collect(),
            None => (0..dims.ndim()).collect(),
        };

        let scales = &node
            .inputs
            .get(args::RESIZE_SCALES)
            .and_then(|x| graph.initializer.get(x));
        let sizes = &node
            .inputs
            .get(args::RESIZE_SIZES)
            .and_then(|x| graph.initializer.get(x));
        dbg!(scales, sizes);
        let scales = match (scales, sizes) {
            (Some(scales), None) => scales.to_1d_floats(),
            (Some(scales), Some(sizes)) if scales.dims.is_scalar() => {
                let scales = zip_eq(axes.iter(), sizes.to_indices()?.iter())
                    .map(|(dim, size)| {
                        let old = dims[*dim] as f64;
                        let new = size.raw() as f64;
                        new / old
                    })
                    .collect();
                Some(scales)
            }

            // Invalid inputs
            (Some(_), Some(_)) | (None, None) => None,

            (None, Some(_)) => unreachable!(),
        }?;

        if !matches!(
            self.keep_aspect_ratio_policy,
            ResizeKeepAspectRatioPolicy::Stretch
        ) {
            unimplemented!()
        }

        for (dim, scale) in zip_eq(axes.iter(), scales.iter()) {
            dims[*dim] = (dims[*dim] as f64 * scale).round() as usize;
        }

        Some(dims)
    }
}

impl Split {
    pub fn split(&self, dims: &ResolvedTensorDims) -> Option<Vec<usize>> {
        let axis = self.axis.index(dims.ndim());
        match self.outputs {
            SplitOutputs::NumOutputs(num_outputs) => {
                let mut res = Vec::new();
                let dim = dims[axis];
                let mut cur = 0;
                let step = dim / num_outputs;
                let bound = dim;
                while cur < bound {
                    let start = cur;
                    let end = (cur + step).min(bound);
                    res.push(end - start);
                    cur = end;
                }
                Some(res)
            }
            SplitOutputs::Split(ref sizes) => {
                let sum = sizes.iter().sum::<usize>();
                if sum != dims[axis] {
                    return None;
                }
                Some(sizes.clone())
            }
        }
    }
}

impl Transpose {
    pub fn perm(&self, rank: usize) -> Option<Vec<usize>> {
        let res = self
            .perm
            .clone()
            .unwrap_or((0..rank).rev().collect::<Vec<_>>());
        let mut flags = vec![false; res.len()];
        for p in res.iter() {
            if res.len() <= *p || flags[*p] {
                return None;
            }
            flags[*p] = true;
        }
        Some(res)
    }
}

impl Operator {
    pub fn name(&self) -> &str {
        match self {
            Operator::Add => "Add",
            Operator::BatchNormalization(_) => "BatchNormalization",
            Operator::Cast(_) => "Cast",
            Operator::Concat(_) => "Concat",
            Operator::Conv(_) => "Conv",
            Operator::Exp => "Exp",
            Operator::Gather(_) => "Gather",
            Operator::Gemm(_) => "Gemm",
            Operator::GlobalAveragePool => "GlobalAveragePool",
            Operator::Identity => "Identity",
            Operator::LeakyReLU(_) => "LeakyReLU",
            Operator::Log => "Log",
            Operator::ReLU => "ReLU",
            Operator::Reshape => "Reshape",
            Operator::MatMul => "MatMul",
            Operator::MaxPool(_) => "MaxPool",
            Operator::Mul => "Mul",
            Operator::ReduceMax(_) => "ReduceMax",
            Operator::ReduceMean(_) => "ReduceMean",
            Operator::ReduceSum(_) => "ReduceSum",
            Operator::Resize(_) => "Resize",
            Operator::Shape(_) => "Shape",
            Operator::Slice => "Slice",
            Operator::Split(_) => "Split",
            Operator::Sigmoid => "Sigmoid",
            Operator::Tanh => "Tanh",
            Operator::Transpose(_) => "Transpose",

            // Custom
            Operator::BatchNormalizationPerChannel(_) => "BatchNormalizationPerChannel (Custom)",
            Operator::BLASGemm(_) => "BLASGemm (Custom)",
            Operator::Contiguous => "Contiguous (Custom)",
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

    pub const RESIZE_ROI: usize = 1;
    pub const RESIZE_SCALES: usize = 2;
    pub const RESIZE_SIZES: usize = 3;

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
    pub const GEMM_C: usize = 2;

    pub const BATCHNORM_DATA: usize = 0;
    pub const BATCHNORM_SCALE: usize = 1;
    pub const BATCHNORM_BIAS: usize = 2;
    pub const BATCHNORM_MEAN: usize = 3;
    pub const BATCHNORM_VAR: usize = 4;

    pub const SLICE_STARTS: usize = 1;
    pub const SLICE_ENDS: usize = 2;
    pub const SLICE_AXES: usize = 3;
    pub const SLICE_STEPS: usize = 4;
}

//#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
//#[derive(Debug, Clone, EnumString, PartialEq)]
//#[test]
//fn test_enum_string() {
//    assert_eq!(AutoPad::try_from("NOT_SET"), Ok(AutoPad::NotSet));
//}
