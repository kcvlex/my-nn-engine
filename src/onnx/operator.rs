use std::ops::Index;

use itertools::izip;
use itertools::zip_eq;

use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::model::ValueId;
use crate::tensor::data::ScalarData;
use crate::tensor::types::DataType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::Tensor;
//use strum_macros::EnumString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Add,
    Attention(Attention),
    BatchedGemm(BatchedGemm),
    BatchNormalization(BatchNormalization),
    Cast(Cast),
    Concat(Concat),
    Constant(Constant),
    ConstantOfShape(ConstantOfShape),
    Conv(Conv),
    Div,
    Exp,
    Gather(Gather),
    GeLU(GeLU),
    Gemm(Gemm),
    GlobalAveragePool,
    Identity,
    LayerNormalization(LayerNormalization),
    LeakyReLU(LeakyReLU),
    Log,
    MatMul,
    MaxPool(Pooling),
    Mul,
    NonZero,
    OneHot(OneHot),
    Pow,
    Reciprocal,
    ReduceMax(Reduce),
    ReduceMean(Reduce),
    ReduceSum(Reduce),
    ReLU,
    Reshape,
    Resize(Resize),
    Shape(Shape),
    Sigmoid,
    Slice,
    Softmax(Softmax),
    Split(Split),
    Sqrt,
    Squeeze(Squeeze),
    Sub,
    Tanh,
    Transpose(Transpose),
    Unsqueeze(Unsqueeze),

    // Custom
    Contiguous(Contiguous),
    Im2Col(Im2Col),
    NHWC2NCHW,
    ReduceMatrix(ReduceOp),
    Reinterpret(Reinterpret),

    // Dummy
    Input(ValueId),
    Output(ValueId),
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Attention {
    pub is_causal: bool,
    pub scale: f32,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct BatchNormalization {
    pub epsilon: f32,
    pub momentum: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OptionalVec<T: Copy + Clone + PartialEq> {
    vec: Option<Vec<T>>,
    default: T,
}

impl<T: Clone + Copy + PartialEq> OptionalVec<T> {
    pub fn new(vec: Option<Vec<T>>, default: T) -> Self {
        Self { vec, default }
    }

    pub fn inner(&self) -> Option<&Vec<T>> {
        self.vec.as_ref()
    }
}

impl<T: Clone + Copy + PartialEq> Index<usize> for OptionalVec<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        self.vec.as_ref().map_or(&self.default, |v| &v[i])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cast {
    pub to: DataType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Concat {
    pub axis: TensorIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constant {
    pub value: Tensor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConstantOfShape {
    pub value: ScalarData,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConvPad {
    NotSet(OptionalVec<(usize, usize)>),
    SameUpper,
    SameLower,
    Valid,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Conv {
    pub pad: ConvPad,
    pub dilations: OptionalVec<usize>,
    pub groups: usize,
    pub kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

#[derive(Debug)]
struct ConvShape<'a> {
    kernel_shape: &'a [usize],
    input_shape: &'a [usize],
    pad: &'a OptionalVec<(usize, usize)>,
    dilations: &'a OptionalVec<usize>,
}

impl ConvShape<'_> {
    fn padded_input_size(&self, i: usize) -> usize {
        self.input_shape[i] + self.pad[i].0 + self.pad[i].1
    }

    // Number of elements between first and last element (inclusive)
    fn distance_per_conv(&self, i: usize) -> usize {
        self.dilations[i] * (self.kernel_shape[i] - 1) + 1
    }
}

impl Conv {
    pub fn output_shape(
        &self,
        input_shape: &ResolvedTensorDims,
        weight_shape: &ResolvedTensorDims,
    ) -> ResolvedTensorDims {
        // TODO: Check bias
        // TODO: Check kernel_shape

        assert!(input_shape.ndim() == weight_shape.ndim());
        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let ndim = input_shape.ndim() - 2;
        let input = &input_shape[2..];

        let feature_map_size = weight_shape[0];

        assert!(feature_map_size.is_multiple_of(self.groups));
        assert!(channels == weight_shape[1] * self.groups);

        let kernel_shape = &weight_shape[2..];
        let default_pad = OptionalVec::new(None, (0, 0));
        let pad = match self.pad {
            ConvPad::NotSet(ref pad) => Some(pad),
            ConvPad::SameUpper | ConvPad::SameLower => None,
            ConvPad::Valid => Some(&default_pad),
        };

        let mut dims = Vec::with_capacity(ndim + 2);
        dims.push(batch_size);
        dims.push(feature_map_size);
        let conv_shape = pad.map(|pad| ConvShape {
            kernel_shape,
            input_shape: input,
            pad,
            dilations: &self.dilations,
        });
        for i in 0..ndim {
            let stride = self.strides[i];
            let dim = if let Some(ref conv_shape) = &conv_shape {
                let (q, _) = num_integer::div_rem(
                    conv_shape.padded_input_size(i) - conv_shape.distance_per_conv(i),
                    stride,
                );
                // assert!(rem == 0);
                q + 1
            } else {
                input[i].div_ceil(stride)
            };
            dims.push(dim);
        }
        ResolvedTensorDims::new(&dims)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gather {
    pub axis: TensorIndex,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct GeLU {
    pub approximate: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LayerNormalization {
    pub axis: TensorIndex,
    pub epsilon: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeakyReLU {
    pub alpha: f64,
}

// TODO: storage_order
#[derive(Debug, Clone, PartialEq)]
pub struct Pooling {
    pub pad: ConvPad,
    pub ceil_mode: bool,
    pub dilations: OptionalVec<usize>,
    pub kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
}

impl Pooling {
    pub fn output_shape(&self, input_shape: &ResolvedTensorDims) -> ResolvedTensorDims {
        let mut dims = Vec::with_capacity(input_shape.ndim());
        dims.push(input_shape[0]);
        dims.push(input_shape[1]);
        let input = &input_shape[2..];
        let default_pad = OptionalVec::new(None, (0, 0));
        let pad = match self.pad {
            ConvPad::NotSet(ref pad) => pad,
            ConvPad::Valid => &default_pad,
            // deprecated attributes
            ConvPad::SameUpper | ConvPad::SameLower => unimplemented!(),
        };
        let conv_shape = ConvShape {
            kernel_shape: &self.kernel_shape[..],
            input_shape: input,
            pad,
            dilations: &self.dilations,
        };
        for i in 0..input.len() {
            let stride = self.strides[i];
            let num = conv_shape.padded_input_size(i) - conv_shape.distance_per_conv(i);
            let dim = if !self.ceil_mode {
                // Floor div
                num / stride + 1
            } else {
                // TODO: Correct?
                //
                // https://onnx.ai/onnx/operators/onnx__MaxPool.html#summary
                //  > Sliding windows that would start in the right padded region are ignored.
                num.div_ceil(stride) + 1
            };
            dims.push(dim);
        }
        ResolvedTensorDims::new(&dims)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct OneHot {
    pub axis: isize,

    pub depth: Option<usize>,
    pub off_value: Option<ScalarData>,
    pub on_value: Option<ScalarData>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reduce {
    pub axes: Vec<i64>,
    pub keepdims: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeNearestMode {
    RoundPreferFloor,
    RoundPreferCeil,
    Floor,
    Ceil,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeCoordinateTransformationMode {
    HalfPixel,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeKeepAspectRatioPolicy {
    Stretch,
    NotLarger,
    NotSmaller,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResizeMode {
    Nearest(ResizeNearestMode),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResizeScale {
    Scales(Vec<f64>),
    Sizes(Vec<i64>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Resize {
    pub axes: Option<Vec<TensorIndex>>,
    pub coordinate_transformation_mode: ResizeCoordinateTransformationMode,
    pub keep_aspect_ratio_policy: ResizeKeepAspectRatioPolicy,
    pub mode: ResizeMode,

    pub scale: Option<ResizeScale>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ReinterpretType {
    Reshape {
        before: Vec<usize>,
        after: Vec<usize>,
    },
    Transpose(Transpose),
    Broadcast {
        before: Vec<usize>,
        after: Vec<usize>,
    },
}

impl ReinterpretType {
    pub fn single_reshape(before: Vec<usize>, after: Vec<usize>) -> Self {
        Self::Reshape { before, after }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reinterpret {
    pub ops: Vec<ReinterpretType>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shape {
    pub start: TensorIndex,
    pub end: Option<TensorIndex>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Slice {
    pub start: isize,
    pub end: isize,
    pub axis: usize,
    pub step: isize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transpose {
    pub perm: Option<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Unsqueeze {
    pub axes: Vec<TensorIndex>,
}

impl Slice {
    pub fn collect_slices(graph: &Graph, node_id: NodeId) -> Option<Vec<Self>> {
        let node = &graph.nodes[node_id];
        let dims = &graph
            .get_resolved_tensor_type(node.inputs[args::SLICE_DATA])?
            .dims;
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

        if node.inputs.get(args::SLICE_STEPS).is_some() {
            let is_all_one = graph
                .initializer
                .get(&node.inputs[args::SLICE_STEPS])
                .and_then(|x| x.to_1d_sints())
                .unwrap()
                .iter()
                .all(|&x| x == 1);
            if !is_all_one {
                unimplemented!("Non-unit step in Slice")
            }
        }
        let steps = vec![1; starts.len()];

        // TODO: Check length of each vector
        Some(
            izip!(
                starts.into_iter(),
                ends.into_iter(),
                axes.into_iter(),
                steps.into_iter()
            )
            .map(|(start, end, axis, step)| {
                if step < 0 {
                    unimplemented!("Negative step in Slice")
                }

                let axis = axis.index(dims.ndim());
                let start = start.index(dims[axis]) as isize;
                let end = end.index(dims[axis]) as isize;
                let start = start.clamp(0, dims[axis] as isize);
                let end = end.clamp(0, dims[axis] as isize);
                Slice {
                    start,
                    end,
                    axis,
                    step,
                }
            })
            .collect::<Vec<_>>(),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Softmax {
    pub axis: TensorIndex,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SplitOutputs {
    NumOutputs(usize),
    Split(Vec<usize>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Split {
    pub axis: TensorIndex,
    pub outputs: Option<SplitOutputs>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Squeeze {
    pub axes: Option<Vec<TensorIndex>>,
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

#[derive(Debug, Clone, PartialEq)]
pub struct Gemm {
    pub alpha: f64,
    pub beta: f64,
    pub trans_a: bool,
    pub trans_b: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchedGemm {
    pub alpha: f64,
    pub beta: f64,
    pub trans_a: bool,
    pub trans_b: bool,
}

impl Default for Gemm {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 0.0,
            trans_a: false,
            trans_b: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Contiguous {
    pub ops: Vec<ReinterpretType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Channel {
    Meld(usize),  // Conv
    Split(usize), // MaxPool
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PadVal {
    Zero,
    NInf,
}

impl Channel {
    pub fn inner(&self) -> usize {
        match self {
            Channel::Meld(v) => *v,
            Channel::Split(v) => *v,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Im2Col {
    pub nbatch: usize,
    pub one_fm_shape: ResolvedTensorDims, // convolution of one image and one kernel (feature map)
    pub pad: ConvPad,
    pub channel: Channel,
    pub dilations: OptionalVec<usize>,
    pub one_kernel_shape: ResolvedTensorDims,
    pub strides: OptionalVec<usize>,
    pub pad_val: PadVal,
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
        let Operator::Resize(resize) = &node.op else {
            unreachable!()
        };

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

        let scale = resize.scale.as_ref()?;
        let scale = match scale {
            ResizeScale::Scales(scales) => scales.clone(),
            ResizeScale::Sizes(sizes) => {
                let res = zip_eq(axes.iter(), sizes.iter())
                    .map(|(dim, size)| {
                        let old = dims[*dim] as f64;
                        let new = *size as f64;
                        new / old
                    })
                    .collect::<Vec<_>>();
                res
            }
        };

        if !matches!(
            self.keep_aspect_ratio_policy,
            ResizeKeepAspectRatioPolicy::Stretch
        ) {
            unimplemented!()
        }

        for (dim, scale) in zip_eq(axes.iter(), scale.iter()) {
            dims[*dim] = (dims[*dim] as f64 * scale).round() as usize;
        }

        Some(dims)
    }
}

impl Split {
    pub fn split(&self, dims: &ResolvedTensorDims) -> Option<Vec<usize>> {
        let axis = self.axis.index(dims.ndim());
        match &self.outputs {
            Some(SplitOutputs::NumOutputs(num_outputs)) => {
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
            Some(SplitOutputs::Split(ref sizes)) => {
                let sum = sizes.iter().sum::<usize>();
                if sum != dims[axis] {
                    return None;
                }
                Some(sizes.clone())
            }
            None => {
                let dim = dims[axis];
                let first = dim / 2;
                let second = dim - first;
                Some(vec![first, second])
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorType {
    Elementwise,
    Bijective,
    Opaque,
    Dummy,
}

impl Operator {
    pub fn name(&self) -> &str {
        match self {
            Operator::Add => "Add",
            Operator::Attention(_) => "Attention",
            Operator::BatchedGemm(_) => "BatchedGemm",
            Operator::BatchNormalization(_) => "BatchNormalization",
            Operator::Cast(_) => "Cast",
            Operator::Concat(_) => "Concat",
            Operator::Constant(_) => "Constant",
            Operator::ConstantOfShape(_) => "ConstantOfShape",
            Operator::Conv(_) => "Conv",
            Operator::Div => "Div",
            Operator::Exp => "Exp",
            Operator::Gather(_) => "Gather",
            Operator::GeLU(_) => "Gelu",
            Operator::Gemm(_) => "Gemm",
            Operator::GlobalAveragePool => "GlobalAveragePool",
            Operator::Identity => "Identity",
            Operator::LayerNormalization(_) => "LayerNormalization",
            Operator::LeakyReLU(_) => "LeakyRelu",
            Operator::Log => "Log",
            Operator::MatMul => "MatMul",
            Operator::MaxPool(_) => "MaxPool",
            Operator::Mul => "Mul",
            Operator::NonZero => "NonZero",
            Operator::OneHot(_) => "OneHot",
            Operator::Pow => "Pow",
            Operator::Reciprocal => "Reciprocal",
            Operator::ReduceMax(_) => "ReduceMax",
            Operator::ReduceMean(_) => "ReduceMean",
            Operator::ReduceSum(_) => "ReduceSum",
            Operator::ReLU => "Relu",
            Operator::Reshape => "Reshape",
            Operator::Resize(_) => "Resize",
            Operator::Shape(_) => "Shape",
            Operator::Sigmoid => "Sigmoid",
            Operator::Slice => "Slice",
            Operator::Softmax(_) => "Softmax",
            Operator::Split(_) => "Split",
            Operator::Sqrt => "Sqrt",
            Operator::Squeeze(_) => "Squeeze",
            Operator::Sub => "Sub",
            Operator::Tanh => "Tanh",
            Operator::Transpose(_) => "Transpose",
            Operator::Unsqueeze(_) => "Unsqueeze",

            // Custom
            Operator::Contiguous(_) => "Contiguous",
            Operator::Im2Col(_) => "Im2Col",
            Operator::NHWC2NCHW => "NHWC2NCHW",
            Operator::ReduceMatrix(_) => "ReduceMatrix",
            Operator::Reinterpret(_) => "Reinterpret",

            // Dummy
            Operator::Input(_) => "Input",
            Operator::Output(_) => "Output",
        }
    }

    pub fn operator_type(&self) -> OperatorType {
        match self {
            Operator::Add |
            Operator::BatchNormalization(_) |
            Operator::Cast(_) |
            Operator::Div |
            Operator::Exp |
            Operator::GeLU(_) |
            Operator::Identity |
            Operator::LeakyReLU(_) |
            Operator::Log |
            Operator::Mul |
            Operator::Pow |
            Operator::Reciprocal |
            Operator::ReLU |
            Operator::Sigmoid |
            Operator::Sqrt |
            Operator::Sub |
            Operator::Tanh => OperatorType::Elementwise,

            Operator::Contiguous(_) |
            Operator::NHWC2NCHW |
            Operator::Reshape |
            Operator::Squeeze(_) |
            Operator::Transpose(_) |
            Operator::Unsqueeze(_) => OperatorType::Bijective,

            Operator::Attention(_) |
            Operator::BatchedGemm(_) |
            Operator::Concat(_) |
            Operator::Constant(_) |
            Operator::ConstantOfShape(_) |
            Operator::Conv(_) |
            Operator::Gather(_) |
            Operator::Gemm(_) |
            Operator::GlobalAveragePool |
            Operator::Im2Col(_) |
            Operator::LayerNormalization(_) |
            Operator::MatMul |
            Operator::MaxPool(_) |
            Operator::NonZero |
            Operator::OneHot(_) |
            Operator::ReduceMax(_) |
            Operator::ReduceMatrix(_) |
            Operator::ReduceMean(_) |
            Operator::ReduceSum(_) |
            Operator::Reinterpret(_) |
            Operator::Resize(_) |
            Operator::Shape(_) |
            Operator::Slice |
            Operator::Softmax(_) |
            Operator::Split(_) => OperatorType::Opaque,

            Operator::Input(_) | Operator::Output(_) => OperatorType::Dummy,
        }
    }

    pub fn is_elementwise(&self) -> bool {
        self.operator_type() == OperatorType::Elementwise
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOp {
    Max,
    Mean,
    Variance,
    Sum,
}

pub mod args {
    pub const ADD_LHS: usize = 0;
    pub const ADD_RHS: usize = 1;

    pub const ATTENTION_Q: usize = 0;
    pub const ATTENTION_K: usize = 1;
    pub const ATTENTION_V: usize = 2;
    pub const ATTENTION_MASK: usize = 3;

    pub const RESHAPE_DATA: usize = 0;
    pub const RESHAPE_SHAPE: usize = 1;

    pub const RESIZE_ROI: usize = 1;
    pub const RESIZE_SCALES: usize = 2;
    pub const RESIZE_SIZES: usize = 3;

    pub const CONV_DATA: usize = 0;
    pub const CONV_WEIGHT: usize = 1;
    pub const CONV_BIAS: usize = 2;

    pub const LAYER_NORM_DATA: usize = 0;
    pub const LAYER_NORM_SCALE: usize = 1;
    pub const LAYER_NORM_BIAS: usize = 2;

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

    pub const SLICE_DATA: usize = 0;
    pub const SLICE_STARTS: usize = 1;
    pub const SLICE_ENDS: usize = 2;
    pub const SLICE_AXES: usize = 3;
    pub const SLICE_STEPS: usize = 4;

    pub const ONEHOT_INDICES: usize = 0;
    pub const ONEHOT_DEPTH: usize = 1;
    pub const ONEHOT_VALUES: usize = 2;
}

//#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
//#[derive(Debug, Clone, EnumString, PartialEq)]
//#[test]
//fn test_enum_string() {
//    assert_eq!(AutoPad::try_from("NOT_SET"), Ok(AutoPad::NotSet));
//}
