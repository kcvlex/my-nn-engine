use crate::codegen::cuda::*;
use crate::{
    schedule::*,
    tensor::types::{DataType, FloatType, SIntType, UIntType},
};
use derive_more::From;
use strum_macros::AsRefStr;

#[derive(From)]
pub enum CudnnApi {
    CudnnOps(CudnnOps),
    CudnnConvForward(CudnnConvForward),
}

impl CudnnApi {
    delegate! {
        to match self {
            CudnnApi::CudnnOps(cudnn_ops) => cudnn_ops,
            CudnnApi::CudnnConvForward(cudnn_conv_fwd) => cudnn_conv_fwd,
        } {
            pub fn fragment(&self) -> String;
        }
    }
}

impl DataType {
    fn cudnn(&self) -> &'static str {
        match self {
            DataType::SInt(SIntType::I32) => "CUDNN_DATA_INT32",
            DataType::SInt(SIntType::I64) => "CUDNN_DATA_INT64",
            DataType::UInt(UIntType::U64) => "CUDNN_DATA_UINT64",
            DataType::Float(FloatType::F32) => "CUDNN_DATA_FLOAT",
            DataType::Float(FloatType::F64) => "CUDNN_DATA_DOUBLE",
        }
    }
}

#[derive(AsRefStr)]
pub enum CudnnConvolutionMode {
    #[strum(serialize = "CUDNN_CONVOLUTION")]
    Convolution,

    #[strum(serialize = "CUDNN_CROSS_CORRELATION")]
    CrossCorrelation,
}

#[derive(AsRefStr)]
pub enum CudnnTensorFormat {
    #[strum(serialize = "CUDNN_TENSOR_NCHW")]
    NCHW,

    #[strum(serialize = "CUDNN_TENSOR_NHWC")]
    NHWC,
}

#[derive(AsRefStr, Clone, Copy)]
pub enum CudnnConvolutionFwdAlgo {
    #[strum(serialize = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM")]
    ImplicitPrecompGemm,
}

pub trait CudnnIdentifier {
    fn input_descriptor(&self) -> String;
    fn output_descriptor(&self) -> String;
    fn filter_descriptor(&self) -> String;
    fn convolution_descriptor(&self) -> String;
    fn workspace_size(&self) -> String;
}

impl CudnnIdentifier for KernelId {
    fn input_descriptor(&self) -> String {
        format!("input_desc{}", self.index())
    }

    fn output_descriptor(&self) -> String {
        format!("output_desc{}", self.index())
    }

    fn filter_descriptor(&self) -> String {
        format!("filter_desc{}", self.index())
    }

    fn convolution_descriptor(&self) -> String {
        format!("conv_desc{}", self.index())
    }

    fn workspace_size(&self) -> String {
        format!("workspace_size{}", self.index())
    }
}

#[derive(Clone, Copy)]
pub struct TensorDescriptor {
    pub id: KernelId,
    pub is_input: bool,
}

impl TensorDescriptor {
    fn fragment(&self) -> String {
        if self.is_input {
            self.id.input_descriptor()
        } else {
            self.id.output_descriptor()
        }
    }
}

#[derive(Clone, Copy)]
pub struct CudnnHandler {
    pub stream_id: StreamId,
}

impl CudnnHandler {
    pub fn new(stream_id: StreamId) -> Self {
        Self { stream_id }
    }

    pub fn handler(&self) -> String {
        format!("cudnn_handler{}", self.stream_id.0)
    }

    pub fn workspace_ptr(&self) -> String {
        format!("workspace_ptr{}", self.stream_id.0)
    }

    pub fn workspace_max_size(&self) -> String {
        format!("workspace_max_size{}", self.stream_id.0)
    }
}

// TODO: Bias
pub enum CudnnOps {
    Create(CudnnHandler),
    CreateTensorDescriptor(TensorDescriptor),
    CreateFilterDescriptor(KernelId),
    CreateConvolutionDescriptor(KernelId),

    SetTensor4dDescriptor {
        desc: TensorDescriptor,
        data_type: DataType,
        format: CudnnTensorFormat,
        nbatch: usize,
        channels: usize,
        height: usize,
        width: usize,
    },
    SetFilter4dDescriptor {
        id: KernelId,
        data_type: DataType,
        format: CudnnTensorFormat,
        out_feature_maps: usize,
        in_feature_maps: usize,
        height: usize,
        width: usize,
    },
    SetConvolution2dDescriptor {
        id: KernelId,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        mode: CudnnConvolutionMode,
        ty: DataType,
    },
    SetStream(CudnnHandler),

    GetConvolutionForwardWorkspaceSize {
        handler: CudnnHandler,
        id: KernelId,
        algo: CudnnConvolutionFwdAlgo,
    },
}

impl CudnnOps {
    pub fn fragment(&self) -> String {
        match self {
            Self::Create(handler) => {
                format!("cudnnCreate(&{})", handler.handler())
            }
            Self::CreateTensorDescriptor(desc) => {
                format!("cudnnCreateTensorDescriptor(&{})", desc.fragment())
            }
            Self::CreateFilterDescriptor(id) => {
                format!("cudnnCreateFilterDescriptor(&{})", id.filter_descriptor())
            }
            Self::CreateConvolutionDescriptor(id) => format!(
                "cudnnCreateConvolutionDescriptor(&{})",
                id.convolution_descriptor()
            ),
            Self::SetTensor4dDescriptor {
                desc,
                data_type,
                format,
                nbatch,
                channels,
                height,
                width,
            } => {
                format!(
                    "cudnnSetTensor4dDescriptor({desc}, {format}, {ty}, {n}, {c}, {h}, {w})",
                    desc = desc.fragment(),
                    format = format.as_ref(),
                    ty = data_type.cudnn(),
                    n = nbatch,
                    c = channels,
                    h = height,
                    w = width,
                )
            }
            Self::SetFilter4dDescriptor {
                id,
                data_type,
                format,
                out_feature_maps,
                in_feature_maps,
                height,
                width,
            } => {
                format!(
                    "cudnnSetFilter4dDescriptor({desc}, {ty}, {format}, {k}, {c}, {h}, {w})",
                    desc = id.filter_descriptor(),
                    ty = data_type.cudnn(),
                    format = format.as_ref(),
                    k = out_feature_maps,
                    c = in_feature_maps,
                    h = height,
                    w = width,
                )
            }
            Self::SetConvolution2dDescriptor {
                id,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
                dilation_h,
                dilation_w,
                mode,
                ty,
            } => {
                format!(
                    "cudnnSetConvolution2dDescriptor({desc}, {pad_h}, {pad_w}, {stride_h}, {stride_w}, {dilation_h}, {dilation_w}, {mode}, {ty})",
                    desc = id.convolution_descriptor(),
                    pad_h = pad_h,
                    pad_w = pad_w,
                    stride_h = stride_h,
                    stride_w = stride_w,
                    dilation_h = dilation_h,
                    dilation_w = dilation_w,
                    mode = mode.as_ref(),
                    ty = ty.cudnn()
                )
            }
            Self::SetStream(handler) => {
                format!(
                    "cudnnSetStream({handler}, {stream})",
                    handler = handler.handler(),
                    stream = handler.stream_id.to_identifier().fragment()
                )
            }
            Self::GetConvolutionForwardWorkspaceSize { handler, id, algo } => {
                format!(
                    "cudnnGetConvolutionForwardWorkspaceSize({handler}, {input_desc}, {filter_desc}, {conv_desc}, {output_desc}, {algo}, &{workspace_size})",
                    handler = handler.handler(),
                    input_desc = id.input_descriptor(),
                    filter_desc = id.filter_descriptor(),
                    conv_desc = id.convolution_descriptor(),
                    output_desc = id.output_descriptor(),
                    algo = algo.as_ref(),
                    workspace_size = id.workspace_size()
                )
            }
        }
    }
}

pub struct CudnnConvForward {
    pub id: KernelId,

    pub handler: CudnnHandler,
    pub alpha: Expr,
    pub in_: Expr,
    pub weights: Expr,
    pub algo: CudnnConvolutionFwdAlgo,
    pub beta: Expr,
    pub out: Expr,
}

impl CudnnConvForward {
    pub fn fragment(&self) -> String {
        assert!(!matches!(self.alpha, Expr::Literal(_)));
        assert!(!matches!(self.beta, Expr::Literal(_)));
        let args = vec![
            self.handler.handler(),
            self.alpha.ref_fragment(),
            self.id.input_descriptor(),
            self.in_.fragment(),
            self.id.filter_descriptor(),
            self.weights.fragment(),
            self.id.convolution_descriptor(),
            self.algo.as_ref().to_string(),
            self.handler.workspace_ptr(),
            self.id.workspace_size(),
            self.beta.ref_fragment(),
            self.id.output_descriptor(),
            self.out.fragment(),
        ];
        format!("cudnnConvolutionForward({})", args.join(", "))
    }
}

macro_rules! impl_into_stmt {
    ($name:ident) => {
        impl From<$name> for Statement {
            fn from(x: $name) -> Statement {
                Statement::CudnnApi(x.into())
            }
        }
    };
}

impl_into_stmt!(CudnnOps);
impl_into_stmt!(CudnnConvForward);
