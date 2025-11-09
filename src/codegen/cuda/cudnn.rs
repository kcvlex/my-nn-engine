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
}

impl CudnnApi {
    delegate! {
        to match self {
            CudnnApi::CudnnOps(cudnn_ops) => cudnn_ops,
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

#[allow(dead_code)]
#[derive(AsRefStr)]
pub enum CudnnConvolutionMode {
    #[strum(serialize = "CUDNN_CONVOLUTION")]
    Convolution,

    #[strum(serialize = "CUDNN_CROSS_CORRELATION")]
    CrossCorrelation,
}

#[allow(dead_code)]
#[derive(AsRefStr)]
pub enum CudnnTensorFormat {
    #[strum(serialize = "CUDNN_TENSOR_NCHW")]
    NCHW,

    #[strum(serialize = "CUDNN_TENSOR_NHWC")]
    NHWC,
}

#[allow(dead_code)]
#[derive(AsRefStr, Clone, Copy)]
pub enum CudnnActivationMode {
    #[strum(serialize = "CUDNN_ACTIVATION_SIGMOID")]
    Sigmoid,

    #[strum(serialize = "CUDNN_ACTIVATION_RELU")]
    Relu,

    #[strum(serialize = "CUDNN_ACTIVATION_TANH")]
    Tanh,

    #[strum(serialize = "CUDNN_ACTIVATION_CLIPPED_RELU")]
    ClippedRelu,

    #[strum(serialize = "CUDNN_ACTIVATION_ELU")]
    Elu,

    #[strum(serialize = "CUDNN_ACTIVATION_IDENTITY")]
    Identity,

    #[strum(serialize = "CUDNN_ACTIVATION_SWISH")]
    Swish,
}

#[allow(dead_code)]
#[derive(AsRefStr)]
pub enum CudnnNanPropagation {
    #[strum(serialize = "CUDNN_NOT_PROPAGATE_NAN")]
    NotPropagateNan,

    #[strum(serialize = "CUDNN_PROPAGATE_NAN")]
    PropagateNan,
}

pub trait CudnnIdentifier {
    fn setting(&self) -> String;

    fn input_descriptor(&self) -> String {
        format!("{}.x_desc", self.setting())
    }

    fn output_descriptor(&self) -> String {
        format!("{}.y_desc", self.setting())
    }

    fn filter_descriptor(&self) -> String {
        format!("{}.w_desc", self.setting())
    }

    fn bias_descriptor(&self) -> String {
        format!("{}.bias_desc", self.setting())
    }

    fn convolution_descriptor(&self) -> String {
        format!("{}.conv_desc", self.setting())
    }

    fn activation_descriptor(&self) -> String {
        format!("{}.activation_desc", self.setting())
    }

    fn workspace_size(&self) -> String {
        format!("{}.workspace_size_in_bytes", self.setting())
    }

    fn fwd_algo(&self) -> String {
        format!("{}.algo", self.setting())
    }
}

#[derive(Clone, Copy)]
pub enum CudnnSettingName {
    DefaultName,
    KernelId(KernelId),
}

impl CudnnIdentifier for CudnnSettingName {
    fn setting(&self) -> String {
        match self {
            CudnnSettingName::DefaultName => "setting".to_string(),
            CudnnSettingName::KernelId(id) => format!("cudnn_setting{}", id.index()),
        }
    }
}

#[derive(Clone, Copy)]
pub enum TensorRole {
    Input,
    Bias,
    Output,
}

#[derive(Clone, Copy)]
pub struct TensorDescriptor {
    pub id: CudnnSettingName,
    pub role: TensorRole,
}

impl TensorDescriptor {
    fn fragment(&self) -> String {
        match self.role {
            TensorRole::Input => self.id.input_descriptor(),
            TensorRole::Bias => self.id.bias_descriptor(),
            TensorRole::Output => self.id.output_descriptor(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum CudnnContext {
    DefaultContext,
    StreamContext(StreamId),
}

impl CudnnContext {
    pub fn ctx(&self) -> String {
        match self {
            CudnnContext::DefaultContext => "cudnn_handler_ctx".to_string(),
            CudnnContext::StreamContext(id) => format!("cudnn_handler_ctx{}", id.0),
        }
    }

    pub fn handler(&self) -> String {
        format!("{}.handle", self.ctx())
    }

    pub fn workspace_ptr(&self) -> String {
        format!("{}.workspace", self.ctx())
    }

    pub fn workspace_max_size(&self) -> String {
        format!("{}.workspace_max_size_in_bytes", self.ctx())
    }
}

// TODO: Bias
pub enum CudnnOps {
    Create(CudnnContext),
    CreateTensorDescriptor(TensorDescriptor),
    CreateFilterDescriptor(CudnnSettingName),
    CreateConvolutionDescriptor(CudnnSettingName),
    CreateActivationDescriptor(CudnnSettingName),

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
        id: CudnnSettingName,
        data_type: DataType,
        format: CudnnTensorFormat,
        out_feature_maps: usize,
        in_feature_maps: usize,
        height: usize,
        width: usize,
    },
    SetConvolution2dDescriptor {
        id: CudnnSettingName,
        pad_h: usize,
        pad_w: usize,
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        mode: CudnnConvolutionMode,
        ty: DataType,
    },
    SetActivationDescriptor {
        id: CudnnSettingName,
        mode: CudnnActivationMode,
        nan_prop: CudnnNanPropagation,

        // ceiling for clipped RELU, alpha for ELU (copied from cudnn_ops.h)
        coef: f64,
    },
    SetStream(StreamId),

    GetConvolutionForwardWorkspaceSize {
        ctx: CudnnContext,
        id: CudnnSettingName,
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
            Self::CreateActivationDescriptor(id) => format!(
                "cudnnCreateActivationDescriptor(&{})",
                id.activation_descriptor()
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
            Self::SetActivationDescriptor {
                id,
                mode,
                nan_prop,
                coef,
            } => {
                format!(
                    "cudnnSetActivationDescriptor({desc}, {mode}, {nan_prop}, {coef})",
                    desc = id.activation_descriptor(),
                    mode = mode.as_ref(),
                    nan_prop = nan_prop.as_ref(),
                    coef = coef
                )
            }
            Self::SetStream(stream_id) => {
                format!(
                    "cudnnSetStream({handler}, {stream})",
                    handler = CudnnContext::StreamContext(*stream_id).handler(),
                    stream = stream_id.to_identifier().fragment()
                )
            }
            Self::GetConvolutionForwardWorkspaceSize { ctx: handler, id } => {
                format!(
                    "cudnnGetConvolutionForwardWorkspaceSize({handler}, {input_desc}, {filter_desc}, {conv_desc}, {output_desc}, {algo}, &{workspace_size})",
                    handler = handler.handler(),
                    input_desc = id.input_descriptor(),
                    filter_desc = id.filter_descriptor(),
                    conv_desc = id.convolution_descriptor(),
                    output_desc = id.output_descriptor(),
                    algo = id.fwd_algo(),
                    workspace_size = id.workspace_size(),
                )
            }
        }
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
