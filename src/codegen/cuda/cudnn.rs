use derive_more::From;
use strum_macros::AsRefStr;

use crate::codegen::cuda::*;
use crate::schedule::*;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

#[derive(From)]
pub enum CudnnApi {
    CudnnOps(CudnnOps),
}

impl std::fmt::Display for CudnnApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudnnApi::CudnnOps(cudnn_ops) => write!(f, "{}", cudnn_ops),
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

impl std::fmt::Display for TensorDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self.role {
                TensorRole::Input => self.id.input_descriptor(),
                TensorRole::Bias => self.id.bias_descriptor(),
                TensorRole::Output => self.id.output_descriptor(),
            }
        )
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
            CudnnContext::StreamContext(id) => format!("cudnn_handler_ctx{}", id.index()),
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

    Destroy(CudnnContext),
    DestroyTensorDescriptor(TensorDescriptor),
    DestroyFilterDescriptor(CudnnSettingName),
    DestroyConvolutionDescriptor(CudnnSettingName),
    DestroyActivationDescriptor(CudnnSettingName),

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

impl std::fmt::Display for CudnnOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(handler) => {
                write!(f, "cudnnCreate(&{})", handler.handler())
            }
            Self::CreateTensorDescriptor(desc) => {
                write!(f, "cudnnCreateTensorDescriptor(&{})", desc)
            }
            Self::CreateFilterDescriptor(id) => {
                write!(
                    f,
                    "cudnnCreateFilterDescriptor(&{})",
                    id.filter_descriptor()
                )
            }
            Self::CreateConvolutionDescriptor(id) => write!(
                f,
                "cudnnCreateConvolutionDescriptor(&{})",
                id.convolution_descriptor()
            ),
            Self::CreateActivationDescriptor(id) => write!(
                f,
                "cudnnCreateActivationDescriptor(&{})",
                id.activation_descriptor()
            ),

            Self::Destroy(handler) => {
                write!(f, "cudnnDestroy({})", handler.handler())
            }
            Self::DestroyTensorDescriptor(desc) => {
                write!(f, "cudnnDestroyTensorDescriptor({})", desc)
            }
            Self::DestroyFilterDescriptor(id) => {
                write!(
                    f,
                    "cudnnDestroyFilterDescriptor({})",
                    id.filter_descriptor()
                )
            }
            Self::DestroyConvolutionDescriptor(id) => write!(
                f,
                "cudnnDestroyConvolutionDescriptor({})",
                id.convolution_descriptor()
            ),
            Self::DestroyActivationDescriptor(id) => write!(
                f,
                "cudnnDestroyActivationDescriptor({})",
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
                write!(
                    f,
                    "cudnnSetTensor4dDescriptor({}, {}, {}, {}, {}, {}, {})",
                    desc,
                    format.as_ref(),
                    data_type.cudnn(),
                    nbatch,
                    channels,
                    height,
                    width,
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
                write!(
                    f,
                    "cudnnSetFilter4dDescriptor({}, {}, {}, {}, {}, {}, {})",
                    id.filter_descriptor(),
                    data_type.cudnn(),
                    format.as_ref(),
                    out_feature_maps,
                    in_feature_maps,
                    height,
                    width,
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
                write!(
                    f,
                    "cudnnSetConvolution2dDescriptor({}, {}, {}, {}, {}, {}, {}, {}, {})",
                    id.convolution_descriptor(),
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    mode.as_ref(),
                    ty.cudnn()
                )
            }
            Self::SetActivationDescriptor {
                id,
                mode,
                nan_prop,
                coef,
            } => {
                write!(
                    f,
                    "cudnnSetActivationDescriptor({}, {}, {}, {})",
                    id.activation_descriptor(),
                    mode.as_ref(),
                    nan_prop.as_ref(),
                    coef
                )
            }
            Self::SetStream(stream_id) => {
                write!(
                    f,
                    "cudnnSetStream({}, {})",
                    CudnnContext::StreamContext(*stream_id).handler(),
                    stream_id
                )
            }
            Self::GetConvolutionForwardWorkspaceSize { ctx: handler, id } => {
                write!(
                    f,
                    "cudnnGetConvolutionForwardWorkspaceSize({}, {}, {}, {}, {}, {}, &{})",
                    handler.handler(),
                    id.input_descriptor(),
                    id.filter_descriptor(),
                    id.convolution_descriptor(),
                    id.output_descriptor(),
                    id.fwd_algo(),
                    id.workspace_size(),
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
