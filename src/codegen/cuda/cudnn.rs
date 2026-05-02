use derive_more::From;

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
    pub fn cudnn(&self) -> &'static str {
        match self {
            DataType::Bool => "CUDNN_DATA_INT8",
            DataType::SInt(SIntType::I8) => "CUDNN_DATA_INT8",
            DataType::SInt(SIntType::I32) => "CUDNN_DATA_INT32",
            DataType::SInt(SIntType::I64) => "CUDNN_DATA_INT64",
            DataType::UInt(UIntType::U8) => "CUDNN_DATA_UINT8",
            DataType::UInt(UIntType::U64) => "CUDNN_DATA_UINT64",
            DataType::Float(FloatType::BF16) => "CUDNN_DATA_BFLOAT16",
            DataType::Float(FloatType::F32) => "CUDNN_DATA_FLOAT",
            DataType::Float(FloatType::F64) => "CUDNN_DATA_DOUBLE",
        }
    }
}

#[derive(Clone, Copy)]
pub enum CudnnSettingName {
    DefaultName,
    KernelId(KernelId),
}

impl CudnnSettingName {
    pub fn setting(&self) -> String {
        match self {
            CudnnSettingName::DefaultName => "setting".to_string(),
            CudnnSettingName::KernelId(id) => format!("cudnn_setting{}", id.index()),
        }
    }

    pub fn state_setting(&self) -> String {
        match self {
            CudnnSettingName::DefaultName => "setting".to_string(),
            CudnnSettingName::KernelId(id) => format!("state->cudnn_setting{}", id.index()),
        }
    }
}

#[derive(Clone, Copy)]
pub struct CudnnContext {
    stream_id: StreamId,
    state_prefix: bool,
}

impl CudnnContext {
    pub fn new(stream_id: StreamId) -> Self {
        Self {
            stream_id,
            state_prefix: false,
        }
    }

    pub fn with_state_prefix(self) -> Self {
        Self {
            state_prefix: true,
            ..self
        }
    }

    fn prefix(&self) -> &'static str {
        if self.state_prefix {
            "state->"
        } else {
            ""
        }
    }

    pub fn ctx(&self) -> String {
        format!(
            "{}cudnn_handler_ctx{}",
            self.prefix(),
            self.stream_id.index()
        )
    }

    pub fn handler(&self) -> String {
        format!("{}.handle", self.ctx())
    }
}

pub enum CudnnOps {
    Create(CudnnContext),
    Destroy(CudnnContext),
    SetStream(CudnnContext),
}

impl std::fmt::Display for CudnnOps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(handler) => {
                write!(f, "cudnnCreate(&{})", handler.handler())
            }
            Self::Destroy(handler) => {
                write!(f, "cudnnDestroy({})", handler.handler())
            }
            Self::SetStream(ctx) => {
                let stream_id = ctx.stream_id;
                write!(
                    f,
                    "cudnnSetStream({}, {}{})",
                    ctx.handler(),
                    ctx.prefix(),
                    stream_id
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
