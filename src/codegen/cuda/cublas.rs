use std::fmt::Display;

use strum_macros::AsRefStr;

use crate::codegen::cuda::*;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;

#[allow(dead_code)]
#[derive(AsRefStr)]
pub enum CublasOperation {
    #[strum(serialize = "CUBLAS_OP_N")]
    Non,

    #[strum(serialize = "CUBLAS_OP_T")]
    Transpose,

    #[strum(serialize = "CUBLAS_OP_C")]
    ConjugateTranspose,
}

impl Display for CublasOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

pub struct GemmArgs {
    pub handler: CublasHandler,
    pub trans_a: CublasOperation,
    pub trans_b: CublasOperation,
    pub a: String,
    pub b: String,
    pub c: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,

    pub alpha: String,
    pub beta: String,

    pub data_ty: DataType,
}

#[derive(Clone, Copy)]
pub struct CublasHandler(StreamId);

impl CublasHandler {
    pub fn new(id: StreamId) -> Self {
        Self(id)
    }
}

impl Display for CublasHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cublas_handle_{}", self.0.index())
    }
}

pub enum CublasApi {
    Create(CublasHandler),
    SetStream(CublasHandler),
    Gemm(GemmArgs),
}

impl Display for CublasApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(handler) => {
                write!(f, "cublasCreate(&{})", handler)
            }
            Self::SetStream(handler) => {
                write!(f, "cublasSetStream({}, {})", handler, handler.0)
            }
            Self::Gemm(GemmArgs {
                handler,
                trans_a,
                trans_b,
                a,
                b,
                c,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                alpha,
                beta,
                data_ty,
            }) => {
                let (c_ty, prefix) = match data_ty {
                    DataType::Float(FloatType::F32) => ("float", 'S'),
                    DataType::Float(FloatType::F64) => ("double", 'D'),
                    _ => panic!("Unsupported data type for cuBLAS GEMM"),
                };
                write!(f,
                    "cublas{}gemm({}, {}, {}, {}, {}, {}, &{}, (const {} *){}, {}, (const {} *){}, {}, &{}, ({} *){}, {})",
                    prefix,
                    handler,
                    trans_a,
                    trans_b,
                    m,
                    n,
                    k,
                    alpha,
                    c_ty,
                    a,
                    lda,
                    c_ty,
                    b,
                    ldb,
                    beta,
                    c_ty,
                    c,
                    ldc,
                )
            }
        }
    }
}
