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

pub struct BatchedGemmArgs {
    pub gemm: GemmArgs,
    pub stride_a: usize,
    pub stride_b: usize,
    pub stride_c: usize,
    pub batch_count: usize,
}

#[derive(Clone, Copy)]
pub struct CublasHandler {
    stream_id: StreamId,
    state_prefix: bool,
}

impl CublasHandler {
    pub fn new(id: StreamId) -> Self {
        Self {
            stream_id: id,
            state_prefix: false,
        }
    }

    pub fn with_state_prefix(self) -> Self {
        Self {
            state_prefix: true,
            ..self
        }
    }
}

impl Display for CublasHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.state_prefix {
            write!(f, "state->cublas_handle_{}", self.stream_id.index())
        } else {
            write!(f, "cublas_handle_{}", self.stream_id.index())
        }
    }
}

pub enum CublasApi {
    Create(CublasHandler),
    SetStream(CublasHandler),
    Gemm(GemmArgs),
    BatchedGemm(BatchedGemmArgs),
    Destroy(CublasHandler),
}

impl Display for CublasApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Create(handler) => {
                write!(f, "cublasCreate(&{})", handler)
            }
            Self::SetStream(handler) => {
                let stream_id = handler.stream_id;
                if handler.state_prefix {
                    write!(f, "cublasSetStream({}, state->{})", handler, stream_id)
                } else {
                    write!(f, "cublasSetStream({}, {})", handler, stream_id)
                }
            }
            Self::Destroy(handler) => {
                write!(f, "cublasDestroy({})", handler)
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
                if matches!(data_ty, DataType::Float(FloatType::BF16)) {
                    write!(
                        f,
                        "
cublasGemmEx(
    {handler},
    {trans_a},
    {trans_b},
    {m}, {n}, {k},
    &{alpha},
    {a}, CUDA_R_16BF, {lda},
    {b}, CUDA_R_16BF, {ldb},
    &{beta},
    {c}, CUDA_R_16BF, {ldc},
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT
)
",
                    )
                } else {
                    let (c_ty, prefix) = match data_ty {
                        DataType::Float(FloatType::F32) => ("float", 'S'),
                        DataType::Float(FloatType::F64) => ("double", 'D'),
                        _ => panic!("Unsupported data type for cuBLAS GEMM"),
                    };
                    write!(
                        f,
                        "
cublas{prefix}gemm(
    {handler},
    {trans_a},
    {trans_b},
    {m}, {n}, {k},
    &{alpha},
    (const {c_ty} *){a}, {lda},
    (const {c_ty} *){b}, {ldb},
    &{beta},
    ({c_ty} *){c}, {ldc}
)
",
                    )
                }
            }
            Self::BatchedGemm(BatchedGemmArgs {
                gemm,
                stride_a,
                stride_b,
                stride_c,
                batch_count,
            }) => {
                let GemmArgs {
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
                } = gemm;
                if matches!(data_ty, DataType::Float(FloatType::BF16)) {
                    write!(
                        f,
                        "
cublasGemmStridedBatchedEx(
    {handler},
    {trans_a},
    {trans_b},
    {m}, {n}, {k},
    &{alpha},
    {a}, CUDA_R_16BF, {lda}, {stride_a},
    {b}, CUDA_R_16BF, {ldb}, {stride_b},
    &{beta},
    {c}, CUDA_R_16BF, {ldc}, {stride_c},
    {batch_count},
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT
)
",
                    )
                } else {
                    let (c_ty, prefix) = match data_ty {
                        DataType::Float(FloatType::F32) => ("float", 'S'),
                        DataType::Float(FloatType::F64) => ("double", 'D'),
                        _ => panic!("Unsupported data type for cuBLAS GEMM"),
                    };
                    write!(
                        f,
                        "
cublas{prefix}gemmStridedBatched(
    {handler},
    {trans_a},
    {trans_b},
    {m}, {n}, {k},
    &{alpha},
    (const {c_ty} *){a}, {lda}, {stride_a},
    (const {c_ty} *){b}, {ldb}, {stride_b},
    &{beta},
    ({c_ty} *){c}, {ldc}, {stride_c},
    {batch_count}
)
",
                    )
                }
            }
        }
    }
}
