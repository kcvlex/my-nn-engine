use std::fmt::Display;

use delegate::delegate;
use derive_more::From;
use itertools::izip;
use itertools::Itertools;

use crate::codegen::cuda::*;
use crate::graph::operator;
use crate::graph::operator::args;
use crate::graph::operator::ReinterpretType;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;

#[derive(From)]
#[allow(clippy::enum_variant_names, clippy::upper_case_acronyms)]
pub enum CUDAKernel {
    AttentionKernel(AttentionKernel),
    AttentionDecodeKernel(AttentionDecodeKernel),
    ConcatKernel(ConcatKernel),
    DequantizeLinearKernel(DequantizeLinearKernel),
    DequantGemvKernel(DequantGemvKernel),
    DequantMatMulKernel(DequantMatMulKernel),
    DequantMatMulWmmaKernel(DequantMatMulWmmaKernel),
    DynamicQuantizeLinearKernel(DynamicQuantizeLinearKernel),
    QuantizedGemvInt8Kernel(QuantizedGemvInt8Kernel),
    QuantizedMatMulInt8Kernel(QuantizedMatMulInt8Kernel),
    ExpandKernel(ExpandKernel),
    GatherKernel(GatherKernel),
    GeneratedKernel(GeneratedKernel),
    KVCacheUpdateKernel(KVCacheUpdateKernel),
    QuantizingKVCacheUpdateKernel(QuantizingKVCacheUpdateKernel),
    LayerNormKernel(LayerNormKernel),
    ReduceMatrixKernel(ReduceMatrixKernel),
    RMSNormKernel(RMSNormKernel),
    RopeKernel(RopeKernel),
    SliceKernel(SliceKernel),
    SoftmaxKernel(SoftmaxKernel),
}

macro_rules! cast {
    ($ty:expr, $e:expr) => {
        format!("({} *)({})", $ty, $e)
    };
}

/// Streaming KV layout (sink + ring) parameters for attention/KV-update
/// kernels. `None` selects the legacy contiguous layout (the kernel sees
/// `ring_sink == ring_window == ring_start == 0`, which the device-side
/// `ring_phys_index` collapses to identity, so emission stays bit-identical).
#[derive(Clone)]
pub struct RingArgs {
    pub sink: Expr,
    pub window: Expr,
    pub start: Expr,
}

impl RingArgs {
    fn args(opt: &Option<RingArgs>) -> [String; 3] {
        match opt {
            Some(r) => [
                r.sink.to_string(),
                r.window.to_string(),
                r.start.to_string(),
            ],
            None => ["0".to_string(), "0".to_string(), "0".to_string()],
        }
    }
}

#[derive(Clone)]
pub struct RopeAttnArgs {
    pub cos_table: Expr,
    pub sin_table: Expr,
    pub kv_position: Expr,
}

impl RopeAttnArgs {
    fn args(opt: &Option<RopeAttnArgs>, data_ty: DataType) -> [String; 3] {
        match opt {
            Some(r) => [
                cast!(data_ty, r.cos_table),
                cast!(data_ty, r.sin_table),
                cast!("long long", r.kv_position),
            ],
            None => [
                "nullptr".to_string(),
                "nullptr".to_string(),
                "nullptr".to_string(),
            ],
        }
    }
}

pub struct AttentionKernel {
    pub data_ty: DataType,
    pub br: usize,
    pub bc: usize,
    pub threads_per_row: usize,
    pub head_dim: usize,

    pub out: Expr,
    pub q: Expr,
    pub k: Expr,
    pub v: Expr,
    pub mask: Option<Expr>,
    pub q_seq_len: usize,
    pub kv_active_seq: Expr,
    pub kv_cache_stride: usize,
    pub q_pos_offset: Expr,
    pub mask_outer_stride: usize,
    pub mask_row_stride: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,

    /// When `Some`, K/V are stored as INT8 with per-(head, token) scale of dtype
    /// `data_ty`, and the `attention_int8` kernel is emitted instead.
    pub kv_quant: Option<KvQuantArgs>,

    pub ring: Option<RingArgs>,
    pub rope: Option<RopeAttnArgs>,

    pub attn: Attention,

    pub use_tensor_core: bool,
}

#[derive(Clone)]
pub struct KvQuantArgs {
    pub k_scale: Expr,
    pub v_scale: Expr,
}

impl AttentionKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let mask = match &self.mask {
            Some(mask) => mask.to_owned(),
            None => "nullptr".to_literal(),
        };
        let kv_ty = if self.kv_quant.is_some() {
            "signed char".to_string()
        } else {
            self.data_ty.to_string()
        };
        let (k_scale, v_scale) = match &self.kv_quant {
            Some(q) => (
                cast!(self.data_ty, q.k_scale),
                cast!(self.data_ty, q.v_scale),
            ),
            None => ("nullptr".to_string(), "nullptr".to_string()),
        };
        let kv_cast = |e: &Expr| {
            if self.kv_quant.is_some() {
                format!("(signed char *)({})", e)
            } else {
                cast!(self.data_ty, e)
            }
        };
        let name = if self.use_tensor_core {
            "attention_tc"
        } else {
            "attention"
        };
        let id = format!(
            "{}<{}, {}, {}, {}, {}, {}>",
            name, self.data_ty, kv_ty, self.br, self.bc, self.threads_per_row, self.head_dim,
        );
        let [ring_sink, ring_window, ring_start] = RingArgs::args(&self.ring);
        let [cos_table, sin_table, kv_position] = RopeAttnArgs::args(&self.rope, self.data_ty);
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.q),
            kv_cast(&self.k),
            kv_cast(&self.v),
            k_scale,
            v_scale,
            self.attn.scale.to_string(),
            if self.attn.is_causal { "1" } else { "0" }.to_string(),
            cast!(self.data_ty, mask),
            self.mask_outer_stride.to_string(),
            self.mask_row_stride.to_string(),
            self.q_seq_len.to_string(),
            self.kv_active_seq.to_string(),
            self.kv_cache_stride.to_string(),
            self.q_pos_offset.to_string(),
            self.num_q_heads.to_string(),
            self.num_kv_heads.to_string(),
            ring_sink,
            ring_window,
            ring_start,
            cos_table,
            sin_table,
            kv_position,
        ];
        (id, args)
    }
}

pub struct AttentionDecodeKernel {
    pub data_ty: DataType,
    pub head_dim: usize,
    pub block_size: usize,

    pub out: Expr,
    pub q: Expr,
    pub k: Expr,
    pub v: Expr,
    pub cache_seq_len: usize,
    pub active_seq_kv: Expr,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,

    pub kv_quant: Option<KvQuantArgs>,
    pub ring: Option<RingArgs>,
    pub rope: Option<RopeAttnArgs>,
    pub attn: Attention,
}

impl AttentionDecodeKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let kv_ty = if self.kv_quant.is_some() {
            "signed char".to_string()
        } else {
            self.data_ty.to_string()
        };
        let (k_scale, v_scale) = match &self.kv_quant {
            Some(q) => (
                cast!(self.data_ty, q.k_scale),
                cast!(self.data_ty, q.v_scale),
            ),
            None => ("nullptr".to_string(), "nullptr".to_string()),
        };
        let kv_cast = |e: &Expr| {
            if self.kv_quant.is_some() {
                format!("(signed char *)({})", e)
            } else {
                cast!(self.data_ty, e)
            }
        };
        let id = format!(
            "attention_decode<{}, {}, {}, {}>",
            self.data_ty, kv_ty, self.head_dim, self.block_size,
        );
        let [ring_sink, ring_window, ring_start] = RingArgs::args(&self.ring);
        let [cos_table, sin_table, kv_position] = RopeAttnArgs::args(&self.rope, self.data_ty);
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.q),
            kv_cast(&self.k),
            kv_cast(&self.v),
            k_scale,
            v_scale,
            self.attn.scale.to_string(),
            self.cache_seq_len.to_string(),
            self.active_seq_kv.to_string(),
            self.num_q_heads.to_string(),
            self.num_kv_heads.to_string(),
            ring_sink,
            ring_window,
            ring_start,
            cos_table,
            sin_table,
            kv_position,
        ];
        (id, args)
    }
}

pub struct GeneratedKernel {
    pub decl: KernelDecl,
    pub args: Vec<Expr>,
}

pub struct KVCacheUpdateKernel {
    pub data_ty: DataType,
    pub head_dim: usize,
    pub cache_seq_len: usize,
    pub new_seq_len: usize,

    pub cache: Expr,
    pub new_kv: Expr,
    pub offset: Expr,
    pub ring: Option<RingArgs>,
}

impl KVCacheUpdateKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("kvcache_update<{}, {}>", self.data_ty, self.head_dim);
        let [ring_sink, ring_window, ring_start] = RingArgs::args(&self.ring);
        let args = vec![
            cast!(self.data_ty, self.cache),
            cast!(self.data_ty, self.new_kv),
            self.cache_seq_len.to_string(),
            self.new_seq_len.to_string(),
            self.offset.to_string(),
            ring_sink,
            ring_window,
            ring_start,
        ];
        (id, args)
    }
}

pub struct QuantizingKVCacheUpdateKernel {
    pub new_ty: DataType,
    pub scale_ty: DataType,
    pub head_dim: usize,
    pub block_size: usize,
    pub max_seq_len: usize,
    pub new_seq_len: usize,

    pub cache: Expr,
    pub scale: Expr,
    pub new_kv: Expr,
    pub offset: Expr,
    pub ring: Option<RingArgs>,
}

impl QuantizingKVCacheUpdateKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "quantizing_kvcache_update<{}, {}, {}, {}>",
            self.new_ty, self.scale_ty, self.head_dim, self.block_size
        );
        let [ring_sink, ring_window, ring_start] = RingArgs::args(&self.ring);
        let args = vec![
            format!("(signed char *)({})", self.cache),
            cast!(self.scale_ty, self.scale),
            cast!(self.new_ty, self.new_kv),
            self.max_seq_len.to_string(),
            self.new_seq_len.to_string(),
            self.offset.to_string(),
            ring_sink,
            ring_window,
            ring_start,
        ];
        (id, args)
    }
}

impl GeneratedKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = self.decl.name();
        let args = self
            .args
            .iter()
            .map(|arg| arg.to_string())
            .collect::<Vec<_>>();
        (id, args)
    }
}

pub struct LayerNormKernel {
    pub data_ty: DataType,
    pub block_size: usize,
    pub axis_dim: usize,

    pub out: Expr,
    pub in_: Expr,
    pub scale: Expr,
    pub bias: Expr,
    pub size: Expr,
    pub epsilon: f64,
}

impl LayerNormKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "layer_norm<{}, {}, {}>",
            self.data_ty, self.block_size, self.axis_dim
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            cast!(self.data_ty, self.scale),
            cast!(self.data_ty, self.bias),
            self.epsilon.to_string(),
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct RMSNormKernel {
    pub data_ty: DataType,
    pub block_size: usize,
    pub axis_dim: usize,

    pub out: Expr,
    pub in_: Expr,
    pub scale: Expr,
    pub size: Expr,
    pub epsilon: f64,
}

impl RMSNormKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "rms_norm<{}, {}, {}>",
            self.data_ty, self.block_size, self.axis_dim
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            cast!(self.data_ty, self.scale),
            self.epsilon.to_string(),
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct RopeKernel {
    pub data_ty: DataType,
    pub head_dim: usize,

    pub out: Expr,
    pub x: Expr,
    pub cos_table: Expr,
    pub sin_table: Expr,
    pub position: Expr,
    pub seq_len: usize,
}

impl RopeKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("rope<{}, {}>", self.data_ty, self.head_dim);
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.x),
            cast!(self.data_ty, self.cos_table),
            cast!(self.data_ty, self.sin_table),
            cast!("long long", self.position),
            self.seq_len.to_string(),
        ];
        (id, args)
    }
}

pub struct SoftmaxKernel {
    pub data_ty: DataType,
    pub block_size: usize,
    pub axis_dim: usize,
    pub axis_stride: usize,

    pub out: Expr,
    pub in_: Expr,
    pub size: Expr,
}

impl SoftmaxKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "softmax<{}, {}, {}, {}>",
            self.data_ty, self.block_size, self.axis_dim, self.axis_stride
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct LaunchKernel {
    pub cuda_kernel: CUDAKernel,
    pub grid_size: Expr,
    pub block_size: Expr,
    pub shared_mem_bytes: Option<String>,
    pub stream_id: StreamId,
}

impl LaunchKernel {
    delegate! {
        to match &self.cuda_kernel {
            CUDAKernel::AttentionKernel(a) => a,
            CUDAKernel::AttentionDecodeKernel(a) => a,
            CUDAKernel::ConcatKernel(c) => c,
            CUDAKernel::DequantizeLinearKernel(d) => d,
            CUDAKernel::DequantGemvKernel(d) => d,
            CUDAKernel::DequantMatMulKernel(d) => d,
            CUDAKernel::DequantMatMulWmmaKernel(d) => d,
            CUDAKernel::DynamicQuantizeLinearKernel(d) => d,
            CUDAKernel::QuantizedGemvInt8Kernel(q) => q,
            CUDAKernel::QuantizedMatMulInt8Kernel(q) => q,
            CUDAKernel::ExpandKernel(e) => e,
            CUDAKernel::GatherKernel(g) => g,
            CUDAKernel::GeneratedKernel(g) => g,
            CUDAKernel::KVCacheUpdateKernel(k) => k,
            CUDAKernel::QuantizingKVCacheUpdateKernel(k) => k,
            CUDAKernel::LayerNormKernel(l) => l,
            CUDAKernel::ReduceMatrixKernel(r) => r,
            CUDAKernel::RMSNormKernel(r) => r,
            CUDAKernel::RopeKernel(r) => r,
            CUDAKernel::SliceKernel(s) => s,
            CUDAKernel::SoftmaxKernel(s) => s,
        } {
            #[call(fragment)]
            fn kernel_fragment(&self) -> (String, Vec<String>);
        }
    }
}

impl std::fmt::Display for LaunchKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (id, args) = self.kernel_fragment();
        write!(
            f,
            "{id}<<<{}, {}, {}, {}>>>({args})",
            self.grid_size,
            self.block_size,
            self.shared_mem_bytes.clone().unwrap_or("0".to_string()),
            self.stream_id,
            args = args.join(", ")
        )
    }
}

#[derive(Clone)]
pub enum TypeSymbol {
    Primitive(DataType),
    Pointer(Box<TypeSymbol>),
}

impl TypeSymbol {
    pub fn to_pointer(&self) -> Self {
        TypeSymbol::Pointer(Box::new(self.clone()))
    }
}

impl Display for TypeSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Primitive(dt) => dt.to_string(),
                Self::Pointer(inner) => format!("{}*", inner),
            }
        )
    }
}

impl From<DataType> for TypeSymbol {
    fn from(dt: DataType) -> Self {
        TypeSymbol::Primitive(dt)
    }
}

#[derive(Clone)]
pub enum KernelExpr {
    KernelVar(KernelVar),
    CallFunction { name: String, args: Vec<KernelExpr> },
    Raw(String),
}

#[derive(Clone, Copy)]
pub enum KernelVar {
    Gid,
    Value(ValueId),
    Local(usize),
}

impl From<KernelVar> for KernelExpr {
    fn from(val: KernelVar) -> Self {
        KernelExpr::KernelVar(val)
    }
}

impl Display for KernelExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                KernelExpr::KernelVar(var) => var.to_string(),
                KernelExpr::CallFunction { name, args } => {
                    let args_str: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                    format!("{}({})", name, args_str.join(", "))
                }
                KernelExpr::Raw(s) => s.clone(),
            }
        )
    }
}

impl Display for KernelVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                KernelVar::Gid => "gid".to_string(),
                KernelVar::Value(value_id) => format!("value_{}", value_id.index()),
                KernelVar::Local(idx) => format!("local_{}", idx),
            }
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub enum FuncQualifier {
    Global,
    Device(DataType),
}

impl Display for FuncQualifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                FuncQualifier::Global => "__global__",
                FuncQualifier::Device(_) => "__device__",
            }
        )
    }
}

#[derive(Clone)]
pub struct KernelDecl {
    pub kernel_id: KernelId,
    pub params: Vec<(KernelVar, TypeSymbol)>,
    pub qualifier: FuncQualifier,
}

impl KernelDecl {
    pub fn name(&self) -> String {
        let suffix = match self.qualifier {
            FuncQualifier::Global => "global",
            FuncQualifier::Device(_) => "device",
        };
        format!("kernel_{}_{}", self.kernel_id.index(), suffix)
    }

    pub fn decl(&self) -> String {
        let args = self
            .params
            .iter()
            .map(|(var, ty)| format!("{} {}", ty, var))
            .collect_vec()
            .join(", ");
        let ret_ty = match self.qualifier {
            FuncQualifier::Global => "void".to_string(),
            FuncQualifier::Device(dt) => dt.to_string(),
        };
        let name = self.name();
        let qual = self.qualifier.to_string();

        format!("{qual} {ret_ty} {name}({args})")
    }
}

struct BuilderContext<'sched> {
    schedule: &'sched Schedule,
    decl: KernelDecl,
    local_slot: usize,
}

impl<'sched> BuilderContext<'sched> {
    fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            schedule,
            decl,
            local_slot: 0,
        }
    }

    fn get_resolved_tensor_type(
        &self,
        value_id: ValueId,
    ) -> Result<&ResolvedTensorType, BuildError> {
        self.schedule
            .get_resolved_tensor_type(value_id)
            .ok_or(BuildError::UnresolvedType(value_id))
    }

    fn new_local_var(&mut self) -> KernelVar {
        let var = KernelVar::Local(self.local_slot);
        self.local_slot += 1;
        var
    }

    fn tensor_idx(
        &self,
        value_id: ValueId,
        offset: KernelVar,
        target_dims: Option<&ResolvedTensorDims>,
    ) -> Result<KernelExpr, BuildError> {
        let ty = self.get_resolved_tensor_type(value_id)?;
        let orig_size = ty.dims.size();
        let (ty, broadcasted) = if let Some(target_dims) = target_dims {
            let res = ty.broadcast(target_dims);
            let broadcasted = res.dims != ty.dims;
            (res, broadcasted)
        } else {
            (ty.clone(), false)
        };
        match ty.dims.ndim() {
            1 => {
                if broadcasted {
                    assert!(orig_size == 1);
                    Ok(KernelExpr::Raw("0".to_string()))
                } else {
                    Ok(KernelVar::Gid.into())
                }
            }
            d @ (2..=5) => {
                let name = format!("to_tensor_idx{}d", d);
                let mut args = vec![offset.into()];
                for cnst in ty.dims[1..].iter().chain(ty.strides().iter()) {
                    args.push(KernelExpr::Raw(cnst.to_string()));
                }
                Ok(KernelExpr::CallFunction { name, args })
            }
            d => Err(BuildError::UnsupportedTensorDim(value_id, d)),
        }
    }
}

pub struct ElementwiseKernelBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> ElementwiseKernelBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    fn unary_op(&self, op: &Operator, x: KernelVar) -> KernelExpr {
        KernelExpr::Raw(match op {
            Operator::Cast(Cast { to }) => format!("({})({})", to, x),
            Operator::Cos => format!("cosf({})", x),
            Operator::Exp => format!("exp({})", x),
            Operator::GeLU(GeLU { approximate }) => {
                if !approximate {
                    unimplemented!()
                }
                // x * (0.5 + 0.5 * tanh(x * (sqrt(2/pi) + 0.044715*sqrt(2/pi)*x^2)))
                format!(
                    "({x} * (0.5 + 0.5 * tanh({x} * (0.7978845608028654 + 0.035677408136300125 * {x} * {x}))))"
                )
            }
            Operator::Clip(Clip { min, max }) => {
                let min = min.expect("Clip min must be constant-folded");
                let max = max.expect("Clip max must be constant-folded");
                format!("fmaxf({}, fminf({}, {}))", min, max, x)
            }
            Operator::Identity => format!("{}", x),
            Operator::IsNaN => format!("(isnan({}) ? (int8_t)1 : (int8_t)0)", x),
            Operator::LeakyReLU(LeakyReLU { alpha }) => {
                format!("((0 <= {}) ? {} : {} * {})", x, x, alpha, x)
            }
            Operator::Log => format!("log({})", x),
            Operator::Neg => format!("-({})", x),
            Operator::Reciprocal => format!("(1.0 / {})", x),
            Operator::ReLU => format!("((0 <= {}) ? {} : 0)", x, x),
            Operator::Sigmoid => format!("(1.0 / (1.0 + exp(-{})))", x),
            Operator::Swish(Swish { alpha }) => format!("({x} / (1.0 + exp(-{alpha} * {x})))"),
            Operator::Sin => format!("sinf({})", x),
            Operator::Sqrt => format!("sqrt({})", x),
            Operator::Tanh => format!("tanh({})", x),
            _ => unreachable!(),
        })
    }

    fn binary_op(&self, op: &Operator, lhs: KernelVar, rhs: KernelVar) -> KernelExpr {
        KernelExpr::Raw(match op {
            Operator::Add => format!("({} + {})", lhs, rhs),
            Operator::And => format!("({} && {})", lhs, rhs),
            Operator::Div => format!("({} / {})", lhs, rhs),
            Operator::Equal => format!("({} == {})", lhs, rhs),
            Operator::LessOrEqual => format!("({} <= {})", lhs, rhs),
            Operator::Mul => format!("({} * {})", lhs, rhs),
            // TODO: Support integer types.
            Operator::Pow => format!("pow({}, {})", lhs, rhs),
            Operator::Sub => format!("({} - {})", lhs, rhs),
            _ => unreachable!(),
        })
    }

    fn single_op(&self, op: &Operator, inputs: &[KernelVar]) -> KernelExpr {
        match op {
            Operator::BatchNormalization(BatchNormalization { epsilon, .. }) => {
                let x = inputs[args::BATCHNORM_DATA];
                let scale = inputs[args::BATCHNORM_SCALE];
                let bias = inputs[args::BATCHNORM_BIAS];
                let mean = inputs[args::BATCHNORM_MEAN];
                let var = inputs[args::BATCHNORM_VAR];
                KernelExpr::Raw(format!(
                    "(({x} - {mean}) / sqrt({var} + {epsilon})) * {scale} + {bias}"
                ))
            }
            uop @ (Operator::Cast(_) |
            Operator::Clip(_) |
            Operator::Cos |
            Operator::Exp |
            Operator::GeLU(_) |
            Operator::Identity |
            Operator::IsNaN |
            Operator::LeakyReLU(_) |
            Operator::Log |
            Operator::Neg |
            Operator::Reciprocal |
            Operator::ReLU |
            Operator::Sigmoid |
            Operator::Sin |
            Operator::Sqrt |
            Operator::Swish(_) |
            Operator::Tanh) => {
                let [a] = inputs else {
                    panic!("Expected 1 input for unary operator")
                };
                self.unary_op(uop, *a)
            }
            binop @ (Operator::Add |
            Operator::And |
            Operator::Div |
            Operator::Equal |
            Operator::LessOrEqual |
            Operator::Mul |
            Operator::Pow |
            Operator::Sub) => {
                let [a, b] = inputs else {
                    panic!("Expected 2 inputs for binary operator")
                };
                self.binary_op(binop, *a, *b)
            }
            _ => unimplemented!(),
        }
    }

    fn build_body(&mut self) -> Result<(String, KernelDecl), BuildError> {
        let mut stmts = Vec::new();
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::ElementWises(ElementWises { ops }) = &kernel.body else {
            unreachable!()
        };

        assert!(kernel.outputs.len() == 1);
        let storage_dt = self
            .ctx
            .get_resolved_tensor_type(kernel.outputs[0])?
            .elem_type;
        let compute_dt = compute_dtype(storage_dt);
        let output: KernelVar = {
            let mut outputs = Vec::new();
            for (op, args) in ops.iter() {
                let mut inputs = Vec::with_capacity(args.len());
                for input in args.iter() {
                    match input {
                        ElementwiseOpArg::Input(i) => {
                            inputs.push(KernelVar::Value(kernel.inputs[*i].unwrap()))
                        }
                        ElementwiseOpArg::NthResult(i) => inputs.push(outputs[*i]),
                    }
                }
                let var = self.ctx.new_local_var();
                let ty = TypeSymbol::Primitive(compute_dt);
                let rhs = self.single_op(op, &inputs);
                stmts.push(format!("{ty} {var} = {rhs};"));
                outputs.push(var)
            }
            outputs.pop().unwrap()
        };
        stmts.push(format!("return {};", output));

        let device_decl = {
            let mut res = self.ctx.decl.clone();
            assert!(matches!(res.qualifier, FuncQualifier::Global));
            res.qualifier = FuncQualifier::Device(compute_dt);
            res.params = res
                .params
                .iter()
                .skip(1)
                .map(|(var, ty)| {
                    assert!(matches!(var, KernelVar::Value(_)));
                    let TypeSymbol::Pointer(inner) = ty else {
                        panic!();
                    };
                    let TypeSymbol::Primitive(dt) = **inner else {
                        panic!();
                    };
                    (*var, TypeSymbol::Primitive(compute_dtype(dt)))
                })
                .collect_vec();
            res
        };
        Ok((stmts.iter().map(|x| x.to_string()).join("\n"), device_decl))
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let (body, device_decl) = self.build_body()?;
        let output = self.ctx.schedule.kernels[self.ctx.decl.kernel_id].outputs[0];
        let output_ty = self.ctx.get_resolved_tensor_type(output)?;
        let size = output_ty.dims.size();
        let gid = KernelVar::Gid;
        let decl = self.ctx.decl.decl();
        let device_decl_name = device_decl.name();
        let storage_dt = output_ty.elem_type;
        let compute_dt = compute_dtype(storage_dt);
        let out_cast = if storage_dt != compute_dt {
            format!("({})", storage_dt)
        } else {
            String::new()
        };
        let args = device_decl
            .params
            .iter()
            .map(|(var, _)| {
                let KernelVar::Value(value_id) = var else {
                    panic!();
                };
                let in_storage = self.ctx.get_resolved_tensor_type(*value_id)?.elem_type;
                let in_compute = compute_dtype(in_storage);
                let in_cast = if in_storage != in_compute {
                    format!("({})", in_compute)
                } else {
                    String::new()
                };
                let idx = self
                    .ctx
                    .tensor_idx(*value_id, KernelVar::Gid, Some(&output_ty.dims))?;
                Ok(format!("{in_cast}{}[{}]", var, idx))
            })
            .collect::<Result<Vec<_>, BuildError>>()?
            .join(", ");
        let out = {
            let idx = self.ctx.tensor_idx(output, KernelVar::Gid, None)?;
            format!("{}[{}]", KernelVar::Value(output), idx)
        };
        let device_decl = device_decl.decl();
        Ok(format!(
            "
{device_decl} {{
    {body}
}};
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {out} = {out_cast}{device_decl_name}({args});
}}
"
        ))
    }
}

fn compute_dtype(storage: DataType) -> DataType {
    match storage {
        DataType::Float(FloatType::BF16) => DataType::Float(FloatType::F32),
        other => other,
    }
}

pub struct SplitBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

fn select_rec<T: Display>(sizes: &[usize], pivot: &T) -> String {
    fn select_rec_impl<T: Display>(
        sizes: &[usize],
        pivot: &T,
        acc_sz: usize,
        depth: usize,
    ) -> String {
        if sizes.len() == 1 {
            return depth.to_string();
        }

        let size = sizes[0];
        let sizes = &sizes[1..];
        let next_acc_sz = acc_sz + size;
        let next_select = select_rec_impl(sizes, pivot, next_acc_sz, depth + 1);
        format!("({pivot} < {next_acc_sz} ? {depth} : {next_select})")
    }

    select_rec_impl(sizes, pivot, 0, 0)
}

fn acc_sizes(sizes: &[usize]) -> Vec<usize> {
    let mut vec = Vec::with_capacity(sizes.len());
    let mut acc = 0;
    for s in sizes.iter() {
        vec.push(acc);
        acc += *s;
    }
    vec
}

impl<'sched> SplitBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::Opaque(Opaque {
            op: Operator::Split(split),
        }) = &kernel.body
        else {
            panic!("Expected Split operator");
        };

        let axis_idx_var = self.ctx.new_local_var();

        let input_id = kernel.inputs[0].unwrap();
        let input_ty = self.ctx.get_resolved_tensor_type(input_id)?;
        // if !input_ty.is_contiguous() {
        //     return Err(BuildError::NonContiguousTensor(input_id));
        // }
        let value_ty: TypeSymbol = input_ty.elem_type.into();
        let ptr_ty = value_ty.to_pointer();
        let gid = KernelVar::Gid;
        let size = input_ty.dims.size();
        let axis = split.axis.index(input_ty.dims.ndim());
        let axis_stride = input_ty.strides()[axis];
        let axis_dim = input_ty.dims[axis];
        let (outs, sizes): (Vec<_>, Vec<_>) = kernel
            .outputs
            .iter()
            .map(|id| {
                let output_ty = self.ctx.get_resolved_tensor_type(*id)?;
                let sz = output_ty.dims[axis];
                Ok((*id, sz))
            })
            .collect::<Result<Vec<_>, BuildError>>()?
            .into_iter()
            .unzip();
        let in_ = KernelVar::Value(input_id);
        let select = select_rec(&sizes[..], &axis_idx_var);
        let sizes_acc = acc_sizes(&sizes[..]);
        let outs = outs
            .into_iter()
            .map(|id| format!("{}", KernelVar::Value(id)))
            .collect::<Vec<_>>()
            .join(", ");
        let sizes = sizes
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let sizes_acc = sizes_acc
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let indexes = izip!(input_ty.dims.iter(), input_ty.strides().iter())
            .map(|(dim, stride)| {
                let stride = stride.max(&1);
                format!("({gid} / {stride}) % {dim}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let output_dims_init = input_ty
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let ndim = input_ty.dims.ndim();
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {value_ty} load = {in_}[{gid}];
    int {axis_idx_var} = ({gid} / {axis_stride}) % {axis_dim};
    int select = {select};
    int indexes[] = {{{indexes}}};
    int sizes[] = {{{sizes}}};
    int sizes_acc[] = {{{sizes_acc}}};
    indexes[{axis}] = {axis_idx_var} - sizes_acc[select];
    int output_dims[] = {{{output_dims_init}}};
    output_dims[{axis}] = sizes[select];
    int out_offset = 0;
    for (int i = 0; i < {ndim}; i++) {{
        out_offset *= output_dims[i];
        out_offset += indexes[i];
    }}
    {ptr_ty} outs[] = {{{outs}}};
    outs[select][out_offset] = load;\n
}}
"
        ))
    }
}

pub struct ConcatKernel {
    pub data_ty: DataType,
    pub ndim: usize,
    pub n_inputs: usize,
    pub axis: usize,
    pub total_size: usize,

    pub out: Expr,
    pub ins: Vec<Expr>,
    pub in_sizes_acc: Vec<usize>,
    pub in_axis_sizes_acc: Vec<usize>,
    pub input_dims: Vec<Vec<usize>>,
    pub input_strides: Vec<Vec<usize>>,
    pub output_dims: Vec<usize>,
}

impl ConcatKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "concat_kernel<{}, {}, {}>",
            self.data_ty, self.ndim, self.n_inputs
        );
        let ty = self.data_ty;
        let join_usize = |v: &[usize]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let join_2d = |v: &[Vec<usize>]| {
            v.iter()
                .map(|row| format!("{{{}}}", join_usize(row)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let ins_list = self
            .ins
            .iter()
            .map(|e| format!("(const {} *)({})", ty, e))
            .collect::<Vec<_>>()
            .join(", ");
        let cfg = format!(
            "ConcatInputs<{}, {}, {}>{{{{ {} }}, {{ {} }}, {{ {} }}, {{ {} }}, {{ {} }}, {{ {} }}}}",
            ty,
            self.ndim,
            self.n_inputs,
            ins_list,
            join_usize(&self.in_sizes_acc),
            join_usize(&self.in_axis_sizes_acc),
            join_2d(&self.input_dims),
            join_2d(&self.input_strides),
            join_usize(&self.output_dims),
        );
        let args = vec![
            cast!(ty, self.out),
            cfg,
            self.axis.to_string(),
            self.total_size.to_string(),
        ];
        (id, args)
    }
}

pub struct ContiguousBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> ContiguousBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let ops = match &kernel.body {
            KernelBody::Opaque(Opaque {
                op: Operator::Contiguous(operator::Contiguous { ops }),
            }) => ops,
            _ => panic!("expected Contiguous"),
        };

        let gid = KernelVar::Gid;
        let input = kernel.inputs[0].unwrap();
        let output = kernel.outputs[0];
        let out_ty = self.ctx.get_resolved_tensor_type(output)?;
        let in_ty = self.ctx.get_resolved_tensor_type(input)?;
        let size = out_ty.dims.size().max(1);
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let decl = self.ctx.decl.decl();

        if ops.is_empty() {
            let input_idx = self.ctx.tensor_idx(input, KernelVar::Gid, None)?;
            return Ok(format!(
                "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {out}[{gid}] = {in_}[{input_idx}];
}}
"
            ));
        }

        let mut body = String::new();

        for (i, d) in out_ty.dims.iter().enumerate().rev() {
            body.push_str(&format!("int idx_{i}=offset%{d};\n"));
            body.push_str(&format!("offset=offset/{d};\n"));
        }
        body.push_str("offset=0;\n");
        for (i, d) in out_ty.dims.iter().enumerate() {
            body.push_str(&format!("offset=offset*{d}+idx_{i};\n"));
        }

        let mut shape: Vec<usize> = out_ty.dims.iter().copied().collect();
        for op in ops.iter().rev() {
            match op {
                ReinterpretType::Reshape { before, .. } => {
                    shape = before.clone();
                }
                ReinterpretType::Transpose(transpose) => {
                    let cur_ndim = shape.len();
                    let perm = transpose
                        .perm
                        .clone()
                        .unwrap_or_else(|| (0..cur_ndim).collect());

                    for (i, d) in shape.iter().enumerate().rev() {
                        body.push_str(&format!("int t_{i}=offset%{d};\n"));
                        body.push_str(&format!("offset=offset/{d};\n"));
                    }

                    let old_shape = shape.clone();
                    let mut inv_perm = vec![0; cur_ndim];
                    for (i, &p) in perm.iter().enumerate() {
                        shape[p] = old_shape[i];
                        inv_perm[p] = i;
                    }

                    body.push_str("offset=0;\n");
                    for (k, d) in shape.iter().enumerate() {
                        body.push_str(&format!("offset=offset*{d}+t_{};\n", inv_perm[k]));
                    }
                }
                ReinterpretType::Broadcast { before, .. } => {
                    for (i, d) in shape.iter().enumerate().rev() {
                        body.push_str(&format!("int t_{i}=offset%{d};\n"));
                        body.push_str(&format!("offset=offset/{d};\n"));
                    }

                    shape = before.clone();
                    body.push_str("offset=0;\n");
                    for (i, d) in before.iter().enumerate() {
                        if *d == 1 {
                            body.push_str(&format!("offset=offset*{d};\n"));
                        } else {
                            body.push_str(&format!("offset=offset*{d}+t_{i};\n"));
                        }
                    }
                }
            }
        }

        let src_strides: Vec<usize> = in_ty.strides().iter().copied().collect();
        for (i, d) in shape.iter().enumerate().rev() {
            body.push_str(&format!("int s_{i}=offset%{d};\n"));
            body.push_str(&format!("offset=offset/{d};\n"));
        }
        body.push_str("int src_idx=0;\n");
        for (i, s) in src_strides.iter().enumerate() {
            body.push_str(&format!("src_idx+=s_{i}*{s};\n"));
        }

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    int offset = {gid};
{body}    
    {out}[{gid}] = {in_}[src_idx];
}}
"
        ))
    }
}

pub struct ResizeBuilder<'sched> {
    ctx: BuilderContext<'sched>,
    stmts: Vec<String>,
}

impl<'sched> ResizeBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
            stmts: Vec::new(),
        }
    }

    fn output_dims(&self) -> Result<Vec<String>, BuildError> {
        let output_ty = self.ctx.schedule.kernels[self.ctx.decl.kernel_id].outputs[0];
        let output_ty = self.ctx.get_resolved_tensor_type(output_ty)?;

        let mut res = Vec::with_capacity(output_ty.dims.ndim());
        let mut acc = 1;
        let gid = KernelVar::Gid;
        for dim in output_ty.dims.iter().rev() {
            res.push(format!("{gid} / {acc} % {dim}"));
            acc *= *dim;
        }

        res.reverse();
        Ok(res)
    }

    fn resize_axis(
        &mut self,
        axis: usize,
        i_dim: usize,
        x_resized: KernelVar,
        resize: &operator::Resize,
    ) -> Result<KernelVar, BuildError> {
        let x_original_var = self.ctx.new_local_var();
        let scale = match resize.scale {
            Some(ref scale) => match scale {
                operator::ResizeScale::Scales(ref scales) => scales[axis] as f32,
                operator::ResizeScale::Sizes(ref sizes) => sizes[axis] as f32 / i_dim as f32,
            },
            None => unreachable!(),
        };
        let x_original = format!("({x_resized} + 0.5) / {scale} - 0.5");
        self.stmts
            .push(format!("float {x_original_var} = {x_original};"));

        let x_original = match resize.mode {
            operator::ResizeMode::Nearest(nearest) => match nearest {
                operator::ResizeNearestMode::RoundPreferFloor => {
                    format!("ceil({x_original_var} - 0.5)")
                }
                operator::ResizeNearestMode::RoundPreferCeil => {
                    format!("floor({x_original_var} + 0.5)")
                }
                operator::ResizeNearestMode::Floor => format!("floor({x_original_var})"),
                operator::ResizeNearestMode::Ceil => format!("ceil({x_original_var})"),
            },
        };
        let x_original_var = self.ctx.new_local_var();
        self.stmts.push(format!(
            "int {x_original_var} = max(0, min((int){x_original}, {v}));",
            v = i_dim - 1
        ));
        Ok(x_original_var)
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let output = kernel.outputs[0];
        let input = kernel.inputs[0].unwrap();
        let KernelBody::Opaque(Opaque {
            op: Operator::Resize(resize),
        }) = &kernel.body
        else {
            panic!("Expected Resize operator");
        };

        let x_resized_vec_var = self.ctx.new_local_var();
        let output_dims = self.output_dims()?.join(", ");
        self.stmts
            .push(format!("int {x_resized_vec_var}[] = {{{output_dims}}};"));

        let output_ty = self.ctx.get_resolved_tensor_type(output)?.clone();
        let input_ty = self.ctx.get_resolved_tensor_type(input)?.clone();
        let axes = match &resize.axes {
            Some(axes) => axes
                .iter()
                .map(|a| a.index(output_ty.dims.ndim()))
                .collect::<Vec<_>>(),
            None => (0..output_ty.dims.ndim()).collect(),
        };
        for (i, axis) in axes.iter().copied().enumerate() {
            let x_resized_var = self.ctx.new_local_var();
            self.stmts.push(format!(
                "float {x_resized_var} = (float){x_resized_vec_var}[{axis}];"
            ));
            let x_original_var = self.resize_axis(i, input_ty.dims[axis], x_resized_var, resize)?;
            self.stmts
                .push(format!("{x_resized_vec_var}[{axis}] = {x_original_var};"));
        }

        let in_offset_var = self.ctx.new_local_var();
        let in_offset = input_ty
            .strides()
            .iter()
            .enumerate()
            .map(|(i, stride)| format!("{x_resized_vec_var}[{i}] * {stride}"))
            .collect::<Vec<_>>()
            .join(" + ");
        self.stmts
            .push(format!("int {in_offset_var} = {in_offset};"));

        let gid = KernelVar::Gid;
        let size = output_ty.dims.size();
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let body = self.stmts.join("\n");
        let decl = self.ctx.decl.decl();
        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {body}
    {out}[{gid}] = {in_}[{in_offset_var}];
}}
"
        ))
    }
}

// Only supports:
//   - on_value and off_value are constants.
//   - axis is the last dimension.
pub struct OneHotBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> OneHotBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::Opaque(Opaque {
            op: Operator::OneHot(one_hot),
        }) = &kernel.body
        else {
            panic!("Expected Resize operator");
        };

        let Some(depth) = one_hot.depth else {
            unimplemented!()
        };
        let Some(on_value) = one_hot.on_value else {
            unimplemented!()
        };
        let Some(off_value) = one_hot.off_value else {
            unimplemented!()
        };

        let indexes = kernel.inputs[args::ONEHOT_INDICES].unwrap();
        let size = self.ctx.get_resolved_tensor_type(indexes)?.dims.size();
        let decl = self.ctx.decl.decl();
        let gid = KernelVar::Gid;
        let out = KernelVar::Value(kernel.outputs[0]);
        let indexes = KernelVar::Value(indexes);

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int index = {indexes}[{gid}];
    if (index < 0) index += {depth};
    for (int i = 0; i < {depth}; i++) {{
        {out}[{gid} * {depth} + i] = (i == index) ? {on_value} : {off_value};
    }}
}}
"
        ))
    }
}

pub struct DequantizeLinearKernel {
    pub in_ty: DataType,
    pub out_ty: DataType,
    pub axis_dim: usize,
    pub inner_size: usize,
    pub total: usize,

    pub out: Expr,
    pub x: Expr,
    pub scale: Expr,
}

impl DequantizeLinearKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("dequantize_linear<{}, {}>", self.in_ty, self.out_ty);
        let args = vec![
            cast!(self.out_ty, self.out),
            cast!(self.in_ty, self.x),
            cast!(self.out_ty, self.scale),
            self.axis_dim.to_string(),
            self.inner_size.to_string(),
            self.total.to_string(),
        ];
        (id, args)
    }
}

pub struct DynamicQuantizeLinearKernel {
    pub float_ty: DataType,
    pub q_ty: DataType,
    pub symmetric: bool,
    pub axis_dim: usize,
    pub inner_size: usize,

    pub y: Expr,
    pub y_scale: Expr,
    pub y_zero_point: Expr,
    pub x: Expr,
}

impl DynamicQuantizeLinearKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "dynamic_quantize_linear_kernel<{}, {}, 256, {}>",
            self.float_ty,
            self.q_ty,
            if self.symmetric { "true" } else { "false" },
        );
        let args = vec![
            cast!(self.q_ty, self.y),
            cast!(self.float_ty, self.y_scale),
            cast!(self.q_ty, self.y_zero_point),
            cast!(self.float_ty, self.x),
            self.axis_dim.to_string(),
            self.inner_size.to_string(),
        ];
        (id, args)
    }
}

pub struct QuantizedGemvInt8Kernel {
    pub n: usize,
    pub k: usize,

    pub out: Expr,
    pub lhs: Expr,
    pub lhs_scale: Expr,
    pub rhs: Expr,
    pub rhs_scale: Expr,
}

impl QuantizedGemvInt8Kernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = "quantized_gemv_int8".to_string();
        let args = vec![
            format!("(__nv_bfloat16 *)({})", self.out),
            format!("(const int8_t *)({})", self.lhs),
            format!("(const __nv_bfloat16 *)({})", self.lhs_scale),
            format!("(const int8_t *)({})", self.rhs),
            format!("(const __nv_bfloat16 *)({})", self.rhs_scale),
            self.n.to_string(),
            self.k.to_string(),
        ];
        (id, args)
    }
}

pub struct QuantizedMatMulInt8Kernel {
    pub bm: usize,
    pub bn: usize,
    pub warp_tile_m: usize,
    pub warp_tile_n: usize,
    pub stages: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub out: Expr,
    pub lhs: Expr,
    pub lhs_scale: Expr,
    pub rhs: Expr,
    pub rhs_scale: Expr,
}

impl QuantizedMatMulInt8Kernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "quantized_matmul_int8<{}, {}, {}, {}, {}>",
            self.bm, self.bn, self.warp_tile_m, self.warp_tile_n, self.stages,
        );
        let args = vec![
            format!("(__nv_bfloat16 *)({})", self.out),
            format!("(const int8_t *)({})", self.lhs),
            format!("(const __nv_bfloat16 *)({})", self.lhs_scale),
            format!("(const int8_t *)({})", self.rhs),
            format!("(const __nv_bfloat16 *)({})", self.rhs_scale),
            self.m.to_string(),
            self.n.to_string(),
            self.k.to_string(),
        ];
        (id, args)
    }
}

pub struct DequantMatMulKernel {
    pub act_ty: DataType,
    pub out_ty: DataType,
    pub block_size: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub out: Expr,
    pub act: Expr,
    pub wq: Expr,
    pub scale: Expr,
}

impl DequantMatMulKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "dequant_matmul<{}, {}, {}>",
            self.act_ty, self.out_ty, self.block_size
        );
        let args = vec![
            cast!(self.out_ty, self.out),
            cast!(self.act_ty, self.act),
            format!("(const signed char *)({})", self.wq),
            cast!(self.out_ty, self.scale),
            self.m.to_string(),
            self.n.to_string(),
            self.k.to_string(),
        ];
        (id, args)
    }
}

pub struct DequantGemvKernel {
    pub act_ty: DataType,
    pub out_ty: DataType,
    pub n: usize,
    pub k: usize,

    pub out: Expr,
    pub act: Expr,
    pub wq: Expr,
    pub scale: Expr,
}

impl DequantGemvKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("dequant_gemv<{}, {}>", self.act_ty, self.out_ty);
        let args = vec![
            cast!(self.out_ty, self.out),
            cast!(self.act_ty, self.act),
            format!("(const signed char *)({})", self.wq),
            cast!(self.out_ty, self.scale),
            self.n.to_string(),
            self.k.to_string(),
        ];
        (id, args)
    }
}

pub struct DequantMatMulWmmaKernel {
    pub bm: usize,
    pub bn: usize,
    pub warp_tile_m: usize,
    pub warp_tile_n: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,

    pub out: Expr,
    pub act: Expr,
    pub wq: Expr,
    pub scale: Expr,
}

impl DequantMatMulWmmaKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "dequant_matmul_wmma<{}, {}, {}, {}>",
            self.bm, self.bn, self.warp_tile_m, self.warp_tile_n,
        );
        let args = vec![
            format!("(__nv_bfloat16 *)({})", self.out),
            format!("(const __nv_bfloat16 *)({})", self.act),
            format!("(const int8_t *)({})", self.wq),
            format!("(const __nv_bfloat16 *)({})", self.scale),
            self.m.to_string(),
            self.n.to_string(),
            self.k.to_string(),
        ];
        (id, args)
    }
}

pub struct GatherKernel {
    pub data_ty: DataType,
    pub idx_ty: DataType,
    pub axis_dim: usize,
    pub repeat: usize,
    pub size: usize,

    pub out: Expr,
    pub in_: Expr,
    pub indices: Expr,
}

impl GatherKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("gather_axis0_kernel<{}, {}>", self.data_ty, self.idx_ty);
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            cast!(self.idx_ty, self.indices),
            self.axis_dim.to_string(),
            self.repeat.to_string(),
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct CopyBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> CopyBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        assert!(matches_opaque!(
            kernel,
            Operator::Identity | Operator::Reinterpret(_)
        ));

        let gid = KernelVar::Gid;
        let input = kernel.inputs[0].unwrap();
        let output = kernel.outputs[0];
        let size = self.ctx.get_resolved_tensor_type(input)?.dims.size();
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {out}[{gid}] = {in_}[{gid}];
}}
"
        ))
    }
}

pub struct PoolBuilder<'sched> {
    ctx: BuilderContext<'sched>,
    is_max: bool,
}

impl<'sched> PoolBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        let kernel = &schedule.kernels[decl.kernel_id];
        let is_max = matches!(
            &kernel.body,
            KernelBody::Opaque(Opaque {
                op: Operator::MaxPool(_)
            })
        );
        Self {
            ctx: BuilderContext::new(schedule, decl),
            is_max,
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let pool = match &kernel.body {
            KernelBody::Opaque(Opaque {
                op: Operator::MaxPool(pool),
            }) => pool,
            KernelBody::Opaque(Opaque {
                op: Operator::AveragePool(pool),
            }) => pool,
            _ => panic!("Expected MaxPool or AveragePool operator"),
        };

        if pool.kernel_shape.ndim() != 2 {
            unimplemented!("Only 2D pooling is supported");
        }

        match pool.layout {
            operator::Layout::NCHW => self.build_nchw(pool),
            operator::Layout::NHWC => self.build_nhwc(pool),
        }
    }

    fn build_nchw(&mut self, pool: &operator::Pooling) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        assert!(pool.layout == operator::Layout::NCHW);

        let input = kernel.inputs[0].unwrap();
        let output = kernel.outputs[0];
        let input_ty = self.ctx.get_resolved_tensor_type(input)?;
        let output_ty = self.ctx.get_resolved_tensor_type(output)?;
        assert!(kernel.inputs.len() == 1);
        assert!(kernel.outputs.len() == 1);
        assert!(input_ty.dims.ndim() == 4 && output_ty.dims.ndim() == 4);
        assert!(input_ty.is_contiguous() && output_ty.is_contiguous());
        assert!(input_ty.dims[0] == output_ty.dims[0]);
        assert!(input_ty.dims[1] == output_ty.dims[1]);
        let nbatch = input_ty.dims[0];
        let channels = input_ty.dims[1];
        let height = input_ty.dims[2];
        let width = input_ty.dims[3];
        let o_height = output_ty.dims[2];
        let o_width = output_ty.dims[3];
        let kernel_h = pool.kernel_shape[0];
        let kernel_w = pool.kernel_shape[1];
        let stride_h = pool.strides[0];
        let stride_w = pool.strides[1];
        let (pad_h, pad_w) = match pool.pad {
            ConvPad::NotSet(ref pad) => (pad[0].0, pad[1].0),
            _ => unimplemented!("Padding type not implemented"),
        };

        let size = nbatch * channels * o_height * o_width;
        let gid = KernelVar::Gid;
        let ty = TypeSymbol::Primitive(input_ty.elem_type);
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let decl = self.ctx.decl.decl();

        if self.is_max {
            Ok(format!(
                "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int o_w_idx = {gid} % {o_width};
    int o_h_idx = ({gid} / {o_width}) % {o_height};
    int o_c_idx = ({gid} / ({o_width} * {o_height})) % {channels};
    int o_b_idx = {gid} / ({o_width} * {o_height} * {channels});

    int i_w_begin = o_w_idx * {stride_w} - {pad_w};
    int i_h_begin = o_h_idx * {stride_h} - {pad_h};
    int i_w_end = i_w_begin + {kernel_w};
    int i_h_end = i_h_begin + {kernel_h};

    {ty} max_val = std::numeric_limits<{ty}>::min();
    for (int h = i_h_begin; h < i_h_end; h++) {{
        for (int w = i_w_begin; w < i_w_end; w++) {{
            if (0 <= h && h < {height} && 0 <= w && w < {width}) {{
                int in_idx = o_b_idx * {channels};
                in_idx += o_c_idx;
                in_idx *= {height};
                in_idx += h;
                in_idx *= {width};
                in_idx += w;
                max_val = max(max_val, {in_}[in_idx]);
            }}
        }}
    }}

    {out}[{gid}] = max_val;
}}
"
            ))
        } else {
            Ok(format!(
                "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int o_w_idx = {gid} % {o_width};
    int o_h_idx = ({gid} / {o_width}) % {o_height};
    int o_c_idx = ({gid} / ({o_width} * {o_height})) % {channels};
    int o_b_idx = {gid} / ({o_width} * {o_height} * {channels});

    int i_w_begin = o_w_idx * {stride_w} - {pad_w};
    int i_h_begin = o_h_idx * {stride_h} - {pad_h};
    int i_w_end = i_w_begin + {kernel_w};
    int i_h_end = i_h_begin + {kernel_h};

    {ty} sum_val = 0;
    int cnt = 0;
    for (int h = i_h_begin; h < i_h_end; h++) {{
        for (int w = i_w_begin; w < i_w_end; w++) {{
            if (0 <= h && h < {height} && 0 <= w && w < {width}) {{
                int in_idx = o_b_idx * {channels};
                in_idx += o_c_idx;
                in_idx *= {height};
                in_idx += h;
                in_idx *= {width};
                in_idx += w;
                sum_val += {in_}[in_idx];
                cnt++;
            }}
        }}
    }}

    {out}[{gid}] = sum_val / ({ty})cnt;
}}
"
            ))
        }
    }

    fn build_nhwc(&mut self, pool: &operator::Pooling) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        assert!(pool.layout == operator::Layout::NHWC);

        let input = kernel.inputs[0].unwrap();
        let output = kernel.outputs[0];
        let input_ty = self.ctx.get_resolved_tensor_type(input)?;
        let output_ty = self.ctx.get_resolved_tensor_type(output)?;
        assert!(kernel.inputs.len() == 1);
        assert!(kernel.outputs.len() == 1);
        assert!(input_ty.dims.ndim() == 4 && output_ty.dims.ndim() == 4);
        assert!(input_ty.is_contiguous() && output_ty.is_contiguous());
        let nbatch = input_ty.dims[0];
        let height = input_ty.dims[1];
        let width = input_ty.dims[2];
        let channels = input_ty.dims[3];
        let o_height = output_ty.dims[1];
        let o_width = output_ty.dims[2];
        assert!(channels == output_ty.dims[3]);
        let kernel_h = pool.kernel_shape[0];
        let kernel_w = pool.kernel_shape[1];
        let stride_h = pool.strides[0];
        let stride_w = pool.strides[1];
        let (pad_h, pad_w) = match pool.pad {
            ConvPad::NotSet(ref pad) => (pad[0].0, pad[1].0),
            _ => unimplemented!("Padding type not implemented"),
        };

        let size = nbatch * o_height * o_width * channels;
        let gid = KernelVar::Gid;
        let ty = TypeSymbol::Primitive(input_ty.elem_type);
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let decl = self.ctx.decl.decl();

        if self.is_max {
            Ok(format!(
                "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int o_c_idx = {gid} % {channels};
    int o_w_idx = ({gid} / {channels}) % {o_width};
    int o_h_idx = ({gid} / ({channels} * {o_width})) % {o_height};
    int o_b_idx = {gid} / ({channels} * {o_width} * {o_height});

    int i_w_begin = o_w_idx * {stride_w} - {pad_w};
    int i_h_begin = o_h_idx * {stride_h} - {pad_h};
    int i_w_end = i_w_begin + {kernel_w};
    int i_h_end = i_h_begin + {kernel_h};

    {ty} max_val = std::numeric_limits<{ty}>::min();
    for (int h = i_h_begin; h < i_h_end; h++) {{
        for (int w = i_w_begin; w < i_w_end; w++) {{
            if (0 <= h && h < {height} && 0 <= w && w < {width}) {{
                int in_idx = o_b_idx * ({height} * {width} * {channels})
                           + h * ({width} * {channels})
                           + w * {channels}
                           + o_c_idx;
                max_val = max(max_val, {in_}[in_idx]);
            }}
        }}
    }}

    {out}[{gid}] = max_val;
}}
"
            ))
        } else {
            Ok(format!(
                "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int o_c_idx = {gid} % {channels};
    int o_w_idx = ({gid} / {channels}) % {o_width};
    int o_h_idx = ({gid} / ({channels} * {o_width})) % {o_height};
    int o_b_idx = {gid} / ({channels} * {o_width} * {o_height});

    int i_w_begin = o_w_idx * {stride_w} - {pad_w};
    int i_h_begin = o_h_idx * {stride_h} - {pad_h};
    int i_w_end = i_w_begin + {kernel_w};
    int i_h_end = i_h_begin + {kernel_h};

    {ty} sum_val = 0;
    int cnt = 0;
    for (int h = i_h_begin; h < i_h_end; h++) {{
        for (int w = i_w_begin; w < i_w_end; w++) {{
            if (0 <= h && h < {height} && 0 <= w && w < {width}) {{
                int in_idx = o_b_idx * ({height} * {width} * {channels})
                           + h * ({width} * {channels})
                           + w * {channels}
                           + o_c_idx;
                sum_val += {in_}[in_idx];
                cnt++;
            }}
        }}
    }}

    {out}[{gid}] = sum_val / ({ty})cnt;
}}
"
            ))
        }
    }
}

pub struct WhereBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> WhereBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::Opaque(Opaque {
            op: Operator::Where,
        }) = &kernel.body
        else {
            panic!("Expected Where operator");
        };

        let cond_id = kernel.inputs[args::WHERE_COND].unwrap();
        let x_id = kernel.inputs[args::WHERE_X].unwrap();
        let y_id = kernel.inputs[args::WHERE_Y].unwrap();
        let output_id = kernel.outputs[0];

        let output_ty = self.ctx.get_resolved_tensor_type(output_id)?;
        let cond_ty = self.ctx.get_resolved_tensor_type(cond_id)?;
        let x_ty = self.ctx.get_resolved_tensor_type(x_id)?;
        let y_ty = self.ctx.get_resolved_tensor_type(y_id)?;

        let size = output_ty.dims.size();
        let ndim = output_ty.dims.ndim();
        let value_ty: TypeSymbol = output_ty.elem_type.into();

        let gid = KernelVar::Gid;
        let out = KernelVar::Value(output_id);
        let cond = KernelVar::Value(cond_id);
        let x = KernelVar::Value(x_id);
        let y = KernelVar::Value(y_id);
        let decl = self.ctx.decl.decl();

        let cond_bc = cond_ty.broadcast(&output_ty.dims);
        let x_bc = x_ty.broadcast(&output_ty.dims);
        let y_bc = y_ty.broadcast(&output_ty.dims);

        let gen_index = |ty: &ResolvedTensorType, var_name: &str| -> String {
            let mut lines = Vec::new();
            lines.push(format!("int {var_name} = 0;"));
            lines.push("{ int rem = gid;".to_string());
            for i in (0..ndim).rev() {
                let out_dim = output_ty.dims[i];
                let stride = ty.stride(i);
                lines.push(format!("int idx_{i} = rem % {out_dim};"));
                lines.push(format!("rem = rem / {out_dim};"));
                if stride != 0 {
                    lines.push(format!("{var_name} += idx_{i} * {stride};"));
                }
            }
            lines.push("}".to_string());
            lines.join("\n    ")
        };

        let cond_idx = gen_index(&cond_bc, "cond_idx");
        let x_idx = gen_index(&x_bc, "x_idx");
        let y_idx = gen_index(&y_bc, "y_idx");

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {cond_idx}
    {x_idx}
    {y_idx}
    {value_ty} result = ({cond}[cond_idx] != 0) ? {x}[x_idx] : {y}[y_idx];
    {out}[{gid}] = result;
}}
"
        ))
    }
}

pub struct ExpandKernel {
    pub data_ty: DataType,
    pub ndim: usize,
    pub size: usize,
    pub output_dims: Vec<usize>,
    pub input_strides: Vec<usize>,

    pub out: Expr,
    pub in_: Expr,
}

impl ExpandKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("expand_kernel<{}, {}>", self.data_ty, self.ndim);
        let join_usize = |v: &[usize]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let cfg = format!(
            "ExpandConfig<{}>{{{{ {} }}, {{ {} }}}}",
            self.ndim,
            join_usize(&self.output_dims),
            join_usize(&self.input_strides),
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            cfg,
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct SliceKernel {
    pub data_ty: DataType,
    pub ndim: usize,
    pub size: usize,
    pub base_offset: isize,
    pub output_dims: Vec<usize>,
    pub input_strides: Vec<usize>,

    pub out: Expr,
    pub in_: Expr,
}

impl SliceKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("slice_kernel<{}, {}>", self.data_ty, self.ndim);
        let join_usize = |v: &[usize]| {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let cfg = format!(
            "SliceConfig<{}>{{{{ {} }}, {{ {} }}, {}}}",
            self.ndim,
            join_usize(&self.output_dims),
            join_usize(&self.input_strides),
            self.base_offset,
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            cfg,
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct ReduceMatrixKernel {
    pub data_ty: DataType,
    pub op: operator::ReduceOp,
    pub block_size: usize,
    pub col: usize,

    pub out: Expr,
    pub in_: Expr,
}

impl ReduceMatrixKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let ty = self.data_ty;
        let op_tag = match self.op {
            operator::ReduceOp::Max => format!("ReduceOpMax<{}>", ty),
            operator::ReduceOp::Mean => format!("ReduceOpMean<{}>", ty),
            operator::ReduceOp::Sum => format!("ReduceOpSum<{}>", ty),
            operator::ReduceOp::Variance => unimplemented!(),
        };
        let id = format!(
            "reduce_matrix_kernel<{}, {}, {}>",
            ty, op_tag, self.block_size
        );
        let args = vec![
            cast!(ty, self.out),
            cast!(ty, self.in_),
            self.col.to_string(),
        ];
        (id, args)
    }
}
