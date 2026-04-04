mod cublas;
mod cudnn;
mod kernel;
mod runtime_api;

use std::cmp::min;
use std::collections::BTreeSet;
use std::collections::HashMap;

use delegate::delegate;
use derive_more::From;
use indexmap::IndexMap;
use itertools::chain;
use itertools::Itertools;

use crate::codegen::cuda::cublas::*;
use crate::codegen::cuda::cudnn::*;
use crate::codegen::cuda::kernel::AttentionKernel;
use crate::codegen::cuda::kernel::ConcatBuilder;
use crate::codegen::cuda::kernel::ContiguousBuilder;
use crate::codegen::cuda::kernel::CopyBuilder;
use crate::codegen::cuda::kernel::ElementwiseKernelBuilder;
use crate::codegen::cuda::kernel::FuncQualifier;
use crate::codegen::cuda::kernel::GatherBuilder;
use crate::codegen::cuda::kernel::GeneratedKernel;
use crate::codegen::cuda::kernel::KernelDecl;
use crate::codegen::cuda::kernel::KernelVar;
use crate::codegen::cuda::kernel::MaxPoolBuilder;
use crate::codegen::cuda::kernel::OneHotBuilder;
use crate::codegen::cuda::kernel::ReduceMatrixBuilder;
use crate::codegen::cuda::kernel::ResizeBuilder;
use crate::codegen::cuda::kernel::SplitBuilder;
use crate::codegen::cuda::kernel::TypeSymbol;
use crate::codegen::cuda::runtime_api::*;
use crate::onnx::model::ValueId;
use crate::onnx::operator;
use crate::onnx::operator::*;
use crate::options::Options;
use crate::schedule::stream::EventId;
use crate::schedule::stream::KernelStreamAssignment;
use crate::schedule::stream::StreamAllocResult;
use crate::schedule::stream::StreamId;
use crate::schedule::*;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

#[derive(Debug)]
pub enum BuildError {
    UnresolvedType(ValueId),
    UnresolvedAllocateInfo(KernelId),
    UnexpectedMemAlloc(ValueId),
    ChunkNotFound(ValueId),
    NoHostVariable(ValueId),
    NoDeviceVariable(ChunkId),
    EventNotFound(ValueId),
    ActivationNotFound(KernelId),
    UnsupportedTensorDim(ValueId, usize),
    NonContiguousTensor(ValueId),
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DataType::SInt(SIntType::I32) => "i32",
                DataType::SInt(SIntType::I64) => "i64",
                DataType::UInt(UIntType::U64) => "u64",
                DataType::Float(FloatType::F32) => "float",
                DataType::Float(FloatType::F64) => "double",
            }
        )
    }
}

enum MemSize {
    Single(SingleMemSize),
    Raw(Expr),
}

impl std::fmt::Display for MemSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemSize::Single(size) => write!(f, "{}", size),
            MemSize::Raw(expr) => write!(f, "{}", expr),
        }
    }
}

#[derive(Clone, Copy)]
struct SizeOf(pub DataType);

impl std::fmt::Display for SizeOf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sizeof({})", self.0)
    }
}

#[derive(Clone, Copy, Default)]
struct SingleMemSize {
    ty: DataType,
    elem_num: usize,
}

impl std::fmt::Display for SingleMemSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.elem_num == 0 {
            write!(f, "0")
        } else {
            write!(f, "{} * {}", self.elem_num, SizeOf(self.ty))
        }
    }
}

impl From<&ResolvedTensorType> for SingleMemSize {
    fn from(ty: &ResolvedTensorType) -> Self {
        SingleMemSize {
            ty: ty.elem_type,
            elem_num: ty.storage_num_elements(),
        }
    }
}

impl From<&ResolvedTensorType> for MemSize {
    fn from(ty: &ResolvedTensorType) -> Self {
        MemSize::Single(SingleMemSize::from(ty))
    }
}

#[derive(Default, Clone)]
struct ChunkMemSize {
    sizes: Vec<SingleMemSize>,
}

impl ChunkMemSize {
    fn append(&mut self, size: SingleMemSize) {
        for ele in self.sizes.iter_mut() {
            if ele.ty == size.ty {
                ele.elem_num = ele.elem_num.max(size.elem_num);
                return;
            }
        }
        self.sizes.push(size);
    }

    fn max_byte_size(&self) -> usize {
        self.sizes
            .iter()
            .map(|s| {
                let elem_size = match s.ty {
                    DataType::Float(FloatType::F32) | DataType::SInt(SIntType::I32) => 4,
                    DataType::Float(FloatType::F64) |
                    DataType::SInt(SIntType::I64) |
                    DataType::UInt(UIntType::U64) => 8,
                };
                s.elem_num * elem_size
            })
            .max()
            .unwrap_or(0)
    }
}

impl std::fmt::Display for ChunkMemSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = self
            .sizes
            .iter()
            .filter(|s| 0 < s.elem_num)
            .map(|s| s.to_string())
            .join(", ");
        if res.is_empty() {
            write!(f, "0")
        } else {
            write!(f, "std::max({{ {} }})", res)
        }
    }
}

#[derive(Clone)]
enum Expr {
    Identifier(String),
    Literal(String),
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Identifier(name) => write!(f, "{}", name),
            Expr::Literal(lit) => write!(f, "{}", lit),
        }
    }
}

trait ToLiteral {
    fn to_literal(&self) -> Expr;
}

impl<T: ToString> ToLiteral for T {
    fn to_literal(&self) -> Expr {
        Expr::Literal(self.to_string())
    }
}

#[derive(From)]
enum Statement {
    LaunchKernel(kernel::LaunchKernel),
    CudaRuntimeApi(CudaRuntimeApi),
    CublasApi(CublasApi),
    CudnnApi(CudnnApi),
    Raw(String),
}

impl std::fmt::Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // TODO: Error handling for kernel launch
            Statement::LaunchKernel(kernel) => write!(f, "{};", kernel),
            Statement::CudaRuntimeApi(api) => write!(f, "cudaCheckErr({});", api),
            Statement::CublasApi(api) => write!(f, "cublasCheckErr({});", api),
            Statement::CudnnApi(api) => write!(f, "cudnnCheckErr({});", api),
            Statement::Raw(stmt) => write!(f, "{}", stmt),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum Include {
    System(&'static str),
    Local(&'static str),
}

pub struct HostCodeGenerator<'sched> {
    schedule: &'sched Schedule,

    stmts: Vec<Statement>,
    init_stmts: Vec<Statement>,
    destroy_stmts: Vec<Statement>,
    state_fields: Vec<String>,

    streams: &'sched HashMap<KernelId, KernelStreamAssignment>,
    to_record_events: BTreeSet<EventId>,

    value2chunk: HashMap<ValueId, ChunkId>,
    hostmem2identifier: HashMap<ValueId, String>,
    devicemem2identifier: Vec<String>,

    cublas_handlers: IndexMap<StreamId, CublasHandler>,
    cudnn_ctxs: IndexMap<StreamId, Vec<KernelId>>,

    separated_codes: Vec<SeparatedCode>,
    includes: BTreeSet<Include>,
}

pub struct HostCode {
    state_fields: Vec<String>,
    init_body: Vec<Statement>,
    destroy_body: Vec<Statement>,
    decl_values: Vec<Statement>,
    decl_cuda_objs: Vec<Statement>,
    computes: Vec<Statement>,
    finalize: Vec<Statement>,
    pub kernel_codes: Vec<SeparatedCode>,

    includes: BTreeSet<Include>,
    profile: bool,
}

const ARG_INPUT: &str = "input";
const ARG_OUTPUT: &str = "output";
const ARG_INITIALIZER: &str = "initializer";
const DEFAULT_BLOCK_SIZE: usize = 256;

struct CudnnCodeGenerator<'sched> {
    schedule: &'sched Schedule,
    kernel_id: KernelId,
}

pub enum SeparatedCode {
    Device(DeviceCode),
    Cudnn(CudnnCode),
}

impl SeparatedCode {
    delegate! {
        to match self {
            SeparatedCode::Device(code) => code,
            SeparatedCode::Cudnn(code) => code,
        } {
            pub fn write_body<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()>;
        }
    }
}

pub struct DeviceCode {
    body: String,
}

impl DeviceCode {
    pub fn write_body<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(self.body.as_bytes())?;
        Ok(())
    }
}

pub struct CudnnCode {
    stmts: Vec<Statement>,
    init_fn: String,
    init_fn_decl: String,
}

impl<'sched> CudnnCodeGenerator<'sched> {
    pub fn new(schedule: &'sched Schedule, kernel_id: KernelId) -> Self {
        CudnnCodeGenerator {
            schedule,
            kernel_id,
        }
    }

    // TODO: Borrow
    fn get_resolved_tensor_type(
        &self,
        value_id: ValueId,
    ) -> Result<&ResolvedTensorType, BuildError> {
        self.schedule
            .get_resolved_tensor_type(value_id)
            .ok_or(BuildError::UnresolvedType(value_id))
    }

    fn generate(&self) -> Result<CudnnCode, BuildError> {
        let mut stmts = Vec::new();
        let setting = CudnnSettingName::DefaultName;

        let kernel = &self.schedule.kernels[self.kernel_id];
        let KernelBody::Opaque(Opaque {
            op: Operator::Conv(ref conv),
        }) = kernel.body
        else {
            unreachable!();
        };

        let input_ty = self
            .get_resolved_tensor_type(kernel.inputs[args::CONV_DATA].unwrap())?
            .clone();
        let weight_ty = self
            .get_resolved_tensor_type(kernel.inputs[args::CONV_WEIGHT].unwrap())?
            .clone();
        let output_ty = self.get_resolved_tensor_type(kernel.outputs[0])?.clone();

        assert!(input_ty.dims.ndim() == 4);
        assert!(weight_ty.dims.ndim() == 4);
        assert!(output_ty.dims.ndim() == 4);

        let (in_n, in_c, in_h, in_w) = match conv.input_layout {
            Layout::NCHW => (0, 1, 2, 3),
            Layout::NHWC => (0, 3, 1, 2),
        };
        let (out_n, out_c, out_h, out_w) = match conv.output_layout {
            Layout::NCHW => (0, 1, 2, 3),
            Layout::NHWC => (0, 3, 1, 2),
        };

        let input_desc = TensorDescriptor {
            id: setting,
            role: TensorRole::Input,
        };
        let output_desc = TensorDescriptor {
            id: setting,
            role: TensorRole::Output,
        };
        stmts.push(CudnnOps::CreateTensorDescriptor(input_desc).into());
        stmts.push(
            CudnnOps::SetTensor4dDescriptor {
                desc: input_desc,
                data_type: input_ty.elem_type,
                format: conv.input_layout,
                nbatch: input_ty.dims[in_n],
                channels: input_ty.dims[in_c],
                height: input_ty.dims[in_h],
                width: input_ty.dims[in_w],
            }
            .into(),
        );
        stmts.push(CudnnOps::CreateTensorDescriptor(output_desc).into());
        stmts.push(
            CudnnOps::SetTensor4dDescriptor {
                desc: output_desc,
                data_type: output_ty.elem_type,
                format: conv.output_layout,
                nbatch: output_ty.dims[out_n],
                channels: output_ty.dims[out_c],
                height: output_ty.dims[out_h],
                width: output_ty.dims[out_w],
            }
            .into(),
        );

        stmts.push(CudnnOps::CreateFilterDescriptor(setting).into());
        stmts.push(
            CudnnOps::SetFilter4dDescriptor {
                id: setting,
                data_type: weight_ty.elem_type,
                format: Layout::NCHW,
                out_feature_maps: weight_ty.dims[0],
                in_feature_maps: weight_ty.dims[1],
                height: weight_ty.dims[2],
                width: weight_ty.dims[3],
            }
            .into(),
        );
        if let Some(bias) = kernel.inputs.get(args::CONV_BIAS).and_then(|x| *x) {
            let bias_ty = self.get_resolved_tensor_type(bias)?.clone();
            assert!(bias_ty.dims.ndim() == 1);
            assert!(bias_ty.is_contiguous());
            let bias_desc = TensorDescriptor {
                id: setting,
                role: TensorRole::Bias,
            };
            stmts.push(CudnnOps::CreateTensorDescriptor(bias_desc).into());
            stmts.push(
                CudnnOps::SetTensor4dDescriptor {
                    desc: bias_desc,
                    data_type: bias_ty.elem_type,
                    format: Layout::NCHW,
                    nbatch: 1,
                    channels: bias_ty.dims[0],
                    height: 1,
                    width: 1,
                }
                .into(),
            );
        }

        let conv = match kernel.body {
            KernelBody::Opaque(Opaque { ref op }) => match op {
                Operator::Conv(ref conv) => conv,
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        };

        stmts.push(CudnnOps::CreateActivationDescriptor(setting).into());
        stmts.push(
            CudnnOps::SetActivationDescriptor {
                id: setting,
                mode: conv.activation,
                nan_prop: CudnnNanPropagation::NotPropagateNan,
                coef: 0.0,
            }
            .into(),
        );
        let (pad_h, pad_w) = match conv.pad {
            ConvPad::NotSet(ref pad) => (pad[0].0, pad[1].0),
            ConvPad::Valid => (0, 0),
            ConvPad::SameUpper | ConvPad::SameLower => {
                let calc = |dim: usize| {
                    let input = input_ty.dims[2 + dim];
                    let output = output_ty.dims[2 + dim];
                    let stride = conv.strides[dim];
                    let ext_len = stride * (output - 1) + weight_ty.dims[2 + dim];
                    let pad_total = ext_len - input;
                    pad_total / 2 +
                        if matches!(conv.pad, ConvPad::SameLower) {
                            pad_total % 2
                        } else {
                            0
                        }
                };
                (calc(0), calc(1))
            }
        };
        stmts.push(CudnnOps::CreateConvolutionDescriptor(setting).into());
        stmts.push(
            CudnnOps::SetConvolution2dDescriptor {
                id: setting,
                ty: weight_ty.elem_type,
                pad_h,
                pad_w,
                stride_h: conv.strides[0],
                stride_w: conv.strides[1],
                dilation_h: conv.dilations[0],
                dilation_w: conv.dilations[1],
                mode: CudnnConvolutionMode::CrossCorrelation,
            }
            .into(),
        );

        stmts.push(Statement::Raw(format!(
            "{}.find_best_algo(&cudnn_handler_ctx);",
            setting.setting(),
        )));

        let init_fn = format!("init_cudnn_{}", self.kernel_id.index());
        let init_fn_decl = format!(
            "void {init_fn}(CudnnConvSetting &{setting}, CudnnHandlerContext &cudnn_handler_ctx)",
            init_fn = init_fn,
            setting = setting.setting(),
        );

        Ok(CudnnCode {
            stmts,
            init_fn,
            init_fn_decl,
        })
    }
}

impl CudnnCode {
    pub fn write_body<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writeln!(writer, "{} {{", self.init_fn_decl)?;
        for stmt in self.stmts.iter() {
            writeln!(writer, "  {stmt}")?;
        }
        writeln!(writer, "}}")?;
        Ok(())
    }
}

fn ceil_pow2(mut x: usize) -> usize {
    if x == 0 {
        return 1;
    }
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x + 1
}

impl<'sched> HostCodeGenerator<'sched> {
    pub fn new(schedule: &'sched Schedule) -> Self {
        let streams = &schedule.analysis.get::<StreamAllocResult>().0;
        let to_record_events = streams
            .values()
            .flat_map(|s| s.to_wait.iter())
            .copied()
            .collect();
        HostCodeGenerator {
            schedule,
            stmts: Vec::new(),
            init_stmts: Vec::new(),
            destroy_stmts: Vec::new(),
            state_fields: Vec::new(),
            streams,
            to_record_events,
            value2chunk: HashMap::new(),
            hostmem2identifier: HashMap::new(),
            devicemem2identifier: Vec::new(),
            cublas_handlers: IndexMap::new(),
            cudnn_ctxs: IndexMap::new(),
            separated_codes: Vec::new(),
            includes: BTreeSet::from([Include::Local("common.cuh"), Include::System("cuda.h")]),
        }
    }

    fn move_statements(&mut self) -> Vec<Statement> {
        let mut stmts = Vec::new();
        std::mem::swap(&mut self.stmts, &mut stmts);
        stmts
    }

    fn gen_decl_values(&mut self) -> Result<Vec<Statement>, BuildError> {
        use std::collections::hash_map::Entry;

        for (arg_name, value_ids) in &[
            (ARG_INPUT, &self.schedule.inputs[..]),
            (ARG_INITIALIZER, &self.schedule.initializers[..]),
            (ARG_OUTPUT, &self.schedule.outputs[..]),
        ] {
            for (idx, value) in value_ids.iter().enumerate() {
                let ty = self.get_resolved_tensor_type(*value)?.elem_type.to_string();
                let value_name = format!("h_{}_{}", arg_name, value.index());
                let stmt = format!("{ty} *{value_name} = ({ty} *)({arg_name}[{idx}]);",);
                self.stmts.push(Statement::Raw(stmt));
                match self.hostmem2identifier.entry(*value) {
                    Entry::Occupied(_) => {
                        unreachable!("duplicate host value");
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(value_name);
                    }
                }
            }
        }

        let mut mem_sizes = vec![
            ChunkMemSize::default();
            self.schedule.max_chunk_id().map(|id| id + 1).unwrap_or(0)
        ];
        let mem_alloc_result = self.schedule.analysis.get::<mem_alloc::MemAllocResult>();
        for (kernel_id, _kernel) in self.schedule.kernels.iter() {
            let mem_alloc = mem_alloc_result
                .0
                .get(&kernel_id)
                .ok_or(BuildError::UnresolvedAllocateInfo(kernel_id))?;
            for mem in mem_alloc.iter() {
                let chunk_id = if let AllocateType::Chunk(chunk_id) = mem.ty {
                    chunk_id
                } else {
                    unreachable!("non chunk");
                };

                match self.value2chunk.entry(mem.value_id) {
                    Entry::Occupied(entry) => {
                        assert!(*entry.get() == chunk_id);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(chunk_id);
                    }
                }

                let size = self.get_resolved_tensor_type(mem.value_id)?.into();
                mem_sizes[chunk_id].append(size);
            }
        }

        const ALIGNMENT: usize = 256;
        let align_up = |size: usize| (size + ALIGNMENT - 1) & !(ALIGNMENT - 1);

        let mut offsets = Vec::with_capacity(mem_sizes.len());
        let mut arena_size: usize = 0;
        for (chunk_id, mem_size) in mem_sizes.iter().enumerate() {
            offsets.push(arena_size);
            arena_size += align_up(mem_size.max_byte_size());
            let name = format!("d_chunk_{chunk_id}");
            self.devicemem2identifier.push(name);
        }

        if arena_size > 0 {
            self.state_fields
                .push("void *d_arena = nullptr;".to_string());
            self.init_stmts.push(
                Malloc {
                    dst: Expr::Identifier("state->d_arena".to_string()),
                    mem_size: MemSize::Raw(Expr::Identifier(format!("{arena_size}"))),
                }
                .into(),
            );
            self.destroy_stmts
                .push(Free(Expr::Identifier("state->d_arena".to_string())).into());
            for (chunk_id, offset) in offsets.iter().enumerate() {
                let name = &self.devicemem2identifier[chunk_id];
                self.stmts.push(Statement::Raw(format!(
                    "void *{name} = (char*)state->d_arena + {offset};"
                )));
            }
        }

        Ok(self.move_statements())
    }

    // TODO: Borrow
    fn get_resolved_tensor_type(
        &self,
        value_id: ValueId,
    ) -> Result<&ResolvedTensorType, BuildError> {
        self.schedule
            .get_resolved_tensor_type(value_id)
            .ok_or(BuildError::UnresolvedType(value_id))
    }

    fn gen_computes(&mut self) -> Result<Vec<Statement>, BuildError> {
        for (kernel_id, _) in self.schedule.kernels.iter() {
            self.call_kernel(kernel_id)?;
        }
        Ok(self.move_statements())
    }

    fn gen_decl_cuda_objs(&mut self) -> Result<Vec<Statement>, BuildError> {
        use runtime_api::StateRef;

        for event_id in self.to_record_events.iter().copied() {
            self.state_fields
                .push(format!("cudaEvent_t {event_id} = nullptr;"));
            self.init_stmts.push(
                EventCreate {
                    event_id: StateRef(event_id),
                }
                .into_checked_stmt(),
            );
            self.destroy_stmts
                .push(EventDestroy(StateRef(event_id)).into_checked_stmt());
            self.stmts.push(Statement::Raw(format!(
                "cudaEvent_t {event_id} = state->{event_id};"
            )));
        }

        for stream_id in self.streams.values().map(|s| s.stream_id).unique().sorted() {
            self.state_fields
                .push(format!("cudaStream_t {stream_id} = nullptr;"));
            self.init_stmts.push(
                StreamCreate {
                    stream_id: StateRef(stream_id),
                }
                .into_checked_stmt(),
            );
            self.destroy_stmts
                .push(StreamDestroy(StateRef(stream_id)).into_checked_stmt());
            self.stmts.push(Statement::Raw(format!(
                "cudaStream_t {stream_id} = state->{stream_id};"
            )));
        }

        for (_, handler) in self.cublas_handlers.iter() {
            let sh = handler.with_state_prefix();
            self.state_fields
                .push(format!("cublasHandle_t {handler} = nullptr;"));
            self.init_stmts.push(CublasApi::Create(sh).into());
            self.init_stmts.push(CublasApi::SetStream(sh).into());
            self.destroy_stmts.push(CublasApi::Destroy(sh).into());
            self.stmts
                .push(Statement::Raw(format!("cublasHandle_t {handler} = {sh};")));
        }

        for (stream_id, kernels) in self.cudnn_ctxs.iter() {
            let ctx = CudnnContext::new(*stream_id);
            let ctx_name = ctx.ctx();

            let sctx = ctx.with_state_prefix();
            let sctx_name = sctx.ctx();

            self.state_fields
                .push(format!("CudnnHandlerContext {ctx_name};"));

            self.init_stmts.push(CudnnOps::Create(sctx).into());
            self.init_stmts.push(CudnnOps::SetStream(sctx).into());

            for kernel_id in kernels.iter().copied() {
                let setting = CudnnSettingName::KernelId(kernel_id);
                let code = CudnnCodeGenerator::new(self.schedule, kernel_id).generate()?;
                self.init_stmts.push(Statement::Raw(format!(
                    "{init_fn}({ss}, {sctx_name});",
                    init_fn = code.init_fn,
                    ss = setting.state_setting(),
                )));
                self.separated_codes.push(SeparatedCode::Cudnn(code));
            }

            for kernel_id in kernels.iter().copied() {
                let setting = CudnnSettingName::KernelId(kernel_id);
                self.init_stmts.push(Statement::Raw(format!(
                    "{ws_max} = std::max({ws_max}, {ss}.workspace_size_in_bytes);",
                    ws_max = sctx.workspace_max_size(),
                    ss = setting.state_setting(),
                )));
            }
            self.init_stmts.push(
                Malloc {
                    dst: Expr::Identifier(sctx.workspace_ptr()),
                    mem_size: MemSize::Raw(Expr::Identifier(sctx.workspace_max_size())),
                }
                .into(),
            );

            self.destroy_stmts
                .push(Free(Expr::Identifier(sctx.workspace_ptr())).into());
            self.destroy_stmts.push(CudnnOps::Destroy(sctx).into());

            self.stmts.push(Statement::Raw(format!(
                "CudnnHandlerContext &{ctx_name} = {sctx_name};"
            )));
        }

        Ok(self.move_statements())
    }

    fn gen_finalize(&mut self) -> Result<Vec<Statement>, BuildError> {
        self.stmts.push(CudaRuntimeApi::DeviceSynchronize.into());
        Ok(self.move_statements())
    }

    fn device_identifier(&self, value_id: ValueId) -> Result<Expr, BuildError> {
        let chunk_id = self
            .value2chunk
            .get(&value_id)
            .ok_or(BuildError::ChunkNotFound(value_id))?;
        Ok(Expr::Identifier(
            self.devicemem2identifier
                .get(*chunk_id)
                .ok_or(BuildError::NoDeviceVariable(*chunk_id))?
                .clone(),
        ))
    }

    fn single_mem_size(&self, value_id: ValueId) -> Result<SingleMemSize, BuildError> {
        self.schedule
            .get_resolved_tensor_type(value_id)
            .ok_or(BuildError::UnresolvedType(value_id))
            .map(|x| x.into())
    }

    fn generate_kernel<F>(
        &mut self,
        kernel_id: KernelId,
        generator: F,
    ) -> Result<GeneratedKernel, BuildError>
    where
        F: Fn(&Schedule, KernelDecl) -> Result<String, BuildError>,
    {
        let params = chain(
            self.schedule.kernels[kernel_id].outputs.iter().copied(),
            self.schedule.kernels[kernel_id]
                .inputs
                .iter()
                .flatten()
                .copied(),
        )
        .map(|id| {
            let ty = self.get_resolved_tensor_type(id)?;
            let type_symbol: TypeSymbol = ty.elem_type.into();
            let type_symbol = type_symbol.to_pointer();
            Ok((KernelVar::Value(id), type_symbol))
        })
        .collect::<Result<Vec<_>, BuildError>>()?;

        let args = params
            .iter()
            .map(|(param, ty)| {
                let KernelVar::Value(p) = param else {
                    unreachable!()
                };
                let ptr = self.device_identifier(*p)?;
                Ok(Expr::Literal(format!("({}){}", ty, ptr)))
            })
            .collect::<Result<Vec<_>, BuildError>>()?;

        let decl = KernelDecl {
            kernel_id,
            params,
            qualifier: FuncQualifier::Global,
        };
        self.separated_codes.push(SeparatedCode::Device(DeviceCode {
            body: generator(self.schedule, decl.clone())?,
        }));
        Ok(GeneratedKernel { decl, args })
    }

    fn call_kernel(&mut self, kernel_id: KernelId) -> Result<(), BuildError> {
        let kernel = &self.schedule.kernels[kernel_id];
        let KernelStreamAssignment {
            stream_id,
            event_id,
            to_wait,
        } = &self.streams[&kernel_id];
        let stream_id = *stream_id;
        let event_id = *event_id;

        for event in to_wait.iter() {
            self.stmts.push(
                WaitEvent {
                    stream_id,
                    event_id: *event,
                }
                .into(),
            );
        }

        let create_launch_kernel = |cuda_kernel: kernel::CUDAKernel, num_threads: usize| {
            let block_size = DEFAULT_BLOCK_SIZE.to_literal();
            let grid_size = num_threads.div_ceil(DEFAULT_BLOCK_SIZE).to_literal();
            Ok(kernel::LaunchKernel {
                cuda_kernel,
                grid_size,
                block_size,
                shared_mem_bytes: None,
                stream_id,
            })
        };

        // Launch the kernel
        match kernel.body {
            KernelBody::Opaque(Opaque { ref op }) => match op {
                Operator::Transfer(kind) => {
                    let value_id = kernel.inputs[0].unwrap();
                    let mem_size = MemSize::Single(self.single_mem_size(kernel.outputs[0])?);
                    match kind {
                        TransferKind::HostToDevice => {
                            let src = Expr::Identifier(self.hostmem2identifier[&value_id].clone());
                            let dst = self.device_identifier(kernel.outputs[0])?;
                            self.stmts.push(
                                Memcpy {
                                    dst,
                                    src,
                                    mem_size,
                                    kind: CudaMemcpyKind::HostToDevice,
                                    stream: stream_id,
                                }
                                .into(),
                            );
                        }
                        TransferKind::DeviceToHost => {
                            let src = self.device_identifier(value_id)?;
                            let dst = Expr::Identifier(
                                self.hostmem2identifier[&kernel.outputs[0]].clone(),
                            );
                            self.stmts.push(
                                Memcpy {
                                    dst,
                                    src,
                                    mem_size,
                                    kind: CudaMemcpyKind::DeviceToHost,
                                    stream: stream_id,
                                }
                                .into(),
                            );
                        }
                    }
                }

                Operator::Identity | Operator::Reinterpret(_) => {
                    let input_chunk = self
                        .value2chunk
                        .get(&kernel.inputs[0].unwrap())
                        .ok_or(BuildError::ChunkNotFound(kernel.inputs[0].unwrap()))?;
                    let output_chunk = self
                        .value2chunk
                        .get(&kernel.outputs[0])
                        .ok_or(BuildError::ChunkNotFound(kernel.outputs[0]))?;
                    if input_chunk != output_chunk {
                        let output_size = self
                            .get_resolved_tensor_type(kernel.outputs[0])?
                            .dims
                            .size();
                        let generated = self.generate_kernel(kernel_id, |sched, decl| {
                            CopyBuilder::new(sched, decl).build()
                        })?;
                        self.stmts.push(
                            create_launch_kernel(
                                kernel::CUDAKernel::GeneratedKernel(generated),
                                output_size,
                            )?
                            .into(),
                        );
                    }
                }

                Operator::Add |
                Operator::Div |
                Operator::BatchNormalization(_) |
                Operator::Cast(_) |
                Operator::Exp |
                Operator::GeLU(_) |
                Operator::LeakyReLU(_) |
                Operator::Log |
                Operator::Mul |
                Operator::Pow |
                Operator::Reciprocal |
                Operator::ReLU |
                Operator::Sigmoid |
                Operator::Sqrt |
                Operator::Sub |
                Operator::Tanh => unreachable!(),

                Operator::Attention(attn) => {
                    self.includes.insert(Include::Local("attention.cuh"));
                    let q = kernel.inputs[args::ATTENTION_Q].unwrap();
                    let k = kernel.inputs[args::ATTENTION_K].unwrap();
                    let v = kernel.inputs[args::ATTENTION_V].unwrap();

                    let q_ty = self.get_resolved_tensor_type(q)?;
                    let k_ty = self.get_resolved_tensor_type(k)?;
                    let v_ty = self.get_resolved_tensor_type(v)?;

                    let q_dims = &q_ty.dims;
                    let k_dims = &k_ty.dims;
                    let v_dims = &v_ty.dims;
                    assert!(q_dims.ndim() == 4);
                    assert!(k_dims.ndim() == 4);
                    assert!(v_dims.ndim() == 4);
                    assert!(q_dims[0] == k_dims[0] && k_dims[0] == v_dims[0]);
                    assert!(q_dims[1] == k_dims[1] && k_dims[1] == v_dims[1]);

                    // TODO: Maybe `q_dims[2] == k_dims[2]` is unnecessary.
                    assert!(q_dims[2] == k_dims[2] && k_dims[2] == v_dims[2]);
                    assert!(q_dims[3] == k_dims[3] && k_dims[3] == v_dims[3]);

                    let seq_q = q_dims[2];

                    let (mask_expr, mask_outer_stride, mask_row_stride) = if let Some(mask_id) =
                        kernel.inputs.get(args::ATTENTION_MASK).and_then(|x| *x)
                    {
                        let mask_ty = self.get_resolved_tensor_type(mask_id)?;
                        let md = &mask_ty.dims;
                        let mndim = md.ndim();
                        let mask_last_row = if mndim >= 2 { md[mndim - 2] } else { 1 };
                        let mask_last_col = md[mndim - 1];
                        let mask_row_stride = if mask_last_row > 1 { mask_last_col } else { 0 };
                        let mask_slice_size = mask_last_row * mask_last_col;
                        let mask_outer_size: usize = if mndim > 2 {
                            md[..mndim - 2].iter().product()
                        } else {
                            1
                        };
                        let mask_outer_stride = if mask_outer_size > 1 {
                            mask_slice_size
                        } else {
                            0
                        };
                        (
                            Some(self.device_identifier(mask_id)?),
                            mask_outer_stride,
                            mask_row_stride,
                        )
                    } else {
                        (None, 0, 0)
                    };

                    let batch_size = q_dims[0];
                    let num_heads = q_dims[1];
                    let head_size = q_dims[3];
                    let threads_per_row = (head_size / 8).clamp(1, 32);
                    let br = ceil_pow2(seq_q / threads_per_row).clamp(1, 256 / threads_per_row);
                    let bc = ceil_pow2(k_dims[2] / threads_per_row).clamp(1, 256 / threads_per_row);
                    let block_size = br * threads_per_row;
                    let grid_size = {
                        let y = batch_size * num_heads;
                        let x = seq_q.div_ceil(br);
                        format!("dim3({x}, {y})")
                    };
                    let cuda_kernel = kernel::CUDAKernel::AttentionKernel(AttentionKernel {
                        data_ty: q_ty.elem_type,
                        br,
                        bc,
                        threads_per_row,
                        head_dim: head_size,
                        q: self.device_identifier(q)?,
                        k: self.device_identifier(k)?,
                        v: self.device_identifier(v)?,
                        mask: mask_expr,
                        n: seq_q,
                        mask_outer_stride,
                        mask_row_stride,
                        out: self.device_identifier(kernel.outputs[0])?,
                        attn: *attn,
                    });

                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size: grid_size.to_literal(),
                            block_size: block_size.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
                }

                Operator::Concat(_) => {
                    let output_size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        ConcatBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::GeneratedKernel(generated),
                            output_size,
                        )?
                        .into(),
                    );
                }

                Operator::Contiguous(_) => {
                    let output_size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        ContiguousBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::GeneratedKernel(generated),
                            output_size,
                        )?
                        .into(),
                    );
                }

                Operator::Conv(ref conv) => {
                    if conv.kernel_shape.ndim() != 2 {
                        unimplemented!("Only 2D convolution is supported");
                    }

                    let cudnn_handler = {
                        let kernels = self.cudnn_ctxs.entry(stream_id).or_default();
                        kernels.push(kernel_id);
                        CudnnContext::new(stream_id)
                    };

                    let input = self.device_identifier(kernel.inputs[args::CONV_DATA].unwrap())?;
                    let weights =
                        self.device_identifier(kernel.inputs[args::CONV_WEIGHT].unwrap())?;
                    let output = self.device_identifier(kernel.outputs[0])?;
                    let input_ty = self
                        .get_resolved_tensor_type(kernel.inputs[0].unwrap())?
                        .clone();
                    let template_ty = input_ty.elem_type.to_string();
                    let setting = CudnnSettingName::KernelId(kernel_id);

                    let ss = setting.state_setting();
                    self.stmts
                        .push(Statement::Raw(format!("{ss}.x = {input};")));
                    self.stmts
                        .push(Statement::Raw(format!("{ss}.w = {weights};")));
                    self.stmts
                        .push(Statement::Raw(format!("{ss}.y = {output};")));
                    let func =
                        if let Some(bias) = kernel.inputs.get(args::CONV_BIAS).and_then(|x| *x) {
                            self.stmts.push(Statement::Raw(format!(
                                "{ss}.bias = {};",
                                self.device_identifier(bias)?,
                            )));
                            "call_conv_bias_activation_forward"
                        } else {
                            "call_conv_forward"
                        };
                    self.stmts.push(Statement::Raw(format!(
                        "{ss}.{func}<{ty}>(&{ctx});",
                        func = func,
                        ty = template_ty,
                        ctx = cudnn_handler.ctx()
                    )));
                }

                Operator::Gather(_) => {
                    let indices_size = self
                        .get_resolved_tensor_type(kernel.inputs[1].unwrap())?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        GatherBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::GeneratedKernel(generated),
                            indices_size,
                        )?
                        .into(),
                    );
                }

                op @ (Operator::Gemm(_) | Operator::BatchedGemm(_)) => {
                    let Gemm {
                        alpha,
                        beta,
                        trans_a,
                        trans_b,
                    } = match op {
                        Operator::Gemm(gemm) => gemm.clone(),
                        Operator::BatchedGemm(gemm) => Gemm {
                            alpha: gemm.alpha,
                            beta: gemm.beta,
                            trans_a: gemm.trans_a,
                            trans_b: gemm.trans_b,
                        },
                        _ => unreachable!(),
                    };

                    let elem_ty = self.get_resolved_tensor_type(kernel.outputs[0])?.elem_type;
                    let c_data_ty = elem_ty.to_string();
                    let alpha = {
                        let var_name = format!("alpha_{}", kernel_id.index());
                        self.stmts.push(Statement::Raw(format!(
                            "{ty} {name} = {value};",
                            ty = c_data_ty,
                            name = var_name,
                            value = alpha,
                        )));
                        var_name
                    };
                    let beta = {
                        let var_name = format!("beta_{}", kernel_id.index());
                        self.stmts.push(Statement::Raw(format!(
                            "{ty} {name} = {value};",
                            ty = c_data_ty,
                            name = var_name,
                            value = beta,
                        )));
                        var_name
                    };
                    let handler = *self
                        .cublas_handlers
                        .entry(stream_id)
                        .or_insert_with(|| CublasHandler::new(stream_id));

                    // cuBLAS is column-major!
                    // We have t(A) and t(B), and want t(C).
                    // t(C) = t(A * B) = t(B) * t(A).
                    let a_ty = self
                        .get_resolved_tensor_type(kernel.inputs[0].unwrap())?
                        .clone();
                    let b_ty = self
                        .get_resolved_tensor_type(kernel.inputs[1].unwrap())?
                        .clone();
                    let (m, k, n) = {
                        let [m, k] = &a_ty.dims.suffix(2)[..] else {
                            unreachable!();
                        };
                        let (m, k) = if trans_a { (k, m) } else { (m, k) };
                        let [k_, n] = &b_ty.dims.suffix(2)[..] else {
                            unreachable!();
                        };
                        let (k_, n) = if trans_b { (n, k_) } else { (k_, n) };
                        assert!(k == k_);
                        (*n, *k, *m)
                    };

                    let transposed_layout = |ty: ResolvedTensorType| {
                        if ty.is_contiguous() {
                            return false;
                        }

                        let ndim = ty.dims.ndim();
                        let row = ty.dims[ndim - 2];
                        assert!(ty.strides()[ndim - 1] == row);
                        assert!(ty.strides()[ndim - 2] == 1);
                        true
                    };
                    let trans_a = trans_a ^ transposed_layout(a_ty);
                    let trans_b = trans_b ^ transposed_layout(b_ty);
                    let lda = if !trans_b { m } else { k };
                    let ldb = if !trans_a { k } else { n };
                    let ldc = m;
                    let (trans_a, trans_b) = {
                        let tmp0 = if trans_b {
                            CublasOperation::Transpose
                        } else {
                            CublasOperation::Non
                        };

                        let tmp1 = if trans_a {
                            CublasOperation::Transpose
                        } else {
                            CublasOperation::Non
                        };
                        (tmp0, tmp1)
                    };

                    // TODO: Copy bias into output chunk if bias_chunk != output_chunk.
                    if kernel.inputs.len() == 3 {
                        assert!(matches!(op, Operator::Gemm(_)));
                        let bias_chunk = self
                            .value2chunk
                            .get(&kernel.inputs[args::GEMM_C].unwrap())
                            .unwrap();
                        let output_chunk = self.value2chunk.get(&kernel.outputs[0]).unwrap();
                        assert!(bias_chunk == output_chunk);
                    }

                    let a = self
                        .device_identifier(kernel.inputs[1].unwrap())?
                        .to_string();
                    let b = self
                        .device_identifier(kernel.inputs[0].unwrap())?
                        .to_string();
                    let c = self.device_identifier(kernel.outputs[0])?.to_string();

                    let gemm = GemmArgs {
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
                        data_ty: elem_ty,
                    };

                    match op {
                        Operator::Gemm(_) => {
                            self.stmts.push(CublasApi::Gemm(gemm).into());
                        }
                        Operator::BatchedGemm(_) => {
                            // cuBLAS A = row-major B (inputs[1]), cuBLAS B = row-major A (inputs[0])
                            let stride_a = m * k; // cuBLAS A stride
                            let stride_b = k * n; // cuBLAS B stride
                            let stride_c = m * n;
                            let batch_count = self
                                .get_resolved_tensor_type(kernel.inputs[1].unwrap())?
                                .dims
                                .size() /
                                stride_a;

                            let bgemm = BatchedGemmArgs {
                                gemm,
                                stride_a,
                                stride_b,
                                stride_c,
                                batch_count,
                            };
                            self.stmts.push(CublasApi::BatchedGemm(bgemm).into());
                        }
                        _ => unreachable!(),
                    }
                }

                Operator::MatMul => {
                    todo!("MatMul with broadcast should be lowered to BatchedGemm or loop of Gemm")
                }

                Operator::LayerNormalization(LayerNormalization { axis, epsilon }) => {
                    self.includes.insert(Include::Local("layer_norm.cuh"));
                    let input_ty = self.get_resolved_tensor_type(
                        kernel.inputs[operator::args::LAYER_NORM_DATA].unwrap(),
                    )?;
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ = self.device_identifier(
                        kernel.inputs[operator::args::LAYER_NORM_DATA].unwrap(),
                    )?;
                    let scale = self.device_identifier(
                        kernel.inputs[operator::args::LAYER_NORM_SCALE].unwrap(),
                    )?;
                    let bias = self.device_identifier(
                        kernel.inputs[operator::args::LAYER_NORM_BIAS].unwrap(),
                    )?;
                    let axis = axis.index(input_ty.dims.ndim());
                    let epsilon = *epsilon;
                    if input_ty.strides().last() != Some(&1) || axis != input_ty.dims.ndim() - 1 {
                        unimplemented!(
                            "LayerNormalization currently supports only last-axis normalization"
                        );
                    }
                    let axis_dim = input_ty.dims[axis];
                    let size = input_ty.dims.size();
                    let block_size = min(DEFAULT_BLOCK_SIZE, ceil_pow2(axis_dim));
                    let grid_size = size / axis_dim;
                    let data_ty = input_ty.elem_type;
                    let cuda_kernel =
                        kernel::CUDAKernel::LayerNormKernel(kernel::LayerNormKernel {
                            data_ty,
                            block_size,
                            axis_dim,
                            out,
                            in_,
                            scale,
                            bias,
                            size: size.to_literal(),
                            epsilon,
                        });
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size: grid_size.to_literal(),
                            block_size: block_size.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
                }

                Operator::MaxPool(_) => {
                    let size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        MaxPoolBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(kernel::CUDAKernel::GeneratedKernel(generated), size)?
                            .into(),
                    );
                }

                Operator::OneHot(_) => {
                    let size = self
                        .get_resolved_tensor_type(kernel.inputs[args::ONEHOT_INDICES].unwrap())?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        OneHotBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(kernel::CUDAKernel::GeneratedKernel(generated), size)?
                            .into(),
                    );
                }

                Operator::ReduceMatrix(_) => {
                    self.includes
                        .insert(Include::System("cooperative_groups.h"));
                    let input_ty = self.get_resolved_tensor_type(kernel.inputs[0].unwrap())?;
                    let [row, col] = input_ty.dims[..] else {
                        panic!("Invalid ReduceMatrix output shape");
                    };
                    let block_size = min(DEFAULT_BLOCK_SIZE, ceil_pow2(col));
                    let grid_size = row.to_literal();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        ReduceMatrixBuilder::new(sched, decl).build(block_size)
                    })?;
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel: kernel::CUDAKernel::GeneratedKernel(generated),
                            grid_size,
                            block_size: block_size.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
                }

                Operator::Softmax(Softmax { axis }) => {
                    self.includes.insert(Include::Local("softmax.cuh"));
                    let input_ty = self.get_resolved_tensor_type(kernel.inputs[0].unwrap())?;
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ = self.device_identifier(kernel.inputs[0].unwrap())?;
                    let axis = axis.index(input_ty.dims.ndim());
                    let axis_dim = input_ty.dims[axis];
                    let axis_stride = input_ty.stride(axis);
                    let size = input_ty.dims.size();
                    let block_size = min(DEFAULT_BLOCK_SIZE, ceil_pow2(axis_dim));
                    let grid_size = size / axis_dim;
                    let data_ty = input_ty.elem_type;
                    let cuda_kernel = kernel::CUDAKernel::SoftmaxKernel(kernel::SoftmaxKernel {
                        data_ty,
                        block_size,
                        out,
                        in_,
                        axis_dim,
                        axis_stride,
                        size: size.to_literal(),
                    });
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size: grid_size.to_literal(),
                            block_size: block_size.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
                }

                Operator::Resize(_) => {
                    let output_size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        ResizeBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::GeneratedKernel(generated),
                            output_size,
                        )?
                        .into(),
                    );
                }

                Operator::Split(_) => {
                    let input_size = self
                        .get_resolved_tensor_type(kernel.inputs[0].unwrap())?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        SplitBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::GeneratedKernel(generated),
                            input_size,
                        )?
                        .into(),
                    );
                }

                _ => {
                    dbg!(&kernel);
                    unimplemented!("Kernel body not implemented: {:?}", op)
                }
            },
            KernelBody::ElementWises(_) => {
                let output_size = self
                    .get_resolved_tensor_type(kernel.outputs[0])?
                    .dims
                    .size();
                let generated = self.generate_kernel(kernel_id, |sched, decl| {
                    ElementwiseKernelBuilder::new(sched, decl).build()
                })?;
                self.stmts.push(
                    create_launch_kernel(
                        kernel::CUDAKernel::GeneratedKernel(generated),
                        output_size,
                    )?
                    .into(),
                );
            }
        }

        if self.to_record_events.contains(&event_id) {
            self.stmts.push(
                RecordEvent {
                    event_id,
                    stream_id,
                }
                .into(),
            );
        }

        Ok(())
    }

    pub fn generate(&mut self, opt: &Options) -> Result<HostCode, BuildError> {
        let decl_values = self.gen_decl_values()?;
        let computes = self.gen_computes()?;
        let finalize = self.gen_finalize()?;

        // NOTE: This must be called at the very end.
        let decl_cuda_objs = self.gen_decl_cuda_objs()?;

        let mut kernel_codes = Vec::new();
        std::mem::swap(&mut self.separated_codes, &mut kernel_codes);

        if !self.cublas_handlers.is_empty() {
            self.includes.insert(Include::System("cublas_v2.h"));
        }
        if !self.cudnn_ctxs.is_empty() {
            self.includes.insert(Include::Local("cudnn_setting.h"));
        }

        let mut state_fields = std::mem::take(&mut self.state_fields);
        let mut destroy_body = std::mem::take(&mut self.destroy_stmts);
        let init_body = std::mem::take(&mut self.init_stmts);
        for (_, kernels) in self.cudnn_ctxs.iter() {
            for kernel_id in kernels.iter().copied() {
                let setting = CudnnSettingName::KernelId(kernel_id);
                state_fields.push(format!("CudnnConvSetting {};", setting.setting()));
                let ss = CudnnSettingName::StateKernelId(kernel_id);
                destroy_body.push(
                    CudnnOps::DestroyTensorDescriptor(TensorDescriptor {
                        id: ss,
                        role: TensorRole::Input,
                    })
                    .into(),
                );
                destroy_body.push(
                    CudnnOps::DestroyTensorDescriptor(TensorDescriptor {
                        id: ss,
                        role: TensorRole::Output,
                    })
                    .into(),
                );
                destroy_body.push(CudnnOps::DestroyFilterDescriptor(ss).into());
                destroy_body.push(CudnnOps::DestroyConvolutionDescriptor(ss).into());
                if self.schedule.kernels[kernel_id]
                    .inputs
                    .get(args::CONV_BIAS)
                    .is_some()
                {
                    destroy_body.push(
                        CudnnOps::DestroyTensorDescriptor(TensorDescriptor {
                            id: ss,
                            role: TensorRole::Bias,
                        })
                        .into(),
                    );
                }
                destroy_body.push(CudnnOps::DestroyActivationDescriptor(ss).into());
            }
        }

        Ok(HostCode {
            state_fields,
            init_body,
            destroy_body,
            decl_values,
            decl_cuda_objs,
            computes,
            finalize,
            kernel_codes,
            includes: self.includes.clone(),
            profile: opt.profile,
        })
    }
}

impl HostCode {
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for h in ["algorithm", "limits", "chrono", "iostream", "cstring"] {
            writeln!(writer, "#include <{}>", h)?;
        }
        for h in &self.includes {
            match h {
                Include::System(name) => writeln!(writer, "#include <{}>", name)?,
                Include::Local(name) => writeln!(writer, "#include \"{}\"", name)?,
            }
        }
        if self
            .includes
            .contains(&Include::System("cooperative_groups.h"))
        {
            writeln!(writer, "namespace cg = cooperative_groups;")?;
        }

        for code in self.kernel_codes.iter() {
            match code {
                SeparatedCode::Cudnn(code) => {
                    code.write_body(writer)?;
                }
                SeparatedCode::Device(code) => {
                    code.write_body(writer)?;
                }
            }
        }

        // ModelState struct
        writeln!(writer, "struct ModelState {{")?;
        for field in &self.state_fields {
            writeln!(writer, "  {field}")?;
        }
        writeln!(writer, "}};\n")?;

        let init_body = self
            .init_body
            .iter()
            .map(|l| format!("  {l}"))
            .collect::<Vec<_>>()
            .join("\n");
        let destroy_body = self
            .destroy_body
            .iter()
            .map(|l| format!("  {l}"))
            .collect::<Vec<_>>()
            .join("\n");

        write!(
            writer,
            r#"extern "C" void* model_init() {{
  auto *state = new ModelState;
{init_body}
  return state;
}}

extern "C" void model_destroy(void *ptr) {{
  auto *state = static_cast<ModelState*>(ptr);
{destroy_body}
  delete state;
}}

extern "C" void model(void *state_ptr, void **{ARG_OUTPUT}, void **{ARG_INPUT}, void **{ARG_INITIALIZER}) {{
  auto *state = static_cast<ModelState*>(state_ptr);
"#
        )?;

        if self.profile {
            writeln!(
                writer,
                "
    auto timer_start = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();"
            )?;
        }

        for stmts in &[
            self.decl_values.as_slice(),
            self.decl_cuda_objs.as_slice(),
            self.computes.as_slice(),
            self.finalize.as_slice(),
        ] {
            for stmt in stmts.iter() {
                writeln!(writer, "  {stmt}")?;
            }
        }

        if self.profile {
            writeln!(writer, "
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);
    std::cout << \"Elapsed time: \" << elapsed_ms.count() << \" ms\" << std::endl;")?;
        }
        writeln!(writer, "}}")?;
        Ok(())
    }
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::Model;
    use crate::options::*;
    use crate::schedule::Schedule;

    #[ignore]
    #[test]
    fn test_cuda() {
        use std::fs::OpenOptions;
        use std::io::BufWriter;
        use std::path::PathBuf;

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/single_op")
            .join("conv.onnx");
        let model = Model::load_from_path(path).unwrap();
        let options = Options::builder().target(Target::CUDA).build();
        let mut schedule = Schedule::new(model.graph, options.clone());
        let schedule_passes = crate::schedule::create_schedule_passes(&options);
        schedule_passes.run(&mut schedule);

        let mut host_gen = HostCodeGenerator::new(&schedule);
        host_gen.gen_decl_values().unwrap();

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("a.cu");
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path.clone())
            .unwrap();
        let mut file = BufWriter::new(file);
        let code = host_gen.generate(&Options::builder().build()).unwrap();
        code.write(&mut file).unwrap();
        println!("Generated CUDA code written to {:?}", path);
    }

    fn generate_cuda_code(model_path: &str) -> String {
        use std::path::PathBuf;

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(model_path);
        let model = Model::load_from_path(path).unwrap();
        let options = Options::builder().target(Target::CUDA).build();
        let mut graph = model.graph;
        crate::transform::transform_graph(&mut graph, &options);
        let mut schedule = Schedule::new(graph, options.clone());
        let schedule_passes = crate::schedule::create_schedule_passes(&options);
        schedule_passes.run(&mut schedule);

        let mut host_gen = HostCodeGenerator::new(&schedule);
        let code = host_gen.generate(&options).unwrap();
        let mut buf = Vec::new();
        code.write(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn test_cuda_two_conv_codegen() {
        let code = generate_cuda_code("models/test/single_op/two_conv.onnx");
        insta::assert_snapshot!(code);
    }
}
