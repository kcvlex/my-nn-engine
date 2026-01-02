mod cublas;
mod cudnn;
mod kernel;
mod runtime_api;

use std::cmp::min;
use std::collections::HashMap;
use std::collections::HashSet;

use delegate::delegate;
use derive_more::From;
use indexmap::IndexMap;
use indexmap::IndexSet;
use itertools::chain;
use itertools::Itertools;

use crate::codegen::cuda::cublas::*;
use crate::codegen::cuda::cudnn::*;
use crate::codegen::cuda::kernel::ConcatBuilder;
use crate::codegen::cuda::kernel::ContiguousBuilder;
use crate::codegen::cuda::kernel::ElementwiseKernelBuilder;
use crate::codegen::cuda::kernel::GeneratedKernel;
use crate::codegen::cuda::kernel::KernelDecl;
use crate::codegen::cuda::kernel::KernelVar;
use crate::codegen::cuda::kernel::ReduceMatrixKernel;
use crate::codegen::cuda::kernel::ResizeBuilder;
use crate::codegen::cuda::kernel::SplitBuilder;
use crate::codegen::cuda::kernel::TypeSymbol;
use crate::codegen::cuda::runtime_api::*;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::options::Options;
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
    Chunk(ChunkMemSize),
    Raw(Expr),
}

impl std::fmt::Display for MemSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemSize::Single(size) => write!(f, "{}", size),
            MemSize::Chunk(size) => write!(f, "{}", size),
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

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct StreamId(usize);

impl StreamId {
    fn index(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream_{}", self.0)
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct EventId(usize);

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "event_{}", self.0)
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

pub struct HostCodeGenerator<'sched> {
    schedule: &'sched Schedule,

    stmts: Vec<Statement>,

    streams: Streams,

    event2stream: Vec<StreamId>,

    value2chunk: HashMap<ValueId, ChunkId>,
    value2event: HashMap<ValueId, EventId>,
    hostmem2identifier: HashMap<ValueId, String>,
    devicemem2identifier: Vec<String>,

    used_event: IndexSet<EventId>,
    to_transfer: HashSet<ValueId>,

    event_slot: IdSlot<EventId, fn(usize) -> EventId>,

    cublas_handlers: IndexMap<StreamId, CublasHandler>,
    cudnn_ctxs: IndexMap<StreamId, Vec<KernelId>>,

    separated_codes: Vec<SeparatedCode>,
}

pub struct HostCode {
    decl_values: Vec<Statement>,
    decl_cuda_objs: Vec<Statement>,
    computes: Vec<Statement>,
    finalize: Vec<Statement>,
    pub kernel_codes: Vec<SeparatedCode>,

    profile: bool,
}

const ARG_INPUT: &str = "input";
const ARG_OUTPUT: &str = "output";
const ARG_INITIALIZER: &str = "initializer";
const MAX_STREAMS: usize = 1; // 16;
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

    fn infer_format(ty: &ResolvedTensorType) -> Option<CudnnTensorFormat> {
        assert!(ty.dims.ndim() == 4);
        if ty.is_contiguous() {
            return Some(CudnnTensorFormat::NCHW);
        }

        let ty = ty.transpose(&[0, 2, 3, 1]);
        if ty.is_contiguous() {
            return Some(CudnnTensorFormat::NHWC);
        }

        None
    }

    fn generate(&self) -> Result<CudnnCode, BuildError> {
        let mut stmts = Vec::new();
        let setting = CudnnSettingName::DefaultName;

        let kernel = &self.schedule.kernels[self.kernel_id];
        let input_ty = self
            .get_resolved_tensor_type(kernel.inputs[args::CONV_DATA])?
            .clone();
        let weight_ty = self
            .get_resolved_tensor_type(kernel.inputs[args::CONV_WEIGHT])?
            .clone();
        let output_ty = self.get_resolved_tensor_type(kernel.outputs[0])?.clone();

        assert!(input_ty.dims.ndim() == 4);
        assert!(weight_ty.dims.ndim() == 4);
        assert!(output_ty.dims.ndim() == 4);
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
                format: Self::infer_format(&input_ty).unwrap(),
                nbatch: input_ty.dims[0],
                channels: input_ty.dims[1],
                height: input_ty.dims[2],
                width: input_ty.dims[3],
            }
            .into(),
        );
        stmts.push(CudnnOps::CreateTensorDescriptor(output_desc).into());
        stmts.push(
            CudnnOps::SetTensor4dDescriptor {
                desc: output_desc,
                data_type: output_ty.elem_type,
                format: Self::infer_format(&output_ty).unwrap(),
                nbatch: output_ty.dims[0],
                channels: output_ty.dims[1],
                height: output_ty.dims[2],
                width: output_ty.dims[3],
            }
            .into(),
        );

        stmts.push(CudnnOps::CreateFilterDescriptor(setting).into());
        stmts.push(
            CudnnOps::SetFilter4dDescriptor {
                id: setting,
                data_type: weight_ty.elem_type,
                format: Self::infer_format(&weight_ty).unwrap(),
                out_feature_maps: weight_ty.dims[0],
                in_feature_maps: weight_ty.dims[1],
                height: weight_ty.dims[2],
                width: weight_ty.dims[3],
            }
            .into(),
        );
        if let Some(bias) = kernel.inputs.get(args::CONV_BIAS).copied() {
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
                    format: CudnnTensorFormat::NCHW,
                    nbatch: 1,
                    channels: bias_ty.dims[0],
                    height: 1,
                    width: 1,
                }
                .into(),
            );

            // TODO: Set proper activation.
            let activation = CudnnActivationMode::Identity;
            stmts.push(CudnnOps::CreateActivationDescriptor(setting).into());
            stmts.push(
                CudnnOps::SetActivationDescriptor {
                    id: setting,
                    mode: activation,
                    nan_prop: CudnnNanPropagation::NotPropagateNan,
                    coef: 0.0, // only used for clipped ReLU
                }
                .into(),
            );
        }

        let conv = match kernel.body {
            KernelBody::SingleKernel(SingleKernel { ref op }) => match op {
                Operator::Conv(ref conv) => conv,
                _ => unimplemented!(),
            },
            _ => unimplemented!(),
        };
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

        let ctx = CudnnContext::DefaultContext;
        stmts.push(CudnnOps::GetConvolutionForwardWorkspaceSize { ctx, id: setting }.into());

        let init_fn = format!("init_cudnn_{}", self.kernel_id.index());
        let init_fn_decl = format!(
            "void {init_fn}(CudnnConvSetting &{setting}, CudnnHandlerContext &{ctx})",
            init_fn = init_fn,
            setting = setting.setting(),
            ctx = ctx.ctx(),
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

struct Streams {
    inner: Vec<StreamId>,
    head: usize,
}

impl Streams {
    fn new() -> Self {
        Streams {
            inner: (0..MAX_STREAMS).map(StreamId).collect(),
            head: 0,
        }
    }

    fn pick_head(&mut self) -> StreamId {
        let res = self.inner[self.head];
        self.head = (self.head + 1) % MAX_STREAMS;
        res
    }

    fn _pick_internal(&mut self, idx: usize) -> StreamId {
        if idx == self.head {
            return self.pick_head();
        }

        let res = self.inner[idx];
        for i in 0..MAX_STREAMS {
            let cur = (idx + i) % MAX_STREAMS;
            let next = (cur + 1) % MAX_STREAMS;
            if next == self.head {
                break;
            }
            self.inner[cur] = self.inner[next];
        }

        res
    }

    fn pick<Pred>(&mut self, pred: Pred) -> StreamId
    where
        Pred: Fn(StreamId) -> bool,
    {
        for i in (self.head..(self.head + MAX_STREAMS)).rev() {
            let idx = i % MAX_STREAMS;
            if pred(self.inner[idx]) {
                return self._pick_internal(idx);
            }
        }
        self.pick_head()
    }
}

struct IdSlot<T, F: Fn(usize) -> T> {
    slot: usize,
    factory: F,
}

impl<T, F: Fn(usize) -> T> IdSlot<T, F> {
    fn new(factory: F) -> Self {
        IdSlot { slot: 0, factory }
    }

    fn issue(&mut self) -> T {
        let res = (self.factory)(self.slot);
        self.slot += 1;
        res
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
        HostCodeGenerator {
            schedule,
            stmts: Vec::new(),
            streams: Streams::new(),
            event2stream: Vec::new(),
            value2chunk: HashMap::new(),
            value2event: HashMap::new(),
            hostmem2identifier: HashMap::new(),
            devicemem2identifier: Vec::new(),
            used_event: IndexSet::new(),
            to_transfer: schedule
                .inputs
                .iter()
                .chain(schedule.initializers.iter())
                .copied()
                .collect(),
            event_slot: IdSlot::new(EventId),
            cublas_handlers: IndexMap::new(),
            cudnn_ctxs: IndexMap::new(),
            separated_codes: Vec::new(),
        }
    }

    fn pick_stream(&mut self, events: &[EventId]) -> StreamId {
        let depends_on = events
            .iter()
            .map(|id| self.event2stream[id.0])
            .unique()
            .collect::<Vec<_>>();
        let pred = |stream_id: StreamId| depends_on.contains(&stream_id);
        self.streams.pick(pred)
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
            (ARG_OUTPUT, &self.schedule.outputs[..]),
            (ARG_INITIALIZER, &self.schedule.initializers[..]),
        ] {
            for (idx, value) in value_ids.iter().enumerate() {
                let ty = self.get_resolved_tensor_type(*value)?.elem_type.to_string();
                let value_name = format!("h_{}_{}", arg_name, value.index());
                let stmt = format!("{ty} *{value_name} = ({ty} *)({arg_name}[{idx}]);",);
                self.stmts.push(Statement::Raw(stmt));
                self.hostmem2identifier.insert(*value, value_name);
            }
        }

        let mut mem_sizes = vec![
            ChunkMemSize::default();
            self.schedule.max_chunk_id().map(|id| id + 1).unwrap_or(0)
        ];
        for (kernel_id, kernel) in self.schedule.kernels.iter() {
            let mem_alloc = kernel
                .mem_alloc
                .as_ref()
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

        for chunk_id in 0..mem_sizes.len() {
            let name = format!("d_chunk_{chunk_id}");
            self.stmts.push(Statement::Raw(format!("void *{name};")));
            self.devicemem2identifier.push(name);
        }
        for (chunk_id, mem_size) in mem_sizes.into_iter().enumerate() {
            let name = &self.devicemem2identifier[chunk_id];
            self.stmts.push(
                Malloc {
                    dst: Expr::Identifier(name.clone()),
                    mem_size: MemSize::Chunk(mem_size),
                }
                .into(),
            );
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
        for event_id in self.used_event.iter().copied() {
            self.stmts
                .push(Statement::Raw(format!("cudaEvent_t {event_id};")));
            self.stmts.push(EventCreate { event_id }.into());
        }

        for stream_id in self.streams.inner.iter().copied() {
            self.stmts
                .push(Statement::Raw(format!("cudaStream_t {stream_id};")));
            self.stmts.push(StreamCreate { stream_id }.into());
        }

        for (_, handler) in self.cublas_handlers.iter() {
            self.stmts.push(Statement::Raw(format!(
                "cublasHandle_t {handler};",
                handler = handler,
            )));
            self.stmts.push(CublasApi::Create(*handler).into());
            self.stmts.push(CublasApi::SetStream(*handler).into());
        }

        for (stream_id, kernels) in self.cudnn_ctxs.iter() {
            let ctx = CudnnContext::StreamContext(*stream_id);
            self.stmts.push(Statement::Raw(format!(
                "CudnnHandlerContext {ctx};",
                ctx = ctx.ctx()
            )));

            self.stmts.push(CudnnOps::Create(ctx).into());
            self.stmts.push(CudnnOps::SetStream(*stream_id).into());
            for kernel_id in kernels.iter().copied() {
                let setting = CudnnSettingName::KernelId(kernel_id);
                self.stmts.push(Statement::Raw(format!(
                    "CudnnConvSetting {setting};",
                    setting = setting.setting()
                )));
                let code = CudnnCodeGenerator::new(self.schedule, kernel_id).generate()?;
                self.stmts.push(Statement::Raw(format!(
                    "{init_fn}({setting}, {ctx});",
                    init_fn = code.init_fn,
                    setting = setting.setting(),
                    ctx = CudnnContext::StreamContext(*stream_id).ctx(),
                )));
                self.stmts.push(Statement::Raw(format!(
                    "{workspace_size_max} = std::max({workspace_size_max}, {workspace_size});",
                    workspace_size_max = ctx.workspace_max_size(),
                    workspace_size = setting.workspace_size(),
                )));
                self.separated_codes.push(SeparatedCode::Cudnn(code));
            }

            self.stmts.push(
                Malloc {
                    dst: Expr::Identifier(ctx.workspace_ptr()),
                    mem_size: MemSize::Raw(Expr::Identifier(ctx.workspace_max_size())),
                }
                .into(),
            );
        }

        Ok(self.move_statements())
    }

    fn gen_finalize(&mut self) -> Result<Vec<Statement>, BuildError> {
        self.stmts.push(CudaRuntimeApi::DeviceSynchronize.into());
        Ok(self.move_statements())
    }

    fn record_event(&mut self, stream_id: StreamId, values: &[ValueId]) -> EventId {
        let event_id = self.event_slot.issue();
        self.event2stream.push(stream_id);
        self.stmts.push(
            RecordEvent {
                event_id,
                stream_id,
            }
            .into(),
        );
        for value in values {
            self.value2event.insert(*value, event_id);
        }
        event_id
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
            self.schedule.kernels[kernel_id].outputs.iter(),
            self.schedule.kernels[kernel_id].inputs.iter(),
        )
        .map(|id| {
            let ty = self.get_resolved_tensor_type(*id)?;
            let type_symbol: TypeSymbol = ty.elem_type.into();
            let type_symbol = type_symbol.to_pointer();
            Ok((KernelVar::Value(*id), type_symbol))
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

        let decl = KernelDecl { kernel_id, params };
        self.separated_codes.push(SeparatedCode::Device(DeviceCode {
            body: generator(self.schedule, decl.clone())?,
        }));
        Ok(GeneratedKernel { decl, args })
    }

    fn call_kernel(&mut self, kernel_id: KernelId) -> Result<(), BuildError> {
        let kernel = &self.schedule.kernels[kernel_id];
        let mem_alloc = kernel
            .mem_alloc
            .as_ref()
            .ok_or(BuildError::UnresolvedAllocateInfo(kernel_id))?;
        let (copy, computed): (Vec<_>, Vec<_>) = mem_alloc
            .iter()
            .cloned()
            .filter(|info| kernel.inputs.contains(&info.value_id))
            .partition(|info| self.to_transfer.remove(&info.value_id));

        let memcpy_stream = self.streams.pick_head();
        for trans in copy.iter() {
            let value_id = trans.value_id;
            let chunk_id = trans
                .ty
                .chunk_id()
                .ok_or(BuildError::UnexpectedMemAlloc(value_id))?;
            let mem_size = MemSize::Single(self.single_mem_size(value_id)?);
            let dst = Expr::Identifier(self.devicemem2identifier[chunk_id].clone());
            let src = Expr::Identifier(self.hostmem2identifier[&value_id].clone());
            self.stmts.push(
                Memcpy {
                    dst,
                    src,
                    mem_size,
                    kind: CudaMemcpyKind::HostToDevice,
                    stream: memcpy_stream,
                }
                .into(),
            );
        }
        let memcpy_event = self.record_event(
            memcpy_stream,
            copy.iter()
                .map(|info| info.value_id)
                .collect::<Vec<_>>()
                .as_slice(),
        );
        let pred_events = computed
            .iter()
            .map(|info| self.value2event[&info.value_id])
            .chain(std::iter::once(memcpy_event))
            .collect::<Vec<_>>();
        let kernel_stream = self.pick_stream(&pred_events);
        let event_to_wait = pred_events
            .iter()
            .filter(|event_id| {
                let stream_id = self.event2stream[event_id.0];
                stream_id != kernel_stream
            })
            .collect::<Vec<_>>();
        for event in event_to_wait {
            self.used_event.insert(*event);
            self.stmts.push(
                WaitEvent {
                    stream_id: kernel_stream,
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
                stream_id: kernel_stream,
            })
        };

        // Launch the kernel
        match kernel.body {
            KernelBody::SingleKernel(SingleKernel { ref op }) => match op {
                Operator::Identity => {
                    let input_chunk = self
                        .value2chunk
                        .get(&kernel.inputs[0])
                        .ok_or(BuildError::ChunkNotFound(kernel.inputs[0]))?;
                    let output_chunk = self
                        .value2chunk
                        .get(&kernel.outputs[0])
                        .ok_or(BuildError::ChunkNotFound(kernel.outputs[0]))?;
                    if input_chunk != output_chunk {
                        unimplemented!("Identity between different chunks is not supported");
                    }
                }
                Operator::Add |
                Operator::BatchNormalization(_) |
                Operator::Exp |
                Operator::LeakyReLU(_) |
                Operator::Log |
                Operator::Mul |
                Operator::Pow |
                Operator::Reciprocal |
                Operator::ReLU |
                Operator::Sigmoid |
                Operator::Sqrt |
                Operator::Sub |
                Operator::Tanh => {
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

                Operator::Contiguous => {
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
                        let kernels = self.cudnn_ctxs.entry(kernel_stream).or_default();
                        kernels.push(kernel_id);
                        CudnnContext::StreamContext(kernel_stream)
                    };

                    let input = self.device_identifier(kernel.inputs[args::CONV_DATA])?;
                    let weights = self.device_identifier(kernel.inputs[args::CONV_WEIGHT])?;
                    let output = self.device_identifier(kernel.outputs[0])?;
                    let input_ty = self.get_resolved_tensor_type(kernel.inputs[0])?.clone();
                    let template_ty = input_ty.elem_type.to_string();
                    let setting = CudnnSettingName::KernelId(kernel_id);

                    self.stmts.push(Statement::Raw(format!(
                        "{setting}.x = {input};",
                        setting = setting.setting(),
                        input = input,
                    )));
                    self.stmts.push(Statement::Raw(format!(
                        "{setting}.w = {weights};",
                        setting = setting.setting(),
                        weights = weights,
                    )));
                    self.stmts.push(Statement::Raw(format!(
                        "{setting}.y = {output};",
                        setting = setting.setting(),
                        output = output,
                    )));
                    let func = if let Some(bias) = kernel.inputs.get(args::CONV_BIAS).copied() {
                        self.stmts.push(Statement::Raw(format!(
                            "{setting}.bias = {bias};",
                            setting = setting.setting(),
                            bias = self.device_identifier(bias)?,
                        )));
                        "call_conv_bias_activation_forward"
                    } else {
                        "call_conv_forward"
                    };
                    self.stmts.push(Statement::Raw(format!(
                        "{setting}.{func}<{ty}>(&{ctx});",
                        setting = setting.setting(),
                        func = func,
                        ty = template_ty,
                        ctx = cudnn_handler.ctx()
                    )));
                }

                Operator::Gemm(Gemm {
                    alpha,
                    beta,
                    trans_a,
                    trans_b,
                }) => {
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
                        .entry(kernel_stream)
                        .or_insert_with(|| CublasHandler::new(kernel_stream));

                    // cuBLAS is column-major!
                    // We have t(A) and t(B), and want t(C).
                    // t(C) = t(A * B) = t(B) * t(A).
                    let (m, k, n) = {
                        let a_ty = self.get_resolved_tensor_type(kernel.inputs[0])?.clone();
                        let b_ty = self.get_resolved_tensor_type(kernel.inputs[1])?.clone();
                        assert!(a_ty.is_contiguous());
                        assert!(b_ty.is_contiguous());
                        let [m, k] = a_ty.dims[..] else {
                            panic!("Invalid GEMM input A shape");
                        };
                        let (m, k) = if *trans_a { (k, m) } else { (m, k) };
                        let [k_, n] = b_ty.dims[..] else {
                            panic!("Invalid GEMM input B shape");
                        };
                        let (k_, n) = if *trans_b { (n, k_) } else { (k_, n) };
                        assert!(k == k_);
                        (n, k, m)
                    };
                    let lda = if !*trans_b { m } else { k };
                    let ldb = if !*trans_a { k } else { n };
                    let ldc = m;
                    let (trans_a, trans_b) = {
                        let tmp0 = if *trans_b {
                            CublasOperation::Transpose
                        } else {
                            CublasOperation::Non
                        };

                        let tmp1 = if *trans_a {
                            CublasOperation::Transpose
                        } else {
                            CublasOperation::Non
                        };
                        (tmp0, tmp1)
                    };

                    // TODO: Copy bias into output chunk if bias_chunk != output_chunk.
                    if kernel.inputs.len() == 3 {
                        let bias_chunk = self.value2chunk.get(&kernel.inputs[2]).unwrap();
                        let output_chunk = self.value2chunk.get(&kernel.outputs[0]).unwrap();
                        assert!(bias_chunk == output_chunk);
                    }

                    let a = self.device_identifier(kernel.inputs[1])?.to_string();
                    let b = self.device_identifier(kernel.inputs[0])?.to_string();
                    let c = self.device_identifier(kernel.outputs[0])?.to_string();

                    self.stmts.push(
                        CublasApi::Gemm(GemmArgs {
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
                        })
                        .into(),
                    );
                }

                Operator::MaxPool(ref pool) => {
                    if pool.kernel_shape.ndim() != 2 {
                        unimplemented!("Only 2D max pooling is supported");
                    }

                    assert!(kernel.inputs.len() == 1);
                    assert!(kernel.outputs.len() == 1);
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ = self.device_identifier(kernel.inputs[0])?;
                    let input_ty = self.get_resolved_tensor_type(kernel.inputs[0])?;
                    let output_ty = self.get_resolved_tensor_type(kernel.outputs[0])?;
                    assert!(input_ty.dims.ndim() == 4 && output_ty.dims.ndim() == 4);
                    assert!(input_ty.is_contiguous() && output_ty.is_contiguous());
                    assert!(input_ty.dims[0] == output_ty.dims[0]);
                    assert!(input_ty.dims[1] == output_ty.dims[1]);
                    let nbatch = input_ty.dims[0].to_literal();
                    let channels = input_ty.dims[1].to_literal();
                    let height = input_ty.dims[2].to_literal();
                    let width = input_ty.dims[3].to_literal();
                    let o_height = output_ty.dims[2].to_literal();
                    let o_width = output_ty.dims[3].to_literal();
                    let kernel_h = pool.kernel_shape[0].to_literal();
                    let kernel_w = pool.kernel_shape[1].to_literal();
                    let stride_h = pool.strides[0].to_literal();
                    let stride_w = pool.strides[1].to_literal();
                    let (pad_h, pad_w) = match pool.pad {
                        ConvPad::NotSet(ref pad) => (pad[0].0, pad[1].0),
                        _ => unimplemented!("Padding type not implemented"),
                    };
                    let pad_h = pad_h.to_literal();
                    let pad_w = pad_w.to_literal();
                    let maxpool = kernel::MaxPoolKernel {
                        ty: input_ty.elem_type,
                        out,
                        in_,
                        nbatch,
                        channels,
                        height,
                        width,
                        o_height,
                        o_width,
                        kernel_h,
                        kernel_w,
                        stride_h,
                        stride_w,
                        pad_h,
                        pad_w,
                    };

                    let output_size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::MaxPoolKernel(maxpool),
                            output_size,
                        )?
                        .into(),
                    );
                }

                Operator::ReduceMatrix(op) => {
                    let input_ty = self.get_resolved_tensor_type(kernel.inputs[0])?;
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ = self.device_identifier(kernel.inputs[0])?;
                    let [row, col] = input_ty.dims[..] else {
                        panic!("Invalid ReduceMatrix output shape");
                    };
                    let block_size = min(DEFAULT_BLOCK_SIZE, ceil_pow2(col));
                    let grid_size = input_ty.dims[0].to_literal();
                    let data_ty = input_ty.elem_type;
                    let cuda_kernel = kernel::CUDAKernel::ReduceMatrixKernel(ReduceMatrixKernel {
                        data_ty,
                        reduce_ty: (*op).into(),
                        block_size,
                        out,
                        in_,
                        row,
                        col,
                    });
                    let shared_mem_bytes = Some(format!("{} * {}", block_size, SizeOf(data_ty)));
                    let block_size = block_size.to_literal();
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size,
                            block_size,
                            shared_mem_bytes,
                            stream_id: kernel_stream,
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
                    let input_size = self.get_resolved_tensor_type(kernel.inputs[0])?.dims.size();
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

                _ => unimplemented!("Kernel body not implemented: {:?}", op),
            },
            KernelBody::FusedElementWises(_) => {
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

        for output in kernel
            .outputs
            .iter()
            .filter(|v| self.schedule.outputs.contains(v))
        {
            let dst = self
                .hostmem2identifier
                .get(output)
                .ok_or(BuildError::NoHostVariable(*output))?;
            let src = self.device_identifier(*output)?;
            let mem_size = MemSize::Single(self.single_mem_size(*output)?);
            self.stmts.push(
                Memcpy {
                    dst: Expr::Identifier(dst.clone()),
                    src,
                    mem_size,
                    kind: CudaMemcpyKind::DeviceToHost,
                    stream: kernel_stream,
                }
                .into(),
            );
        }
        self.record_event(kernel_stream, &kernel.outputs);

        Ok(())
    }

    pub fn generate(&mut self, opt: &Options) -> Result<HostCode, BuildError> {
        let decl_values = self.gen_decl_values()?;
        let computes = self
            .gen_computes()?
            .into_iter()
            .filter(|stmt| match stmt {
                Statement::CudaRuntimeApi(CudaRuntimeApi::RecordEvent(RecordEvent {
                    event_id,
                    ..
                })) => self.used_event.contains(event_id),
                _ => true,
            })
            .collect();
        let finalize = self.gen_finalize()?;

        // NOTE: This must be called at the very end.
        let decl_cuda_objs = self.gen_decl_cuda_objs()?;

        let mut kernel_codes = Vec::new();
        std::mem::swap(&mut self.separated_codes, &mut kernel_codes);

        Ok(HostCode {
            decl_values,
            decl_cuda_objs,
            computes,
            finalize,
            kernel_codes,
            profile: opt.profile,
        })
    }
}

impl HostCode {
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for h in ["algorithm", "limits", "chrono", "iostream"] {
            writeln!(writer, "#include <{}>", h)?;
        }
        for h in [
            "common.cuh",
            "cuda.h",
            "cublas_v2.h",
            "cudnn.h",
            "pool.cuh",
            "reduce.cuh",
            "cudnn_setting.h",
        ] {
            writeln!(writer, "#include \"{}\"", h)?;
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

        let profile = if self.profile { "true" } else { "false" };
        writeln!(
            writer,
            "
extern \"C\" void model(void **{ARG_OUTPUT}, void **{ARG_INPUT}, void **{ARG_INITIALIZER}) {{
    auto timer_start = std::chrono::high_resolution_clock::now();
    "
        )?;
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
        writeln!(writer, "
    auto timer_end = std::chrono::high_resolution_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(timer_end - timer_start);
    if ({profile}) std::cout << \"Elapsed time: \" << elapsed_ms.count() << \" ms\" << std::endl;
}}
    "
)?;
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
            .join("models/test/operator")
            .join("conv.onnx");
        let model = Model::load_from_path(path).unwrap();
        let mut schedule =
            Schedule::new(model.graph, Options::builder().target(Target::CUDA).build());
        schedule.assign_mem();

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
}
