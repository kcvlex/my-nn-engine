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
use crate::codegen::cuda::kernel::ContiguousBuilder;
use crate::codegen::cuda::kernel::CopyBuilder;
use crate::codegen::cuda::kernel::ElementwiseKernelBuilder;
use crate::codegen::cuda::kernel::FuncQualifier;
use crate::codegen::cuda::kernel::GeneratedKernel;
use crate::codegen::cuda::kernel::KernelDecl;
use crate::codegen::cuda::kernel::KernelVar;
use crate::codegen::cuda::kernel::OneHotBuilder;
use crate::codegen::cuda::kernel::PoolBuilder;
use crate::codegen::cuda::kernel::ResizeBuilder;
use crate::codegen::cuda::kernel::SplitBuilder;
use crate::codegen::cuda::kernel::TypeSymbol;
use crate::codegen::cuda::kernel::WhereBuilder;
use crate::codegen::cuda::runtime_api::*;
use crate::graph::operator;
use crate::graph::operator::*;
use crate::graph::ValueId;
use crate::options::Options;
use crate::schedule::EventId;
use crate::schedule::StreamId;
use crate::schedule::*;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

#[derive(Debug, thiserror::Error)]
pub enum BuildError {
    #[error("unresolved type for value {0:?}")]
    UnresolvedType(ValueId),
    #[error("unresolved allocate info for kernel {0:?}")]
    UnresolvedAllocateInfo(KernelId),
    #[error("unexpected memory allocation for value {0:?}")]
    UnexpectedMemAlloc(ValueId),
    #[error("chunk not found for value {0:?}")]
    ChunkNotFound(ValueId),
    #[error("no host variable for value {0:?}")]
    NoHostVariable(ValueId),
    #[error("no device variable for chunk {0:?}")]
    NoDeviceVariable(ChunkId),
    #[error("event not found for value {0:?}")]
    EventNotFound(ValueId),
    #[error("activation not found for kernel {0:?}")]
    ActivationNotFound(KernelId),
    #[error("unsupported tensor dim for value {0:?}: {1}")]
    UnsupportedTensorDim(ValueId, usize),
    #[error("non-contiguous tensor for value {0:?}")]
    NonContiguousTensor(ValueId),
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DataType::Bool => "int8_t",
                DataType::SInt(SIntType::I8) => "int8_t",
                DataType::SInt(SIntType::I32) => "i32",
                DataType::SInt(SIntType::I64) => "i64",
                DataType::UInt(UIntType::U8) => "uint8_t",
                DataType::UInt(UIntType::U64) => "u64",
                DataType::Float(FloatType::BF16) => "__nv_bfloat16",
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

#[derive(Clone, PartialEq)]
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
#[allow(clippy::large_enum_variant)]
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

struct KernelStreamView {
    stream_id: StreamId,
    event_id: EventId,
    to_wait: Vec<EventId>,
}

pub struct HostCodeGenerator<'sched> {
    schedule: &'sched Schedule,
    cuda_arch: u32,

    stmts: Vec<Statement>,
    init_stmts: Vec<Statement>,
    destroy_stmts: Vec<Statement>,
    state_fields: Vec<String>,

    streams: HashMap<KernelId, KernelStreamView>,
    transfer_streams: HashMap<usize, KernelStreamView>,
    to_record_events: BTreeSet<EventId>,

    value2place: HashMap<ValueId, AllocPlace>,
    chunk_names: HashMap<ChunkId, String>,
    initializer_arg_idx: HashMap<ValueId, usize>,
    used_device_initializers: std::cell::RefCell<BTreeSet<ValueId>>,

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
const ARG_SESSION_STATE: &str = "session_state";
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
        let (_out_n, _out_c, out_h, out_w) = match conv.output_layout {
            Layout::NCHW => (0, 1, 2, 3),
            Layout::NHWC => (0, 3, 1, 2),
        };

        let has_bias = kernel
            .inputs
            .get(args::CONV_BIAS)
            .and_then(|x| *x)
            .is_some();
        let has_relu = conv.activation == Activation::ReLU;
        let x_is_nhwc = conv.input_layout == Layout::NHWC;
        let y_is_nhwc = conv.output_layout == Layout::NHWC;

        let (pad_h_pre, pad_w_pre, pad_h_post, pad_w_post) = match conv.pad {
            ConvPad::NotSet(ref pad) => (pad[0].0, pad[1].0, pad[0].1, pad[1].1),
            ConvPad::Valid => (0, 0, 0, 0),
            ConvPad::SameUpper | ConvPad::SameLower => {
                let calc = |dim: usize| {
                    let input = input_ty.dims[2 + dim];
                    let output = output_ty.dims[2 + dim];
                    let stride = conv.strides[dim];
                    let ext_len = stride * (output - 1) + weight_ty.dims[2 + dim];
                    let pad_total = ext_len - input;
                    let pad_pre = pad_total / 2 +
                        if matches!(conv.pad, ConvPad::SameLower) {
                            pad_total % 2
                        } else {
                            0
                        };
                    let pad_post = pad_total - pad_pre;
                    (pad_pre, pad_post)
                };
                let (ph_pre, ph_post) = calc(0);
                let (pw_pre, pw_post) = calc(1);
                (ph_pre, pw_pre, ph_post, pw_post)
            }
        };

        let data_type = input_ty.elem_type.cudnn();
        let n = input_ty.dims[in_n];
        let c = input_ty.dims[in_c];
        let h = input_ty.dims[in_h];
        let w = input_ty.dims[in_w];
        let k = weight_ty.dims[0];
        let r = weight_ty.dims[2];
        let s = weight_ty.dims[3];
        let out_h = output_ty.dims[out_h];
        let out_w = output_ty.dims[out_w];

        stmts.push(Statement::Raw(format!(
            "{ss}.build(&cudnn_handler_ctx, {data_type}, \
             {n}, {c}, {h}, {w}, {k}, {r}, {s}, {out_h}, {out_w}, \
             {pad_h_pre}, {pad_w_pre}, {pad_h_post}, {pad_w_post}, \
             {stride_h}, {stride_w}, {dil_h}, {dil_w}, \
             {groups}, {has_bias}, {has_relu}, {x_is_nhwc}, {y_is_nhwc});",
            ss = setting.setting(),
            stride_h = conv.strides[0],
            stride_w = conv.strides[1],
            dil_h = conv.dilations[0],
            dil_w = conv.dilations[1],
            groups = conv.group,
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
    pub fn new(schedule: &'sched Schedule, cuda_arch: u32) -> Self {
        let plan = schedule
            .execution_plan
            .as_ref()
            .expect("ExecutionPlan must be built before codegen");
        let mut streams: HashMap<KernelId, KernelStreamView> = HashMap::new();
        let mut transfer_streams: HashMap<usize, KernelStreamView> = HashMap::new();
        let mut to_record_events: BTreeSet<EventId> = BTreeSet::new();
        let mut pending_waits: Vec<EventId> = Vec::new();
        for (idx, step) in plan.steps.iter().enumerate() {
            match step {
                Step::SyncWait(SyncWaitStep { event, .. }) => {
                    to_record_events.insert(*event);
                    pending_waits.push(*event);
                }
                Step::Kernel(k) => {
                    streams.insert(
                        k.kernel,
                        KernelStreamView {
                            stream_id: k.context.stream,
                            event_id: k
                                .records_event
                                .expect("CUDA kernel step must have records_event"),
                            to_wait: std::mem::take(&mut pending_waits),
                        },
                    );
                }
                Step::Transfer(t) => {
                    transfer_streams.insert(
                        idx,
                        KernelStreamView {
                            stream_id: t.context.stream,
                            event_id: t
                                .records_event
                                .expect("CUDA transfer step must have records_event"),
                            to_wait: std::mem::take(&mut pending_waits),
                        },
                    );
                }
            }
        }
        HostCodeGenerator {
            schedule,
            cuda_arch,
            stmts: Vec::new(),
            init_stmts: Vec::new(),
            destroy_stmts: Vec::new(),
            state_fields: Vec::new(),
            streams,
            transfer_streams,
            to_record_events,
            value2place: HashMap::new(),
            chunk_names: HashMap::new(),
            initializer_arg_idx: HashMap::new(),
            used_device_initializers: std::cell::RefCell::new(BTreeSet::new()),
            cublas_handlers: IndexMap::new(),
            cudnn_ctxs: IndexMap::new(),
            separated_codes: Vec::new(),
            includes: BTreeSet::from([
                Include::Local("common.cuh"),
                Include::System("cuda.h"),
                Include::System("cuda_bf16.h"),
            ]),
        }
    }

    fn move_statements(&mut self) -> Vec<Statement> {
        let mut stmts = Vec::new();
        std::mem::swap(&mut self.stmts, &mut stmts);
        stmts
    }

    fn gen_decl_values(&mut self) -> Result<Vec<Statement>, BuildError> {
        for (arg_name, value_ids) in &[
            (ARG_INPUT, &self.schedule.inputs[..]),
            (ARG_OUTPUT, &self.schedule.outputs[..]),
        ] {
            for (idx, value) in value_ids.iter().enumerate() {
                let ty = self.get_resolved_tensor_type(*value)?.elem_type.to_string();
                let value_name = format!("h_{}_{}", arg_name, value.index());
                let stmt = format!("{ty} *{value_name} = ({ty} *)({arg_name}[{idx}]);");
                self.stmts.push(Statement::Raw(stmt));
            }
        }

        for (idx, value) in self.schedule.initializers.iter().enumerate() {
            self.initializer_arg_idx.insert(*value, idx);
        }

        for (idx, value) in self.schedule.session_states.iter().enumerate() {
            let _rty = self.get_resolved_tensor_type(*value)?;
            let device_name = format!("d_session_state_{}", value.index());
            self.state_fields
                .push(format!("void *{device_name} = nullptr;"));
            self.init_stmts.push(Statement::Raw(format!(
                "state->{device_name} = {ARG_SESSION_STATE}[{idx}];"
            )));
        }

        let plan = self
            .schedule
            .execution_plan
            .as_ref()
            .expect("ExecutionPlan must be built before codegen");
        for v in &self.schedule.inputs {
            self.value2place.insert(*v, AllocPlace::Input(*v));
        }
        for v in &self.schedule.outputs {
            self.value2place.insert(*v, AllocPlace::Output(*v));
        }
        for v in &self.schedule.initializers {
            self.value2place.insert(*v, AllocPlace::Initializer(*v));
        }
        for v in &self.schedule.session_states {
            self.value2place.insert(*v, AllocPlace::SessionState(*v));
        }
        for step in &plan.steps {
            match step {
                Step::Kernel(k) => {
                    for b in &k.bindings {
                        self.value2place.insert(b.value, b.place);
                    }
                }
                Step::Transfer(_) | Step::SyncWait(_) => {}
            }
        }

        for chunk in &plan.chunks {
            let name = format!("d_chunk_{}", chunk.id);
            self.chunk_names.insert(chunk.id, name);
        }

        for arena in &plan.arenas {
            if arena.size == 0 {
                continue;
            }
            let field = format!("d_arena_{}", arena.id);
            self.state_fields.push(format!("void *{field} = nullptr;"));
            match arena.tier {
                MemoryTier::GpuArena => {
                    self.init_stmts.push(
                        Malloc {
                            dst: Expr::Identifier(format!("state->{field}")),
                            mem_size: MemSize::Raw(Expr::Identifier(format!("{}", arena.size))),
                        }
                        .into(),
                    );
                    self.destroy_stmts
                        .push(Free(Expr::Identifier(format!("state->{field}"))).into());
                }
                MemoryTier::HostArena => {
                    unimplemented!("HostArena allocation not yet supported in CUDA codegen");
                }
            }
        }
        for chunk in &plan.chunks {
            let name = &self.chunk_names[&chunk.id];
            let offset = chunk.offset;
            let arena_id = chunk.arena;
            self.stmts.push(Statement::Raw(format!(
                "void *{name} = (char*)state->d_arena_{arena_id} + {offset};"
            )));
        }

        Ok(self.move_statements())
    }

    fn emit_used_initializers(&mut self) {
        let used: Vec<ValueId> = self
            .used_device_initializers
            .borrow()
            .iter()
            .copied()
            .collect();
        let mut emitted: BTreeSet<String> = BTreeSet::new();
        for value in used {
            let arg_idx = self.initializer_arg_idx[&value];
            let device_name = format!("d_init_{}", value.index());
            if !emitted.insert(device_name.clone()) {
                continue;
            }
            self.state_fields
                .push(format!("void *{device_name} = nullptr;"));
            self.init_stmts.push(Statement::Raw(format!(
                "state->{device_name} = {ARG_INITIALIZER}[{arg_idx}];"
            )));
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

    fn gen_computes(&mut self) -> Result<Vec<Statement>, BuildError> {
        let plan = self
            .schedule
            .execution_plan
            .as_ref()
            .expect("ExecutionPlan must be built before codegen");
        // Snapshot dispatch list so we don't borrow `plan` across the &mut self
        // emit calls.
        enum Op {
            Kernel(KernelId),
            Transfer(usize),
        }
        let ops: Vec<Op> = plan
            .steps
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| match s {
                Step::Kernel(k) => Some(Op::Kernel(k.kernel)),
                Step::Transfer(_) => Some(Op::Transfer(idx)),
                Step::SyncWait(_) => None,
            })
            .collect();
        for op in ops {
            match op {
                Op::Kernel(kid) => self.call_kernel(kid)?,
                Op::Transfer(idx) => self.emit_transfer(idx)?,
            }
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

        let active_streams: Vec<StreamId> = self
            .streams
            .values()
            .map(|s| s.stream_id)
            .chain(self.transfer_streams.values().map(|s| s.stream_id))
            .unique()
            .sorted()
            .collect();
        for stream_id in active_streams {
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
                self.init_stmts.push(
                    Malloc {
                        dst: Expr::Identifier(format!("{}.workspace", setting.state_setting())),
                        mem_size: MemSize::Raw(Expr::Identifier(format!(
                            "{}.workspace_size",
                            setting.state_setting()
                        ))),
                    }
                    .into(),
                );
                self.separated_codes.push(SeparatedCode::Cudnn(code));
            }

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
        let place = self
            .value2place
            .get(&value_id)
            .copied()
            .ok_or(BuildError::ChunkNotFound(value_id))?;
        match place {
            AllocPlace::Chunk(c) => self
                .chunk_names
                .get(&c)
                .map(|n| Expr::Identifier(n.clone()))
                .ok_or(BuildError::NoDeviceVariable(c)),
            AllocPlace::Initializer(src) => {
                self.used_device_initializers.borrow_mut().insert(src);
                Ok(Expr::Identifier(format!("state->d_init_{}", src.index())))
            }
            AllocPlace::SessionState(src) => Ok(Expr::Identifier(format!(
                "state->d_session_state_{}",
                src.index()
            ))),
            AllocPlace::Input(_) | AllocPlace::Output(_) => {
                panic!("device_identifier called for host place: {:?}", place)
            }
        }
    }

    fn host_name(&self, value_id: ValueId) -> String {
        let place = self
            .value2place
            .get(&value_id)
            .copied()
            .unwrap_or_else(|| panic!("host_name: no place for {value_id:?}"));
        match place {
            AllocPlace::Input(src) => format!("h_input_{}", src.index()),
            AllocPlace::Output(src) => format!("h_output_{}", src.index()),
            _ => panic!("host_name called for device place: {:?}", place),
        }
    }

    fn ring_args_from_slots(
        &self,
        kernel: &Kernel,
        sink_slot: usize,
        window_slot: usize,
        start_slot: usize,
    ) -> Option<kernel::RingArgs> {
        let sink_id = kernel.inputs.get(sink_slot).and_then(|x| *x)?;
        let window_id = kernel.inputs.get(window_slot).and_then(|x| *x)?;
        let start_id = kernel.inputs.get(start_slot).and_then(|x| *x)?;
        let host_int = |id| Expr::Identifier(format!("(int)(*{})", self.host_name(id)));
        Some(kernel::RingArgs {
            sink: host_int(sink_id),
            window: host_int(window_id),
            start: host_int(start_id),
        })
    }

    fn rope_attn_args_from_slots(
        &self,
        kernel: &Kernel,
        cos_slot: usize,
        sin_slot: usize,
        pos_slot: usize,
    ) -> Result<Option<kernel::RopeAttnArgs>, BuildError> {
        let cos_id = match kernel.inputs.get(cos_slot).and_then(|x| *x) {
            Some(id) => id,
            None => return Ok(None),
        };
        let sin_id = kernel.inputs.get(sin_slot).and_then(|x| *x);
        let pos_id = kernel.inputs.get(pos_slot).and_then(|x| *x);
        let (sin_id, pos_id) = match (sin_id, pos_id) {
            (Some(s), Some(p)) => (s, p),
            _ => panic!("Attention RoPE: cos+sin+kv_position must all be provided together"),
        };
        Ok(Some(kernel::RopeAttnArgs {
            cos_table: self.device_identifier(cos_id)?,
            sin_table: self.device_identifier(sin_id)?,
            kv_position: self.device_identifier(pos_id)?,
        }))
    }

    fn place_identifier(&self, place: AllocPlace) -> Result<Expr, BuildError> {
        match place {
            AllocPlace::Chunk(c) => self
                .chunk_names
                .get(&c)
                .map(|n| Expr::Identifier(n.clone()))
                .ok_or(BuildError::NoDeviceVariable(c)),
            AllocPlace::Initializer(src) => {
                self.used_device_initializers.borrow_mut().insert(src);
                Ok(Expr::Identifier(format!("state->d_init_{}", src.index())))
            }
            AllocPlace::SessionState(src) => Ok(Expr::Identifier(format!(
                "state->d_session_state_{}",
                src.index()
            ))),
            AllocPlace::Input(src) => Ok(Expr::Identifier(format!("h_input_{}", src.index()))),
            AllocPlace::Output(src) => Ok(Expr::Identifier(format!("h_output_{}", src.index()))),
        }
    }

    fn same_device_buffer(&self, a: ValueId, b: ValueId) -> bool {
        match (self.value2place.get(&a), self.value2place.get(&b)) {
            (Some(pa), Some(pb)) => pa == pb,
            _ => false,
        }
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
        // Dedup the outputs+inputs list by ValueId so each value appears as a
        // single kernel parameter. The same ValueId can appear in multiple
        // input slots (e.g. Concat(A, A) from RoPE, or Mul(x, x) from
        // canonicalized Pow(x, 2)) and nvcc rejects duplicate parameter names.
        let kernel = &self.schedule.kernels[kernel_id];
        let attribute_input = |idx: usize| -> bool {
            match &kernel.body {
                KernelBody::Opaque(Opaque { op }) => op.is_attribute_input(idx),
                KernelBody::ElementWises(_) => false,
            }
        };
        let mut seen = std::collections::HashSet::<ValueId>::new();
        let params = chain(
            kernel.outputs.iter().copied().map(Some),
            kernel
                .inputs
                .iter()
                .enumerate()
                .map(|(idx, inp)| inp.filter(|_| !attribute_input(idx))),
        )
        .flatten()
        .filter(|id| seen.insert(*id))
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

    fn emit_transfer(&mut self, step_idx: usize) -> Result<(), BuildError> {
        // Pull the step + stream view out into local copies so we can mutate
        // self below without overlapping borrows.
        let plan = self
            .schedule
            .execution_plan
            .as_ref()
            .expect("ExecutionPlan must be built before codegen");
        let Step::Transfer(t) = &plan.steps[step_idx] else {
            unreachable!()
        };
        let src_value = t.src.value;
        let dst_value = t.dst.value;
        let src_place = t.src.place;
        let dst_place = t.dst.place;

        let view = &self.transfer_streams[&step_idx];
        let stream_id = view.stream_id;
        let event_id = view.event_id;
        let to_wait = view.to_wait.clone();

        for event in to_wait {
            self.stmts.push(
                WaitEvent {
                    stream_id,
                    event_id: event,
                }
                .into(),
            );
        }

        let _ = src_value;
        let src_expr = self.place_identifier(src_place)?;
        let dst_expr = self.place_identifier(dst_place)?;
        let kind = match (src_place, dst_place) {
            (AllocPlace::Input(_), _) => CudaMemcpyKind::HostToDevice,
            (_, AllocPlace::Output(_)) => CudaMemcpyKind::DeviceToHost,
            other => panic!("unsupported Transfer place pair: {:?}", other),
        };
        let mem_size = MemSize::Single(self.single_mem_size(dst_value)?);
        self.stmts.push(
            Memcpy {
                dst: dst_expr,
                src: src_expr,
                mem_size,
                kind,
                stream: stream_id,
            }
            .into(),
        );

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

    fn call_kernel(&mut self, kernel_id: KernelId) -> Result<(), BuildError> {
        let kernel = &self.schedule.kernels[kernel_id];
        let KernelStreamView {
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
                Operator::Transfer(_) => unreachable!(
                    "Operator::Transfer should be lowered to Step::Transfer by the scheduler"
                ),

                Operator::Identity | Operator::Reinterpret(_) => {
                    let in_v = kernel.inputs[0].unwrap();
                    let out_v = kernel.outputs[0];
                    if !self.same_device_buffer(in_v, out_v) {
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
                Operator::And |
                Operator::BatchNormalization(_) |
                Operator::Cast(_) |
                Operator::Clip(_) |
                Operator::Cos |
                Operator::Div |
                Operator::Equal |
                Operator::Exp |
                Operator::GeLU(_) |
                Operator::IsNaN |
                Operator::LeakyReLU(_) |
                Operator::LessOrEqual |
                Operator::Log |
                Operator::Mul |
                Operator::Pow |
                Operator::Reciprocal |
                Operator::ReLU |
                Operator::Sigmoid |
                Operator::Sin |
                Operator::Sqrt |
                Operator::Sub |
                Operator::Swish(_) |
                Operator::Tanh => unreachable!(),

                Operator::Attention(attn) => {
                    self.includes.insert(Include::Local("attention.cuh"));
                    let q = kernel.inputs[args::ATTENTION_Q].unwrap();
                    let k = kernel.inputs[args::ATTENTION_K].unwrap();
                    let v = kernel.inputs[args::ATTENTION_V].unwrap();
                    let k_scale_id = kernel.inputs.get(args::ATTENTION_K_SCALE).and_then(|x| *x);
                    let v_scale_id = kernel.inputs.get(args::ATTENTION_V_SCALE).and_then(|x| *x);
                    let kv_quant = match (k_scale_id, v_scale_id) {
                        (Some(ks), Some(vs)) => Some(kernel::KvQuantArgs {
                            k_scale: self.device_identifier(ks)?,
                            v_scale: self.device_identifier(vs)?,
                        }),
                        (None, None) => None,
                        _ => panic!("Attention: K and V scale must be both present or both absent"),
                    };

                    let ring = self.ring_args_from_slots(
                        kernel,
                        args::ATTENTION_RING_SINK,
                        args::ATTENTION_RING_WINDOW,
                        args::ATTENTION_RING_START,
                    );
                    let rope = self.rope_attn_args_from_slots(
                        kernel,
                        args::ATTENTION_ROPE_COS,
                        args::ATTENTION_ROPE_SIN,
                        args::ATTENTION_ROPE_KV_POSITION,
                    )?;

                    let q_ty = self.get_resolved_tensor_type(q)?;
                    let k_ty = self.get_resolved_tensor_type(k)?;
                    let v_ty = self.get_resolved_tensor_type(v)?;

                    let q_dims = &q_ty.dims;
                    let k_dims = &k_ty.dims;
                    let v_dims = &v_ty.dims;
                    assert!(q_dims.ndim() == 4);
                    assert!(k_dims.ndim() == 4);
                    assert!(v_dims.ndim() == 4);
                    assert!(q_ty.is_contiguous());
                    assert!(k_ty.is_contiguous());
                    assert!(v_ty.is_contiguous());
                    assert!(q_dims[0] == k_dims[0] && k_dims[0] == v_dims[0]);
                    assert!(k_dims[1] == v_dims[1]);
                    assert!(
                        q_dims[1] % k_dims[1] == 0,
                        "Q heads ({}) must be a multiple of K/V heads ({})",
                        q_dims[1],
                        k_dims[1],
                    );
                    assert!(k_dims[2] == v_dims[2]);
                    assert!(q_dims[3] == k_dims[3] && k_dims[3] == v_dims[3]);

                    let seq_q = q_dims[2];
                    let seq_k = k_dims[2];

                    if seq_q == 1 && seq_k > 1 {
                        let batch_size = q_dims[0];
                        let num_q_heads = q_dims[1];
                        let num_kv_heads = k_dims[1];
                        let head_size = q_dims[3];
                        let block_size = DEFAULT_BLOCK_SIZE;
                        let grid_size = batch_size * num_q_heads;
                        let active_seq_kv_expr = if let Some(active_id) = kernel
                            .inputs
                            .get(args::ATTENTION_ACTIVE_SEQ_KV)
                            .and_then(|x| *x)
                        {
                            let host_name = self.host_name(active_id);
                            Expr::Identifier(format!("(int)(*{host_name})"))
                        } else {
                            seq_k.to_literal()
                        };
                        let cuda_kernel = kernel::CUDAKernel::AttentionDecodeKernel(
                            kernel::AttentionDecodeKernel {
                                data_ty: q_ty.elem_type,
                                head_dim: head_size,
                                block_size,
                                out: self.device_identifier(kernel.outputs[0])?,
                                q: self.device_identifier(q)?,
                                k: self.device_identifier(k)?,
                                v: self.device_identifier(v)?,
                                cache_seq_len: seq_k,
                                active_seq_kv: active_seq_kv_expr,
                                num_q_heads,
                                num_kv_heads,
                                kv_quant: kv_quant.clone(),
                                ring: ring.clone(),
                                rope: rope.clone(),
                                attn: *attn,
                            },
                        );
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
                    } else {
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
                        let num_q_heads = q_dims[1];
                        let num_kv_heads = k_dims[1];
                        let head_size = q_dims[3];
                        let threads_per_row = (head_size / 8).clamp(1, 32);
                        let br = ceil_pow2(seq_q / threads_per_row).clamp(1, 256 / threads_per_row);
                        let bc =
                            ceil_pow2(k_dims[2] / threads_per_row).clamp(1, 256 / threads_per_row);
                        let block_size = br.max(bc) * threads_per_row;
                        let grid_size = {
                            let y = batch_size * num_q_heads;
                            let x = seq_q.div_ceil(br);
                            format!("dim3({x}, {y})")
                        };

                        let kv_active_seq_expr = if let Some(active_id) = kernel
                            .inputs
                            .get(args::ATTENTION_ACTIVE_SEQ_KV)
                            .and_then(|x| *x)
                        {
                            let host_name = self.host_name(active_id);
                            Expr::Identifier(format!("(int)(*{host_name})"))
                        } else {
                            seq_k.to_literal()
                        };
                        let q_pos_offset_expr = if seq_q == seq_k {
                            "0".to_literal()
                        } else {
                            Expr::Identifier(format!("({kv_active_seq_expr} - {seq_q})"))
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
                            q_seq_len: seq_q,
                            kv_active_seq: kv_active_seq_expr,
                            kv_cache_stride: seq_k,
                            q_pos_offset: q_pos_offset_expr,
                            mask_outer_stride,
                            mask_row_stride,
                            num_q_heads,
                            num_kv_heads,
                            out: self.device_identifier(kernel.outputs[0])?,
                            kv_quant,
                            ring,
                            rope,
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
                }

                Operator::Concat(Concat { axis }) => {
                    self.includes.insert(Include::Local("concat.cuh"));
                    let output_ty = self.get_resolved_tensor_type(kernel.outputs[0])?.clone();
                    if !output_ty.is_contiguous() {
                        return Err(BuildError::NonContiguousTensor(kernel.outputs[0]));
                    }
                    let ndim = output_ty.dims.ndim();
                    let axis = axis.index(ndim);
                    let total_size = output_ty.dims.size();
                    let input_ids: Vec<ValueId> =
                        kernel.inputs.iter().map(|v| v.unwrap()).collect();
                    let input_tys = input_ids
                        .iter()
                        .map(|id| self.get_resolved_tensor_type(*id).cloned())
                        .collect::<Result<Vec<_>, _>>()?;
                    let tensor_sizes: Vec<usize> =
                        input_tys.iter().map(|ty| ty.dims.size()).collect();
                    let axis_sizes: Vec<usize> = input_tys.iter().map(|ty| ty.dims[axis]).collect();
                    let in_sizes_acc: Vec<usize> = std::iter::once(0)
                        .chain(tensor_sizes.iter().scan(0usize, |s, v| {
                            *s += v;
                            Some(*s)
                        }))
                        .take(input_ids.len())
                        .collect();
                    let in_axis_sizes_acc: Vec<usize> = std::iter::once(0)
                        .chain(axis_sizes.iter().scan(0usize, |s, v| {
                            *s += v;
                            Some(*s)
                        }))
                        .take(input_ids.len())
                        .collect();
                    let input_dims: Vec<Vec<usize>> = input_tys
                        .iter()
                        .map(|ty| ty.dims.iter().copied().collect())
                        .collect();
                    let input_strides: Vec<Vec<usize>> = input_tys
                        .iter()
                        .map(|ty| ty.strides().iter().copied().collect())
                        .collect();
                    let output_dims: Vec<usize> = output_ty.dims.iter().copied().collect();
                    let ins: Vec<Expr> = input_ids
                        .iter()
                        .map(|id| self.device_identifier(*id))
                        .collect::<Result<Vec<_>, _>>()?;
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let data_ty = output_ty.elem_type;
                    let cuda_kernel = kernel::CUDAKernel::ConcatKernel(kernel::ConcatKernel {
                        data_ty,
                        ndim,
                        n_inputs: input_ids.len(),
                        axis,
                        total_size,
                        out,
                        ins,
                        in_sizes_acc,
                        in_axis_sizes_acc,
                        input_dims,
                        input_strides,
                        output_dims,
                    });
                    self.stmts
                        .push(create_launch_kernel(cuda_kernel, total_size)?.into());
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
                    let setting = CudnnSettingName::KernelId(kernel_id);

                    let bias = if let Some(bias_id) =
                        kernel.inputs.get(args::CONV_BIAS).and_then(|x| *x)
                    {
                        self.device_identifier(bias_id)?.to_string()
                    } else {
                        "nullptr".to_string()
                    };
                    self.stmts.push(Statement::Raw(format!(
                        "{ss}.execute(&{ctx}, {input}, {weights}, {output}, {bias});",
                        ss = setting.state_setting(),
                        ctx = cudnn_handler.ctx(),
                    )));
                }

                Operator::DequantMatMul(DequantMatMul { axis }) => {
                    let lhs_ty = self
                        .get_resolved_tensor_type(kernel.inputs[args::DEQUANT_MATMUL_LHS].unwrap())?
                        .clone();
                    let rhs_ty = self
                        .get_resolved_tensor_type(kernel.inputs[args::DEQUANT_MATMUL_RHS].unwrap())?
                        .clone();
                    let scale_ty = self
                        .get_resolved_tensor_type(
                            kernel.inputs[args::DEQUANT_MATMUL_SCALE].unwrap(),
                        )?
                        .clone();
                    let axis_idx = axis.index(rhs_ty.dims.ndim());
                    assert_eq!(axis_idx, 0, "DequantMatMul kernel assumes axis=0");
                    let n = rhs_ty.dims[0];
                    let k = rhs_ty.dims[1];
                    let m = lhs_ty.dims.size() / k;
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let act =
                        self.device_identifier(kernel.inputs[args::DEQUANT_MATMUL_LHS].unwrap())?;
                    let wq =
                        self.device_identifier(kernel.inputs[args::DEQUANT_MATMUL_RHS].unwrap())?;
                    let scale =
                        self.device_identifier(kernel.inputs[args::DEQUANT_MATMUL_SCALE].unwrap())?;

                    let dtype_ok = matches!(lhs_ty.elem_type, DataType::Float(FloatType::BF16)) &&
                        matches!(scale_ty.elem_type, DataType::Float(FloatType::BF16));
                    // BF16 WMMA fragments require sm_80+ (Ampere); fall back on older arches.
                    // cp.async.cg requires 16-byte-aligned src addresses; with row stride K
                    // in bf16, that needs K % 8 == 0.
                    let k_aligned = k % 8 == 0;
                    let wmma_tile = if !dtype_ok || m < 16 || self.cuda_arch < 80 || !k_aligned {
                        None
                    } else if 128 <= m && 128 <= n {
                        Some((128usize, 128usize))
                    } else if 64 <= m && 64 <= n {
                        Some((64usize, 64usize))
                    } else if 32 <= m && 32 <= n {
                        Some((32usize, 32usize))
                    } else if 16 <= m && 32 <= n {
                        Some((16usize, 32usize))
                    } else {
                        Some((16usize, 16usize))
                    };

                    if let Some((bm, bn)) = wmma_tile {
                        self.includes
                            .insert(Include::Local("dequant_matmul_wmma.cuh"));
                        let warp_tile_m = if 128 <= bm { 32usize } else { 16usize };
                        let warp_tile_n = if 128 <= bn { 32usize } else { 16usize };
                        debug_assert!(bm % warp_tile_m == 0 && bn % warp_tile_n == 0);
                        let warps = (bm / warp_tile_m) * (bn / warp_tile_n);
                        let block_size = warps * 32;
                        let cuda_kernel = kernel::CUDAKernel::DequantMatMulWmmaKernel(
                            kernel::DequantMatMulWmmaKernel {
                                bm,
                                bn,
                                warp_tile_m,
                                warp_tile_n,
                                m,
                                n,
                                k,
                                out,
                                act,
                                wq,
                                scale,
                            },
                        );
                        let grid_x = n.div_ceil(bn);
                        let grid_y = m.div_ceil(bm);
                        self.stmts.push(
                            kernel::LaunchKernel {
                                cuda_kernel,
                                grid_size: Expr::Identifier(format!(
                                    "dim3({}, {}, 1)",
                                    grid_x, grid_y,
                                )),
                                block_size: block_size.to_literal(),
                                shared_mem_bytes: None,
                                stream_id,
                            }
                            .into(),
                        );
                    } else if m == 1 && dtype_ok && k % 16 == 0 {
                        // M=1 (decode): use specialized GEMV kernel.
                        self.includes.insert(Include::Local("dequant_gemv.cuh"));
                        let cuda_kernel =
                            kernel::CUDAKernel::DequantGemvKernel(kernel::DequantGemvKernel {
                                act_ty: lhs_ty.elem_type,
                                out_ty: scale_ty.elem_type,
                                n,
                                k,
                                out,
                                act,
                                wq,
                                scale,
                            });
                        self.stmts.push(
                            kernel::LaunchKernel {
                                cuda_kernel,
                                grid_size: Expr::Identifier(format!("dim3({}, 1, 1)", n)),
                                block_size: 32.to_literal(),
                                shared_mem_bytes: None,
                                stream_id,
                            }
                            .into(),
                        );
                    } else {
                        self.includes.insert(Include::Local("dequant_matmul.cuh"));
                        let block_size = DEFAULT_BLOCK_SIZE;
                        let cuda_kernel =
                            kernel::CUDAKernel::DequantMatMulKernel(kernel::DequantMatMulKernel {
                                act_ty: lhs_ty.elem_type,
                                out_ty: scale_ty.elem_type,
                                block_size,
                                m,
                                n,
                                k,
                                out,
                                act,
                                wq,
                                scale,
                            });
                        self.stmts.push(
                            kernel::LaunchKernel {
                                cuda_kernel,
                                grid_size: Expr::Identifier(format!("dim3({}, {}, 1)", n, m)),
                                block_size: block_size.to_literal(),
                                shared_mem_bytes: None,
                                stream_id,
                            }
                            .into(),
                        );
                    }
                }

                Operator::DequantizeLinear(DequantizeLinear { axis }) => {
                    self.includes.insert(Include::Local("dequantize.cuh"));
                    let x_ty = self
                        .get_resolved_tensor_type(kernel.inputs[args::DEQUANTIZE_X].unwrap())?
                        .clone();
                    let scale_ty = self
                        .get_resolved_tensor_type(kernel.inputs[args::DEQUANTIZE_SCALE].unwrap())?
                        .clone();
                    let axis_idx = axis.index(x_ty.dims.ndim());
                    let axis_dim = x_ty.dims[axis_idx];
                    let inner_size: usize = x_ty.dims[axis_idx + 1..].iter().product();
                    let total = x_ty.dims.size();
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let x = self.device_identifier(kernel.inputs[args::DEQUANTIZE_X].unwrap())?;
                    let scale =
                        self.device_identifier(kernel.inputs[args::DEQUANTIZE_SCALE].unwrap())?;
                    let cuda_kernel = kernel::CUDAKernel::DequantizeLinearKernel(
                        kernel::DequantizeLinearKernel {
                            in_ty: x_ty.elem_type,
                            out_ty: scale_ty.elem_type,
                            axis_dim,
                            inner_size,
                            total,
                            out,
                            x,
                            scale,
                        },
                    );
                    self.stmts
                        .push(create_launch_kernel(cuda_kernel, total)?.into());
                }

                Operator::Gather(gather) => {
                    self.includes.insert(Include::Local("gather.cuh"));
                    if gather.axis.raw() != 0 {
                        unimplemented!("Gather: only axis=0 is supported");
                    }
                    let input_ty = self
                        .get_resolved_tensor_type(kernel.inputs[0].unwrap())?
                        .clone();
                    let indices_ty = self
                        .get_resolved_tensor_type(kernel.inputs[1].unwrap())?
                        .clone();
                    let axis_dim = input_ty.dims[0];
                    let repeat = input_ty.dims.size() / axis_dim;
                    let size = indices_ty.dims.size();
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ = self.device_identifier(kernel.inputs[0].unwrap())?;
                    let indices = self.device_identifier(kernel.inputs[1].unwrap())?;
                    let cuda_kernel = kernel::CUDAKernel::GatherKernel(kernel::GatherKernel {
                        data_ty: input_ty.elem_type,
                        idx_ty: indices_ty.elem_type,
                        axis_dim,
                        repeat,
                        size,
                        out,
                        in_,
                        indices,
                    });
                    self.stmts
                        .push(create_launch_kernel(cuda_kernel, size)?.into());
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
                    let scalar_ty = match elem_ty {
                        DataType::Float(FloatType::BF16) => DataType::Float(FloatType::F32),
                        other => other,
                    };
                    let scalar_ty_str = scalar_ty.to_string();
                    let alpha = {
                        let var_name = format!("alpha_{}", kernel_id.index());
                        self.stmts.push(Statement::Raw(format!(
                            "{ty} {name} = {value};",
                            ty = scalar_ty_str,
                            name = var_name,
                            value = alpha,
                        )));
                        var_name
                    };
                    let beta = {
                        let var_name = format!("beta_{}", kernel_id.index());
                        self.stmts.push(Statement::Raw(format!(
                            "{ty} {name} = {value};",
                            ty = scalar_ty_str,
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

                    if kernel.inputs.len() == 3 {
                        assert!(matches!(op, Operator::Gemm(_)));
                        let bias_id = kernel.inputs[args::GEMM_C].unwrap();
                        let bias_dev = self.device_identifier(bias_id)?;
                        let output_dev = self.device_identifier(kernel.outputs[0])?;
                        if bias_dev != output_dev {
                            let mem_size =
                                MemSize::Single(self.single_mem_size(kernel.outputs[0])?);
                            self.stmts.push(
                                Memcpy {
                                    dst: output_dev.clone(),
                                    src: bias_dev,
                                    mem_size,
                                    kind: CudaMemcpyKind::DeviceToDevice,
                                    stream: stream_id,
                                }
                                .into(),
                            );
                        }
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

                Operator::QuantizingKVCacheUpdate => {
                    self.includes
                        .insert(Include::Local("quantizing_kvcache_update.cuh"));
                    let cache_id = kernel.inputs[args::QKVCACHE_UPDATE_CACHE].unwrap();
                    let scale_id = kernel.inputs[args::QKVCACHE_UPDATE_SCALE].unwrap();
                    let new_id = kernel.inputs[args::QKVCACHE_UPDATE_NEW].unwrap();
                    let offset_id = kernel.inputs[args::QKVCACHE_UPDATE_OFFSET].unwrap();

                    let cache_ty = self.get_resolved_tensor_type(cache_id)?;
                    let scale_ty = self.get_resolved_tensor_type(scale_id)?;
                    let new_ty = self.get_resolved_tensor_type(new_id)?;

                    assert!(cache_ty.is_contiguous());
                    assert!(new_ty.is_contiguous());
                    assert_eq!(cache_ty.dims.ndim(), 4);
                    assert_eq!(new_ty.dims.ndim(), 4);
                    assert_eq!(scale_ty.dims.ndim(), 3);
                    let batch = cache_ty.dims[0];
                    let heads = cache_ty.dims[1];
                    let max_seq_len = cache_ty.dims[2];
                    let head_dim = cache_ty.dims[3];
                    let new_seq_len = new_ty.dims[2];

                    let offset_host = self.host_name(offset_id);
                    let offset_expr = Expr::Identifier(format!("(int)(*{})", offset_host));

                    let block_size = DEFAULT_BLOCK_SIZE.min(head_dim).max(32);
                    let cuda_kernel = kernel::CUDAKernel::QuantizingKVCacheUpdateKernel(
                        kernel::QuantizingKVCacheUpdateKernel {
                            new_ty: new_ty.elem_type,
                            scale_ty: scale_ty.elem_type,
                            head_dim,
                            block_size,
                            max_seq_len,
                            new_seq_len,
                            cache: self.device_identifier(kernel.outputs[0])?,
                            scale: self.device_identifier(scale_id)?,
                            new_kv: self.device_identifier(new_id)?,
                            offset: offset_expr,
                            ring: self.ring_args_from_slots(
                                kernel,
                                args::QKVCACHE_UPDATE_RING_SINK,
                                args::QKVCACHE_UPDATE_RING_WINDOW,
                                args::QKVCACHE_UPDATE_RING_START,
                            ),
                        },
                    );
                    let grid = batch * heads * new_seq_len;
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size: grid.to_literal(),
                            block_size: block_size.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
                }

                Operator::KVCacheUpdate => {
                    self.includes.insert(Include::Local("kvcache_update.cuh"));
                    let cache_id = kernel.inputs[args::KVCACHE_UPDATE_CACHE].unwrap();
                    let new_id = kernel.inputs[args::KVCACHE_UPDATE_NEW].unwrap();
                    let offset_id = kernel.inputs[args::KVCACHE_UPDATE_OFFSET].unwrap();

                    let cache_ty = self.get_resolved_tensor_type(cache_id)?;
                    let new_ty = self.get_resolved_tensor_type(new_id)?;

                    assert!(cache_ty.is_contiguous());
                    assert!(new_ty.is_contiguous());
                    assert_eq!(cache_ty.dims.ndim(), 4);
                    assert_eq!(new_ty.dims.ndim(), 4);
                    assert_eq!(cache_ty.dims[0], new_ty.dims[0]);
                    assert_eq!(cache_ty.dims[1], new_ty.dims[1]);
                    assert_eq!(cache_ty.dims[3], new_ty.dims[3]);

                    let batch_heads = cache_ty.dims[0] * cache_ty.dims[1];
                    let cache_seq_len = cache_ty.dims[2];
                    let new_seq_len = new_ty.dims[2];
                    let head_dim = cache_ty.dims[3];

                    let offset_host = self.host_name(offset_id);
                    let offset_expr = Expr::Identifier(format!("(int)(*{})", offset_host));

                    let block_size = DEFAULT_BLOCK_SIZE.min(new_seq_len * head_dim).max(32);
                    let cuda_kernel =
                        kernel::CUDAKernel::KVCacheUpdateKernel(kernel::KVCacheUpdateKernel {
                            data_ty: cache_ty.elem_type,
                            head_dim,
                            cache_seq_len,
                            new_seq_len,
                            cache: self.device_identifier(kernel.outputs[0])?,
                            new_kv: self.device_identifier(new_id)?,
                            offset: offset_expr,
                            ring: self.ring_args_from_slots(
                                kernel,
                                args::KVCACHE_UPDATE_RING_SINK,
                                args::KVCACHE_UPDATE_RING_WINDOW,
                                args::KVCACHE_UPDATE_RING_START,
                            ),
                        });
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size: batch_heads.to_literal(),
                            block_size: block_size.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
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

                Operator::RMSNormalization(RMSNormalization { axis, epsilon }) => {
                    self.includes.insert(Include::Local("rms_norm.cuh"));
                    let input_ty = self
                        .get_resolved_tensor_type(
                            kernel.inputs[operator::args::RMS_NORM_DATA].unwrap(),
                        )?
                        .clone();
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ =
                        self.device_identifier(kernel.inputs[args::RMS_NORM_DATA].unwrap())?;
                    let scale =
                        self.device_identifier(kernel.inputs[args::RMS_NORM_SCALE].unwrap())?;
                    let axis = axis.index(input_ty.dims.ndim());
                    let epsilon = *epsilon;
                    if input_ty.strides().last() != Some(&1) || axis != input_ty.dims.ndim() - 1 {
                        unimplemented!(
                            "RMSNormalization currently supports only last-axis normalization"
                        );
                    }
                    let axis_dim = input_ty.dims[axis];
                    let size = input_ty.dims.size();
                    let block_size = min(DEFAULT_BLOCK_SIZE, ceil_pow2(axis_dim));
                    let grid_size = size / axis_dim;
                    let data_ty = input_ty.elem_type;
                    let cuda_kernel = kernel::CUDAKernel::RMSNormKernel(kernel::RMSNormKernel {
                        data_ty,
                        block_size,
                        axis_dim,
                        out,
                        in_,
                        scale,
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

                Operator::Rope(operator::Rope { head_dim }) => {
                    self.includes.insert(Include::Local("rope.cuh"));
                    let head_dim = *head_dim;
                    let x_id = kernel.inputs[args::ROPE_X].unwrap();
                    let cos_id = kernel.inputs[args::ROPE_COS].unwrap();
                    let sin_id = kernel.inputs[args::ROPE_SIN].unwrap();
                    let pos_id = kernel.inputs[args::ROPE_POSITION].unwrap();
                    let x_ty = self.get_resolved_tensor_type(x_id)?.clone();
                    let pos_ty = self.get_resolved_tensor_type(pos_id)?.clone();
                    assert!(x_ty.is_contiguous());
                    let ndim = x_ty.dims.ndim();
                    assert!(2 <= ndim);
                    assert_eq!(x_ty.dims[ndim - 1], head_dim);
                    let seq_len = x_ty.dims[ndim - 2];
                    assert_eq!(pos_ty.dims.size(), seq_len);
                    let outer = x_ty.dims.size() / (seq_len * head_dim);
                    let half = head_dim / 2;
                    let cuda_kernel = kernel::CUDAKernel::RopeKernel(kernel::RopeKernel {
                        data_ty: x_ty.elem_type,
                        head_dim,
                        out: self.device_identifier(kernel.outputs[0])?,
                        x: self.device_identifier(x_id)?,
                        cos_table: self.device_identifier(cos_id)?,
                        sin_table: self.device_identifier(sin_id)?,
                        position: self.device_identifier(pos_id)?,
                        seq_len,
                    });
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
                            grid_size: (outer * seq_len).to_literal(),
                            block_size: half.to_literal(),
                            shared_mem_bytes: None,
                            stream_id,
                        }
                        .into(),
                    );
                }

                Operator::AveragePool(_) | Operator::MaxPool(_) => {
                    let size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        PoolBuilder::new(sched, decl).build()
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

                Operator::ReduceMatrix(reduce_op) => {
                    self.includes.insert(Include::Local("reduce.cuh"));
                    let input_id = kernel.inputs[0].unwrap();
                    let input_ty = self.get_resolved_tensor_type(input_id)?.clone();
                    assert!(input_ty.is_contiguous());
                    let [row, col] = input_ty.dims[..] else {
                        panic!("Invalid ReduceMatrix output shape");
                    };
                    let block_size = min(DEFAULT_BLOCK_SIZE, ceil_pow2(col));
                    let grid_size = row.to_literal();
                    let out = self.device_identifier(kernel.outputs[0])?;
                    let in_ = self.device_identifier(input_id)?;
                    let cuda_kernel =
                        kernel::CUDAKernel::ReduceMatrixKernel(kernel::ReduceMatrixKernel {
                            data_ty: input_ty.elem_type,
                            op: *reduce_op,
                            block_size,
                            col,
                            out,
                            in_,
                        });
                    self.stmts.push(
                        kernel::LaunchKernel {
                            cuda_kernel,
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

                Operator::Where => {
                    let output_size = self
                        .get_resolved_tensor_type(kernel.outputs[0])?
                        .dims
                        .size();
                    let generated = self.generate_kernel(kernel_id, |sched, decl| {
                        WhereBuilder::new(sched, decl).build()
                    })?;
                    self.stmts.push(
                        create_launch_kernel(
                            kernel::CUDAKernel::GeneratedKernel(generated),
                            output_size,
                        )?
                        .into(),
                    );
                }

                Operator::Expand => {
                    self.includes.insert(Include::Local("expand.cuh"));
                    let input_id = kernel.inputs[0].unwrap();
                    let output_id = kernel.outputs[0];
                    let output_ty = self.get_resolved_tensor_type(output_id)?.clone();
                    let input_ty = self.get_resolved_tensor_type(input_id)?.clone();
                    let ndim = output_ty.dims.ndim();
                    let size = output_ty.dims.size();
                    let src_bc = input_ty.broadcast(&output_ty.dims);
                    let output_dims: Vec<usize> = output_ty.dims.iter().copied().collect();
                    let input_strides: Vec<usize> = (0..ndim).map(|i| src_bc.stride(i)).collect();
                    let out = self.device_identifier(output_id)?;
                    let in_ = self.device_identifier(input_id)?;
                    let cuda_kernel = kernel::CUDAKernel::ExpandKernel(kernel::ExpandKernel {
                        data_ty: output_ty.elem_type,
                        ndim,
                        size,
                        output_dims,
                        input_strides,
                        out,
                        in_,
                    });
                    self.stmts
                        .push(create_launch_kernel(cuda_kernel, size)?.into());
                }

                Operator::Slice => {
                    self.includes.insert(Include::Local("slice.cuh"));
                    let input_id = kernel.inputs[args::SLICE_DATA].unwrap();
                    let output_id = kernel.outputs[0];
                    let output_ty = self.get_resolved_tensor_type(output_id)?.clone();
                    let input_ty = self.get_resolved_tensor_type(input_id)?.clone();
                    let slices =
                        operator::Slice::collect_from_inputs(self.schedule.graph(), &kernel.inputs)
                            .expect("Failed to collect slices");
                    let ndim = output_ty.dims.ndim();
                    assert_eq!(ndim, input_ty.dims.ndim());
                    let mut start_per_axis = vec![0isize; ndim];
                    for slice in slices.iter() {
                        if slice.step != 1 {
                            panic!("Slice: only step=1 is supported, got step={}", slice.step);
                        }
                        start_per_axis[slice.axis] = slice.start;
                    }
                    let base_offset: isize = (0..ndim)
                        .map(|i| start_per_axis[i] * input_ty.stride(i) as isize)
                        .sum();
                    let size = output_ty.dims.size();
                    let output_dims: Vec<usize> = output_ty.dims.iter().copied().collect();
                    let input_strides: Vec<usize> = (0..ndim).map(|i| input_ty.stride(i)).collect();
                    let out = self.device_identifier(output_id)?;
                    let in_ = self.device_identifier(input_id)?;
                    let cuda_kernel = kernel::CUDAKernel::SliceKernel(kernel::SliceKernel {
                        data_ty: output_ty.elem_type,
                        ndim,
                        size,
                        base_offset,
                        output_dims,
                        input_strides,
                        out,
                        in_,
                    });
                    self.stmts
                        .push(create_launch_kernel(cuda_kernel, size)?.into());
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
        self.emit_used_initializers();
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
                destroy_body.push(
                    Free(Expr::Identifier(format!(
                        "{}.workspace",
                        setting.state_setting()
                    )))
                    .into(),
                );
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
            r#"extern "C" void* model_init(void **{ARG_INITIALIZER}, void **{ARG_SESSION_STATE}) {{
  auto *state = new ModelState;
  (void){ARG_SESSION_STATE};
{init_body}
  return state;
}}

extern "C" void model_destroy(void *ptr) {{
  auto *state = static_cast<ModelState*>(ptr);
{destroy_body}
  delete state;
}}

extern "C" void model(void *state_ptr, void **{ARG_OUTPUT}, void **{ARG_INPUT}) {{
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
    use crate::onnx::Model;
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

        let mut host_gen = HostCodeGenerator::new(&schedule, 80);
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
        crate::transform::transform_graph(
            &mut graph,
            &options,
            &crate::session::SessionConfig::default(),
        );
        let mut schedule = Schedule::new(graph, options.clone());
        let schedule_passes = crate::schedule::create_schedule_passes(&options);
        schedule_passes.run(&mut schedule);

        let mut host_gen = HostCodeGenerator::new(&schedule, 80);
        let code = host_gen.generate(&options).unwrap();
        let mut buf = Vec::new();
        code.write(&mut buf).unwrap();
        String::from_utf8(buf).unwrap()
    }

    #[test]
    fn test_cuda_two_conv_codegen() {
        let code = generate_cuda_code("models/test/single_op/two_conv/model.onnx");
        insta::assert_snapshot!(code);
    }
}
