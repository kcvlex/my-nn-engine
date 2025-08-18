mod kernel;
mod runtime_api;

use crate::codegen::cuda::runtime_api::*;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::types::ResolvedTensorType;
use crate::{
    schedule::*,
    tensor::types::{DataType, FloatType, SIntType, UIntType},
};
use delegate::delegate;
use derive_more::From;
use indexmap::{IndexMap, IndexSet};
use itertools::Itertools;
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub enum BuildError {
    UnresolvedType(ValueId),
    UnresolvedAllocateInfo(KernelId),
    UnexpectedMemAlloc(ValueId),
    ChunkNotFound(ValueId),
    NoHostVariable(ValueId),
    NoDeviceVariable(ChunkId),
    EventNotFound(ValueId),
}

impl DataType {
    fn fragment(&self) -> &'static str {
        match self {
            DataType::SInt(SIntType::I32) => "i32",
            DataType::SInt(SIntType::I64) => "i64",
            DataType::UInt(UIntType::U64) => "u64",
            DataType::Float(FloatType::F32) => "float",
            DataType::Float(FloatType::F64) => "double",
        }
    }
}

enum MemSize {
    Single(SingleMemSize),
    Chunk(ChunkMemSize),
}

impl MemSize {
    delegate! {
        to match self {
            MemSize::Single(size) => size,
            MemSize::Chunk(size) => size,
        } {
            fn fragment(&self) -> String;
        }
    }
}

#[derive(Clone, Copy, Default)]
struct SingleMemSize {
    ty: DataType,
    elem_num: usize,
}

impl SingleMemSize {
    fn fragment(&self) -> String {
        if self.elem_num == 0 {
            "0".to_owned()
        } else {
            format!("{} * sizeof({})", self.elem_num, self.ty.fragment())
        }
    }
}

impl From<&ResolvedTensorType> for SingleMemSize {
    fn from(ty: &ResolvedTensorType) -> Self {
        SingleMemSize {
            ty: ty.elem_type,
            elem_num: ty.dims.size(),
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

    fn fragment(&self) -> String {
        let res = self
            .sizes
            .iter()
            .filter(|s| 0 < s.elem_num)
            .map(|s| s.fragment())
            .join(", ");
        if res.is_empty() {
            "0".to_owned()
        } else {
            format!("std::max({{ {} }})", res)
        }
    }
}

enum Expr {
    Identifier(String),
    Literal(String),
}

impl Expr {
    fn fragment(&self) -> String {
        match self {
            Expr::Identifier(name) => name.clone(),
            Expr::Literal(lit) => lit.clone(),
        }
    }
}

trait ToIdentifier {
    fn to_identifier(&self) -> Expr;
}

impl ToIdentifier for String {
    fn to_identifier(&self) -> Expr {
        Expr::Identifier(self.clone())
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

impl ToIdentifier for StreamId {
    fn to_identifier(&self) -> Expr {
        Expr::Identifier(format!("stream_{}", self.0))
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct EventId(usize);

impl ToIdentifier for EventId {
    fn to_identifier(&self) -> Expr {
        Expr::Identifier(format!("event_{}", self.0))
    }
}

#[derive(From)]
enum Statement {
    LaunchKernel(kernel::LaunchKernel),
    CudaRuntimeApi(CudaRuntimeApi),
    Raw(String),
}

impl Statement {
    fn fragment(&self) -> String {
        match self {
            // TODO: Error handling for kernel launch
            Statement::LaunchKernel(kernel) => format!("{};", kernel.fragment()),
            Statement::CudaRuntimeApi(api) => format!("cudaCheckErr({});", api.fragment()),
            Statement::Raw(stmt) => stmt.clone(),
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
}

pub struct HostCode {
    decl_values: Vec<Statement>,
    decl_cuda_objs: Vec<Statement>,
    computes: Vec<Statement>,
    finalize: Vec<Statement>,
}

const ARG_INPUT: &str = "input";
const ARG_OUTPUT: &str = "output";
const ARG_INITIALIZER: &str = "initializer";
const MAX_STREAMS: usize = 16;
const DEFAULT_BLOCK_SIZE: usize = 256;

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
                let ty = self
                    .schedule
                    .get_resolved_tensor_type(*value)
                    .ok_or(BuildError::UnresolvedType(*value))?
                    .elem_type
                    .fragment();
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

                let size = self
                    .schedule
                    .get_resolved_tensor_type(mem.value_id)
                    .ok_or(BuildError::UnresolvedType(mem.value_id))?
                    .into();
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

    fn gen_computes(&mut self) -> Result<Vec<Statement>, BuildError> {
        for (kernel_id, _) in self.schedule.kernels.iter() {
            self.call_kernel(kernel_id)?;
        }
        Ok(self.move_statements())
    }

    fn gen_decl_cuda_objs(&mut self) -> Result<Vec<Statement>, BuildError> {
        for event_id in self.used_event.iter() {
            let name = event_id.to_identifier().fragment();
            self.stmts
                .push(Statement::Raw(format!("cudaEvent_t {name};")));
        }
        for stream_id in self.streams.inner.iter() {
            let name = stream_id.to_identifier().fragment();
            self.stmts
                .push(Statement::Raw(format!("cudaStream_t {name};")));
        }

        for event_id in self.used_event.iter().copied() {
            self.stmts.push(EventCreate { event_id }.into());
        }
        for stream_id in self.streams.inner.iter().copied() {
            self.stmts.push(StreamCreate { stream_id }.into());
        }
        Ok(self.move_statements())
    }

    fn gen_finalize(&mut self) -> Result<Vec<Statement>, BuildError> {
        let mut output_events = IndexMap::new();
        for value_id in self.schedule.outputs.iter().copied() {
            let event_id = self
                .value2event
                .get(&value_id)
                .copied()
                .ok_or(BuildError::EventNotFound(value_id))?;
            output_events
                .entry(event_id)
                .or_insert_with(Vec::new)
                .push(value_id);
        }

        let mut events_sync = Vec::with_capacity(output_events.len());
        for (event_id, values) in output_events.iter() {
            let stream_id = self.event2stream[event_id.0];
            for value_id in values.iter().copied() {
                let dst = self
                    .hostmem2identifier
                    .get(&value_id)
                    .ok_or(BuildError::NoHostVariable(value_id))?;
                let src = self.device_identifier(value_id)?;
                let mem_size = MemSize::Single(self.single_mem_size(value_id)?);
                self.stmts.push(
                    Memcpy {
                        dst: Expr::Identifier(dst.clone()),
                        src,
                        mem_size,
                        kind: CudaMemcpyKind::DeviceToHost,
                        stream: stream_id,
                    }
                    .into(),
                );
            }
            let new_event = self.record_event(stream_id, values);
            events_sync.push(new_event);
        }

        for event_id in events_sync.iter().copied() {
            self.stmts.push(EventSynchronize { event_id }.into());
            self.used_event.insert(event_id);
        }
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

        // Verify
        for info in copy.iter() {
            assert!(info.is_first_use);
        }

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

        // Launch the kernel
        match kernel.body {
            KernelBody::SingleKernel(SingleKernel {
                op: Operator::MaxPool(ref pool),
            }) => {
                if pool.kernel_shape.ndim() != 2 {
                    unimplemented!("Only 2D max pooling is supported");
                }

                assert!(kernel.inputs.len() == 1);
                assert!(kernel.outputs.len() == 1);
                let out = self.device_identifier(kernel.outputs[0])?;
                let in_ = self.device_identifier(kernel.inputs[0])?;
                let input_ty = self
                    .schedule
                    .get_resolved_tensor_type(kernel.inputs[0])
                    .ok_or(BuildError::UnresolvedType(kernel.inputs[0]))?;
                let output_ty = self
                    .schedule
                    .get_resolved_tensor_type(kernel.outputs[0])
                    .ok_or(BuildError::UnresolvedType(kernel.outputs[0]))?;
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
                let output_size = output_ty.dims.size();
                let block_size = DEFAULT_BLOCK_SIZE.to_literal();
                let grid_size = output_size.div_ceil(DEFAULT_BLOCK_SIZE).to_literal();
                self.stmts.push(
                    kernel::LaunchKernel {
                        cuda_kernel: kernel::CUDAKernel::MaxPoolKernel(maxpool),
                        grid_size,
                        block_size,
                        shared_mem_bytes: None,
                        stream_id: kernel_stream,
                    }
                    .into(),
                );
            }
            _ => unimplemented!("Kernel body not implemented"),
        }

        self.record_event(kernel_stream, &kernel.outputs);

        Ok(())
    }

    pub fn generate(&mut self) -> Result<HostCode, BuildError> {
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

        Ok(HostCode {
            decl_values,
            decl_cuda_objs,
            computes,
            finalize,
        })
    }
}

impl HostCode {
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        for h in ["algorithm", "limits"] {
            writer.write_all(format!("#include <{}>\n", h).as_bytes())?;
        }
        for h in ["common.h", "cuda.h", "pool.cu"] {
            writer.write_all(format!("#include \"{}\"\n", h).as_bytes())?;
        }

        writer.write_all(format!("extern \"C\" void model(void **{ARG_OUTPUT}, void **{ARG_INPUT}, void **{ARG_INITIALIZER}) {{\n").as_bytes())?;
        for stmts in &[
            self.decl_values.as_slice(),
            self.decl_cuda_objs.as_slice(),
            self.computes.as_slice(),
            self.finalize.as_slice(),
        ] {
            for stmt in stmts.iter() {
                writer.write_all(b"  ")?;
                writer.write_all(stmt.fragment().as_bytes())?;
                writer.write_all(b"\n")?;
            }
        }
        writer.write_all(b"}\n")?;
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
        use std::io::prelude::*;
        use std::io::BufWriter;
        use std::path::PathBuf;

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/operator")
            .join("maxpool.onnx");
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
        let code = host_gen.generate().unwrap();
        code.write(&mut file).unwrap();
        println!("Generated CUDA code written to {:?}", path);
    }
}
