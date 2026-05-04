use crate::graph::ValueId;
use crate::schedule::ChunkId;
use crate::schedule::KernelId;

pub type ArenaId = usize;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Ord, PartialOrd)]
pub struct StreamId(pub usize);

impl StreamId {
    pub fn index(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for StreamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stream_{}", self.0)
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Ord, PartialOrd)]
pub struct EventId(pub usize);

impl EventId {
    pub fn index(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "event_{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    CPU,
    CUDA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecutionContext {
    pub device: Device,
    pub stream: StreamId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MemoryTier {
    GpuArena,
    HostArena,
}

#[derive(Debug, Clone, Copy)]
pub struct ArenaInfo {
    pub id: ArenaId,
    pub tier: MemoryTier,
    pub size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ChunkInfo {
    pub id: ChunkId,
    pub arena: ArenaId,
    pub size: usize,
    pub offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EventInfo {
    pub id: EventId,
    pub stream: StreamId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocPlace {
    Chunk(ChunkId),
    SessionState(ValueId),
    Initializer(ValueId),
    Input(ValueId),
    Output(ValueId),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingRole {
    Input,
    Output,
    Workspace,
}

#[derive(Debug, Clone, Copy)]
pub struct ValueBinding {
    pub value: ValueId,
    pub role: BindingRole,
    pub place: AllocPlace,
    pub is_first_use: bool,
}

#[derive(Debug, Clone)]
pub struct KernelStep {
    pub kernel: KernelId,
    pub context: ExecutionContext,
    pub bindings: Vec<ValueBinding>,
    pub records_event: Option<EventId>,
}

#[derive(Debug, Clone)]
pub struct SyncWaitStep {
    pub context: ExecutionContext,
    pub event: EventId,
}

#[derive(Debug, Clone)]
pub struct TransferStep {
    pub src: ValueBinding,
    pub dst: ValueBinding,
    pub context: ExecutionContext,
    pub records_event: Option<EventId>,
}

#[derive(Debug, Clone)]
pub enum Step {
    Kernel(KernelStep),
    SyncWait(SyncWaitStep),
    Transfer(TransferStep),
}

#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    pub steps: Vec<Step>,
    pub chunks: Vec<ChunkInfo>,
    pub arenas: Vec<ArenaInfo>,
    pub events: Vec<EventInfo>,
    pub stream_count: usize,
}
