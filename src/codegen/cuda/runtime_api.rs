use crate::codegen::cuda::*;
use delegate::delegate;
use derive_more::From;

#[derive(From)]
pub enum CudaRuntimeApi {
    EventCreate(EventCreate),
    EventSynchronize(EventSynchronize),
    StreamCreate(StreamCreate),
    RecordEvent(RecordEvent),
    Malloc(Malloc),
    Memcpy(Memcpy),
    WaitEvent(WaitEvent),
}

impl CudaRuntimeApi {
    delegate! {
        to match self {
            CudaRuntimeApi::EventCreate(event_create) => event_create,
            CudaRuntimeApi::EventSynchronize(event_sync) => event_sync,
            CudaRuntimeApi::StreamCreate(stream_create) => stream_create,
            CudaRuntimeApi::RecordEvent(event) => event,
            CudaRuntimeApi::Malloc(malloc) => malloc,
            CudaRuntimeApi::Memcpy(memcpy) => memcpy,
            CudaRuntimeApi::WaitEvent(wait_event) => wait_event,
        } {
            pub fn fragment(&self) -> String;
        }
    }
}

pub struct EventCreate {
    pub event_id: EventId,
}

impl EventCreate {
    fn fragment(&self) -> String {
        format!(
            "cudaEventCreate(&{})",
            self.event_id.to_identifier().fragment()
        )
    }
}

pub struct EventSynchronize {
    pub event_id: EventId,
}

impl EventSynchronize {
    fn fragment(&self) -> String {
        format!(
            "cudaEventSynchronize({})",
            self.event_id.to_identifier().fragment()
        )
    }
}

pub struct StreamCreate {
    pub stream_id: StreamId,
}

impl StreamCreate {
    fn fragment(&self) -> String {
        format!(
            "cudaStreamCreate(&{})",
            self.stream_id.to_identifier().fragment()
        )
    }
}

pub struct RecordEvent {
    pub event_id: EventId,
    pub stream_id: StreamId,
}

impl RecordEvent {
    fn fragment(&self) -> String {
        format!(
            "cudaEventRecord({}, {})",
            self.event_id.to_identifier().fragment(),
            self.stream_id.to_identifier().fragment()
        )
    }
}

pub struct Malloc {
    pub dst: Expr,
    pub mem_size: MemSize,
}

impl Malloc {
    fn fragment(&self) -> String {
        format!(
            "cudaMalloc(&{dst}, {mem_size})",
            dst = self.dst.fragment(),
            mem_size = self.mem_size.fragment()
        )
    }
}

pub enum CudaMemcpyKind {
    HostToDevice,
    DeviceToHost,
}

impl CudaMemcpyKind {
    fn fragment(&self) -> &'static str {
        match self {
            CudaMemcpyKind::HostToDevice => "cudaMemcpyHostToDevice",
            CudaMemcpyKind::DeviceToHost => "cudaMemcpyDeviceToHost",
        }
    }
}

pub struct Memcpy {
    pub dst: Expr,
    pub src: Expr,
    pub mem_size: MemSize,
    pub kind: CudaMemcpyKind,
    pub stream: StreamId,
}

impl Memcpy {
    fn fragment(&self) -> String {
        format!(
            "cudaMemcpyAsync({dst}, {src}, {mem_size}, {kind}, {stream})",
            dst = self.dst.fragment(),
            src = self.src.fragment(),
            mem_size = self.mem_size.fragment(),
            kind = self.kind.fragment(),
            stream = self.stream.to_identifier().fragment()
        )
    }
}

pub struct WaitEvent {
    pub stream_id: StreamId,
    pub event_id: EventId,
}

impl WaitEvent {
    fn fragment(&self) -> String {
        format!(
            "cudaStreamWaitEvent({stream}, {event})",
            stream = self.stream_id.to_identifier().fragment(),
            event = self.event_id.to_identifier().fragment()
        )
    }
}

macro_rules! impl_into_stmt {
    ($name:ident) => {
        impl From<$name> for Statement {
            fn from(x: $name) -> Statement {
                Statement::CudaRuntimeApi(x.into())
            }
        }
    };
}

impl_into_stmt!(EventCreate);
impl_into_stmt!(EventSynchronize);
impl_into_stmt!(StreamCreate);
impl_into_stmt!(RecordEvent);
impl_into_stmt!(Malloc);
impl_into_stmt!(Memcpy);
impl_into_stmt!(WaitEvent);
