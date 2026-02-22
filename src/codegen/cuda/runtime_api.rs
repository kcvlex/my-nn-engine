use derive_more::From;

use crate::codegen::cuda::*;

#[derive(From)]
pub enum CudaRuntimeApi {
    EventCreate(EventCreate),
    EventSynchronize(EventSynchronize),
    StreamCreate(StreamCreate),
    RecordEvent(RecordEvent),
    Malloc(Malloc),
    Free(Free),
    Memcpy(Memcpy),
    WaitEvent(WaitEvent),
    DeviceSynchronize,
}

impl std::fmt::Display for CudaRuntimeApi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaRuntimeApi::EventCreate(event_create) => write!(f, "{}", event_create),
            CudaRuntimeApi::EventSynchronize(event_sync) => write!(f, "{}", event_sync),
            CudaRuntimeApi::StreamCreate(stream_create) => write!(f, "{}", stream_create),
            CudaRuntimeApi::RecordEvent(event) => write!(f, "{}", event),
            CudaRuntimeApi::Malloc(malloc) => write!(f, "{}", malloc),
            CudaRuntimeApi::Free(free) => write!(f, "{}", free),
            CudaRuntimeApi::Memcpy(memcpy) => write!(f, "{}", memcpy),
            CudaRuntimeApi::WaitEvent(wait_event) => write!(f, "{}", wait_event),
            CudaRuntimeApi::DeviceSynchronize => write!(f, "cudaDeviceSynchronize()"),
        }
    }
}

pub struct EventCreate {
    pub event_id: EventId,
}

impl std::fmt::Display for EventCreate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudaEventCreate(&{})", self.event_id)
    }
}

pub struct EventSynchronize {
    pub event_id: EventId,
}

impl std::fmt::Display for EventSynchronize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudaEventSynchronize({})", self.event_id)
    }
}

pub struct StreamCreate {
    pub stream_id: StreamId,
}

impl std::fmt::Display for StreamCreate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudaStreamCreate(&{})", self.stream_id)
    }
}

pub struct RecordEvent {
    pub event_id: EventId,
    pub stream_id: StreamId,
}

impl std::fmt::Display for RecordEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudaEventRecord({}, {})", self.event_id, self.stream_id)
    }
}

pub struct Malloc {
    pub dst: Expr,
    pub mem_size: MemSize,
}

impl std::fmt::Display for Malloc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudaMalloc(&{}, {})", self.dst, self.mem_size)
    }
}

pub struct Free(pub Expr);

impl std::fmt::Display for Free {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cudaFree({})", self.0)
    }
}

pub enum CudaMemcpyKind {
    HostToDevice,
    DeviceToHost,
}

impl std::fmt::Display for CudaMemcpyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                CudaMemcpyKind::HostToDevice => "cudaMemcpyHostToDevice",
                CudaMemcpyKind::DeviceToHost => "cudaMemcpyDeviceToHost",
            }
        )
    }
}

pub struct Memcpy {
    pub dst: Expr,
    pub src: Expr,
    pub mem_size: MemSize,
    pub kind: CudaMemcpyKind,
    pub stream: StreamId,
}

impl std::fmt::Display for Memcpy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cudaMemcpyAsync({}, {}, {}, {}, {})",
            self.dst, self.src, self.mem_size, self.kind, self.stream
        )
    }
}

pub struct WaitEvent {
    pub stream_id: StreamId,
    pub event_id: EventId,
}

impl std::fmt::Display for WaitEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cudaStreamWaitEvent({}, {})",
            self.stream_id, self.event_id
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
impl_into_stmt!(Free);
impl_into_stmt!(Memcpy);
impl_into_stmt!(WaitEvent);
