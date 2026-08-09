pub mod uds;

pub use uds::UdsCommunicator;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
}

impl DataType {
    pub fn size_of(self) -> usize {
        match self {
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
}

#[derive(Debug, thiserror::Error)]
pub enum CommError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol: {0}")]
    Protocol(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

pub trait Communicator: Send + Sync {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;
    fn all_reduce(&self, buf: &mut [u8], dtype: DataType, op: ReduceOp) -> Result<(), CommError>;
    fn barrier(&self) -> Result<(), CommError>;
}
