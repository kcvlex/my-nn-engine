use std::io::Read;
use std::io::Write;
use std::os::unix::net::UnixListener;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

use crate::CommError;
use crate::Communicator;
use crate::DataType;
use crate::ReduceOp;

const HELLO_MAGIC: u32 = 0x4d59_4e43;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_RETRY_INTERVAL: Duration = Duration::from_millis(5);

const TAG_ALL_REDUCE: u8 = 1;
const TAG_BARRIER: u8 = 2;

/// Star topology over Unix domain sockets: rank 0 is the hub and performs a
/// naive gather + broadcast for each collective.
pub struct UdsCommunicator {
    rank: usize,
    world_size: usize,
    endpoint: Mutex<Endpoint>,
}

enum Endpoint {
    /// peers[i] is the stream to rank i + 1.
    Hub {
        peers: Vec<UnixStream>,
    },
    Spoke {
        hub: UnixStream,
    },
}

impl UdsCommunicator {
    pub fn from_env() -> Result<Self, CommError> {
        let rank = env_usize("MYNN_RANK")?;
        let world_size = env_usize("MYNN_WORLD_SIZE")?;
        let path = std::env::var("MYNN_COMM_SOCK")
            .map_err(|_| CommError::InvalidConfig("MYNN_COMM_SOCK is not set".into()))?;
        Self::connect(Path::new(&path), rank, world_size)
    }

    pub fn connect(path: &Path, rank: usize, world_size: usize) -> Result<Self, CommError> {
        if world_size == 0 || rank >= world_size {
            return Err(CommError::InvalidConfig(format!(
                "rank {rank} out of range for world_size {world_size}"
            )));
        }
        let endpoint = if rank == 0 {
            let _ = std::fs::remove_file(path);
            let listener = UnixListener::bind(path)?;
            let mut peers: Vec<Option<UnixStream>> = (1..world_size).map(|_| None).collect();
            for _ in 1..world_size {
                let (mut stream, _) = listener.accept()?;
                let peer_rank = read_hello(&mut stream)?;
                if !(1..world_size).contains(&peer_rank) {
                    return Err(CommError::Protocol(format!(
                        "unexpected rank {peer_rank} connected"
                    )));
                }
                if peers[peer_rank - 1].replace(stream).is_some() {
                    return Err(CommError::Protocol(format!(
                        "rank {peer_rank} connected twice"
                    )));
                }
            }
            Endpoint::Hub {
                peers: peers.into_iter().map(Option::unwrap).collect(),
            }
        } else {
            let mut hub = connect_with_retry(path)?;
            write_hello(&mut hub, rank)?;
            Endpoint::Spoke { hub }
        };
        Ok(Self {
            rank,
            world_size,
            endpoint: Mutex::new(endpoint),
        })
    }
}

impl Communicator for UdsCommunicator {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce(&self, buf: &mut [u8], dtype: DataType, op: ReduceOp) -> Result<(), CommError> {
        if !buf.len().is_multiple_of(dtype.size_of()) {
            return Err(CommError::InvalidConfig(format!(
                "buffer of {} bytes is not a multiple of {:?} element size",
                buf.len(),
                dtype
            )));
        }
        match &mut *self.endpoint.lock().unwrap() {
            Endpoint::Hub { peers } => {
                let mut recv = vec![0u8; buf.len()];
                for peer in peers.iter_mut() {
                    read_frame(peer, TAG_ALL_REDUCE, buf.len(), &mut recv)?;
                    reduce(buf, &recv, dtype, op);
                }
                for peer in peers.iter_mut() {
                    write_frame(peer, TAG_ALL_REDUCE, buf)?;
                }
            }
            Endpoint::Spoke { hub } => {
                write_frame(hub, TAG_ALL_REDUCE, buf)?;
                read_frame(hub, TAG_ALL_REDUCE, buf.len(), buf)?;
            }
        }
        Ok(())
    }

    fn barrier(&self) -> Result<(), CommError> {
        match &mut *self.endpoint.lock().unwrap() {
            Endpoint::Hub { peers } => {
                for peer in peers.iter_mut() {
                    read_frame(peer, TAG_BARRIER, 0, &mut [])?;
                }
                for peer in peers.iter_mut() {
                    write_frame(peer, TAG_BARRIER, &[])?;
                }
            }
            Endpoint::Spoke { hub } => {
                write_frame(hub, TAG_BARRIER, &[])?;
                read_frame(hub, TAG_BARRIER, 0, &mut [])?;
            }
        }
        Ok(())
    }
}

fn reduce(acc: &mut [u8], other: &[u8], dtype: DataType, op: ReduceOp) {
    match (dtype, op) {
        (DataType::F32, ReduceOp::Sum) => {
            for (a, b) in acc.chunks_exact_mut(4).zip(other.chunks_exact(4)) {
                let sum = f32::from_ne_bytes(a.try_into().unwrap()) +
                    f32::from_ne_bytes(b.try_into().unwrap());
                a.copy_from_slice(&sum.to_ne_bytes());
            }
        }
    }
}

fn connect_with_retry(path: &Path) -> Result<UnixStream, CommError> {
    let deadline = Instant::now() + CONNECT_TIMEOUT;
    loop {
        match UnixStream::connect(path) {
            Ok(stream) => return Ok(stream),
            Err(e) if Instant::now() >= deadline => return Err(CommError::Io(e)),
            Err(_) => std::thread::sleep(CONNECT_RETRY_INTERVAL),
        }
    }
}

fn write_hello(stream: &mut UnixStream, rank: usize) -> Result<(), CommError> {
    let mut msg = [0u8; 8];
    msg[..4].copy_from_slice(&HELLO_MAGIC.to_le_bytes());
    msg[4..].copy_from_slice(&(rank as u32).to_le_bytes());
    stream.write_all(&msg)?;
    Ok(())
}

fn read_hello(stream: &mut UnixStream) -> Result<usize, CommError> {
    let mut msg = [0u8; 8];
    stream.read_exact(&mut msg)?;
    let magic = u32::from_le_bytes(msg[..4].try_into().unwrap());
    if magic != HELLO_MAGIC {
        return Err(CommError::Protocol(format!("bad hello magic {magic:#x}")));
    }
    Ok(u32::from_le_bytes(msg[4..].try_into().unwrap()) as usize)
}

fn write_frame(stream: &mut UnixStream, tag: u8, payload: &[u8]) -> Result<(), CommError> {
    let mut header = [0u8; 9];
    header[0] = tag;
    header[1..].copy_from_slice(&(payload.len() as u64).to_le_bytes());
    stream.write_all(&header)?;
    stream.write_all(payload)?;
    Ok(())
}

fn read_frame(
    stream: &mut UnixStream,
    expected_tag: u8,
    expected_len: usize,
    out: &mut [u8],
) -> Result<(), CommError> {
    let mut header = [0u8; 9];
    stream.read_exact(&mut header)?;
    let tag = header[0];
    let len = u64::from_le_bytes(header[1..].try_into().unwrap()) as usize;
    if tag != expected_tag || len != expected_len {
        return Err(CommError::Protocol(format!(
            "expected frame (tag {expected_tag}, len {expected_len}), got (tag {tag}, len {len})"
        )));
    }
    stream.read_exact(&mut out[..len])?;
    Ok(())
}

fn env_usize(key: &str) -> Result<usize, CommError> {
    let value =
        std::env::var(key).map_err(|_| CommError::InvalidConfig(format!("{key} is not set")))?;
    value
        .parse()
        .map_err(|_| CommError::InvalidConfig(format!("{key}={value} is not a number")))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::*;

    fn run_world<F, R>(world_size: usize, f: F) -> Vec<R>
    where
        F: Fn(UdsCommunicator) -> R + Sync,
        R: Send,
    {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("comm.sock");
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..world_size)
                .map(|rank| {
                    let path = &path;
                    let f = &f;
                    s.spawn(move || f(UdsCommunicator::connect(path, rank, world_size).unwrap()))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        })
    }

    fn to_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_ne_bytes()).collect()
    }

    fn to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn all_reduce_sums_across_ranks() {
        let results = run_world(3, |comm| {
            let base = (comm.rank() + 1) as f32;
            let mut buf = to_bytes(&[base, base * 10.0, -base]);
            comm.all_reduce(&mut buf, DataType::F32, ReduceOp::Sum)
                .unwrap();
            to_f32(&buf)
        });
        for result in results {
            assert_eq!(result, vec![6.0, 60.0, -6.0]);
        }
    }

    #[test]
    fn all_reduce_world_size_one_is_identity() {
        let results = run_world(1, |comm| {
            let mut buf = to_bytes(&[1.5, -2.5]);
            comm.all_reduce(&mut buf, DataType::F32, ReduceOp::Sum)
                .unwrap();
            comm.barrier().unwrap();
            to_f32(&buf)
        });
        assert_eq!(results, vec![vec![1.5, -2.5]]);
    }

    #[test]
    fn consecutive_collectives_reuse_the_connection() {
        run_world(2, |comm| {
            for round in 0..3 {
                let mut buf = to_bytes(&[round as f32 + comm.rank() as f32]);
                comm.all_reduce(&mut buf, DataType::F32, ReduceOp::Sum)
                    .unwrap();
                assert_eq!(to_f32(&buf), vec![round as f32 * 2.0 + 1.0]);
            }
            comm.barrier().unwrap();
        });
    }

    #[test]
    fn barrier_waits_for_all_ranks() {
        let arrived = AtomicUsize::new(0);
        run_world(4, |comm| {
            arrived.fetch_add(1, Ordering::SeqCst);
            comm.barrier().unwrap();
            assert_eq!(arrived.load(Ordering::SeqCst), 4);
        });
    }

    #[test]
    fn mismatched_lengths_error_out() {
        let results = run_world(2, |comm| {
            let len = if comm.rank() == 0 { 4 } else { 8 };
            let mut buf = vec![0u8; len];
            comm.all_reduce(&mut buf, DataType::F32, ReduceOp::Sum)
                .err()
        });
        assert!(results.iter().all(Option::is_some));
    }
}
