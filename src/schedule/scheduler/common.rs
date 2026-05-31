//! Shared planning infrastructure for the scheduler variants
//! ([`super::memory_aware`], [`super::prefetch`]): dependency analysis, the
//! chunk allocator, and small plan-building helpers.

use std::collections::HashMap;
use std::collections::HashSet;

use crate::graph::ValueId;
use crate::schedule::*;

pub(crate) const ALIGNMENT: usize = 256;

pub(crate) fn align_up(size: usize) -> usize {
    size.next_multiple_of(ALIGNMENT)
}

/// How kernels are placed across devices.
#[derive(Debug, Clone, Copy)]
pub enum PlacementStrategy {
    Uniform(Device),
    StructuralKvTouch,
}

/// Producer / consumer / use-count relationships derived from the kernel graph.
pub(crate) struct Deps {
    pub(crate) value2producer: HashMap<ValueId, KernelId>,
    pub(crate) value_uses_count: HashMap<ValueId, usize>,
    pub(crate) kernel_preds: HashMap<KernelId, HashSet<KernelId>>,
    pub(crate) kernel_consumers: HashMap<KernelId, HashSet<KernelId>>,
}

impl Deps {
    pub(crate) fn new(schedule: &Schedule) -> Self {
        let mut value2producer = HashMap::new();
        for (kid, kernel) in schedule.kernels.iter() {
            for &out in &kernel.outputs {
                value2producer.insert(out, kid);
            }
        }

        let mut value_uses_count: HashMap<ValueId, usize> = HashMap::new();
        let mut kernel_preds: HashMap<KernelId, HashSet<KernelId>> = HashMap::new();
        let mut kernel_consumers: HashMap<KernelId, HashSet<KernelId>> = HashMap::new();
        for (kid, _) in schedule.kernels.iter() {
            kernel_preds.insert(kid, HashSet::new());
            kernel_consumers.insert(kid, HashSet::new());
        }
        for (kid, kernel) in schedule.kernels.iter() {
            for input in kernel.inputs.iter().flatten() {
                *value_uses_count.entry(*input).or_insert(0) += 1;
                if let Some(&prod) = value2producer.get(input) {
                    if prod != kid {
                        kernel_preds.get_mut(&kid).unwrap().insert(prod);
                        kernel_consumers.get_mut(&prod).unwrap().insert(kid);
                    }
                }
            }
        }

        Self {
            value2producer,
            value_uses_count,
            kernel_preds,
            kernel_consumers,
        }
    }
}

pub(crate) struct ChunkState {
    pub(crate) arena_id: ArenaId,
    pub(crate) size: usize,
    pub(crate) live_uses: usize,
    pub(crate) first_use: bool,
}

pub(crate) struct ArenaChunks {
    pub(crate) tier: MemoryTier,
    pub(crate) owned: Vec<ChunkId>,
    pub(crate) free_per_stream: Vec<Vec<ChunkId>>,
}

impl ArenaChunks {
    pub(crate) fn new(tier: MemoryTier, num_streams: usize) -> Self {
        Self {
            tier,
            owned: Vec::new(),
            free_per_stream: vec![Vec::new(); num_streams],
        }
    }
}

#[derive(Default)]
pub(crate) struct ChunkAllocator {
    pub(crate) arena2chunks: Vec<ArenaChunks>,
    pub(crate) all_chunks: Vec<ChunkState>,
}

impl ChunkAllocator {
    pub(crate) fn new_arena(&mut self, tier: MemoryTier, num_streams: usize) -> ArenaId {
        let arena_id = self.arena2chunks.len();
        self.arena2chunks.push(ArenaChunks::new(tier, num_streams));
        arena_id
    }

    pub(crate) fn chunk_tier(&self, id: ChunkId) -> MemoryTier {
        self.arena2chunks[self.all_chunks[id].arena_id].tier
    }

    pub(crate) fn alloc(
        &mut self,
        size: usize,
        arena_id: ArenaId,
        stream: StreamId,
        uses: usize,
    ) -> ChunkId {
        use std::cmp::max;
        let id = if let Some(id) = self.arena2chunks[arena_id].free_per_stream[stream.index()].pop()
        {
            id
        } else {
            let id = self.all_chunks.len();
            self.arena2chunks[arena_id].owned.push(id);
            self.all_chunks.push(ChunkState {
                arena_id,
                size: 0,
                live_uses: 0,
                first_use: false,
            });
            id
        };
        let entry = &mut self.all_chunks[id];
        entry.size = max(entry.size, size);
        entry.live_uses += uses;
        id
    }

    /// Release a chunk whose value will never be read again (e.g. an unused
    /// multi-output of a kernel).
    pub(crate) fn release_if_dead(&mut self, id: ChunkId, stream: StreamId) {
        if self.all_chunks[id].live_uses == 0 {
            self.free(id, stream);
        }
    }

    pub(crate) fn add_uses(&mut self, id: ChunkId, uses: usize) {
        self.all_chunks[id].live_uses += uses;
    }

    pub(crate) fn mark_as_first_use(&mut self, id: ChunkId) -> bool {
        let res = !self.all_chunks[id].first_use;
        self.all_chunks[id].first_use = true;
        res
    }

    pub(crate) fn consume(&mut self, id: ChunkId, stream: StreamId) {
        let entry = &mut self.all_chunks[id];
        entry.live_uses = entry.live_uses.saturating_sub(1);
        if entry.live_uses == 0 {
            self.free(id, stream);
        }
    }

    pub(crate) fn free(&mut self, id: ChunkId, stream: StreamId) {
        let arena_id = self.all_chunks[id].arena_id;
        self.arena2chunks[arena_id].free_per_stream[stream.index()].push(id);
    }
}

pub(crate) struct KernelInfo {
    pub(crate) order: usize,
    pub(crate) stream: StreamId,
    pub(crate) event: EventId,
    pub(crate) device: Device,
}

#[derive(Clone, Copy)]
pub(crate) struct TransferRecord {
    pub(crate) place: AllocPlace,
    pub(crate) event: EventId,
    pub(crate) stream: StreamId,
}

#[derive(Clone, Copy)]
pub(crate) struct PostTransfer {
    pub(crate) src: ValueBinding,
    pub(crate) dst: ValueBinding,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_allocator() -> (ChunkAllocator, ArenaId, StreamId) {
        let mut a = ChunkAllocator::default();
        let arena = a.new_arena(MemoryTier::GpuArena, 1);
        (a, arena, StreamId(0))
    }

    #[test]
    fn alloc_with_positive_uses_is_not_released_immediately() {
        let (mut a, arena, stream) = fresh_allocator();

        let cid = a.alloc(1024, arena, stream, 1);
        // Still live: release_if_dead must be a no-op.
        a.release_if_dead(cid, stream);

        // A second alloc should NOT reuse cid because cid still has live_uses=1.
        let cid2 = a.alloc(1024, arena, stream, 1);
        assert_ne!(cid, cid2);

        // Consuming cid once drops live_uses to 0 and frees it.
        a.consume(cid, stream);
        let cid3 = a.alloc(1024, arena, stream, 1);
        assert_eq!(cid3, cid, "freed chunk should be reused");
    }

    #[test]
    fn alloc_with_zero_uses_is_reusable_after_release_if_dead() {
        let (mut a, arena, stream) = fresh_allocator();

        // Multi-output kernel emits an output no later kernel reads.
        let dead = a.alloc(512, arena, stream, 0);
        a.release_if_dead(dead, stream);

        // The slot must come back via the free list.
        let next = a.alloc(2048, arena, stream, 1);
        assert_eq!(next, dead, "dead-on-arrival chunk should be reused");
        // The chunk grows to fit the larger request and stays at that size.
        assert_eq!(a.all_chunks[next].size, 2048);
    }

    #[test]
    fn release_if_dead_is_noop_when_post_transfer_added_uses() {
        // Simulates the graph-output path: alloc(uses=0), then add_uses(1) for
        // the post-transfer that copies the chunk to host. The chunk must NOT
        // be returned to the free list because the post-transfer will read it.
        let (mut a, arena, stream) = fresh_allocator();

        let cid = a.alloc(1024, arena, stream, 0);
        a.add_uses(cid, 1);
        a.release_if_dead(cid, stream);

        let cid2 = a.alloc(1024, arena, stream, 1);
        assert_ne!(
            cid, cid2,
            "chunk pending a post-transfer must not be reused"
        );
    }

    #[test]
    fn unused_output_does_not_steal_chunks_intended_for_reuse() {
        // Mirrors the planner bug surfaced by DynamicQuantizeLinear's unused
        // zero-point output: a previously freed chunk on the free list is
        // popped to back the dead-on-arrival output, then never returned, so
        // the next legitimate alloc has to create a new chunk and the arena
        // grows. With release_if_dead, the chunk goes straight back into the
        // free list so the next legitimate alloc reuses it.
        let (mut a, arena, stream) = fresh_allocator();

        // 1. A real output is allocated and consumed, returning to free list.
        let big = a.alloc(11_272_192, arena, stream, 1);
        a.consume(big, stream);

        // 2. Some later kernel's unused multi-output picks up the same slot...
        let dead = a.alloc(512, arena, stream, 0);
        assert_eq!(dead, big, "free-list pop should reuse the slot");
        a.release_if_dead(dead, stream);

        // 3. ...then the next legitimate alloc reuses the same slot again,
        //    keeping the total chunk count at 1.
        let next = a.alloc(11_272_192, arena, stream, 1);
        assert_eq!(next, big, "slot must still be reusable after a dead claim");
        assert_eq!(a.all_chunks.len(), 1, "only one chunk should exist");
    }
}
