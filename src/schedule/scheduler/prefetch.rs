//! Weight-prefetch scheduler (experimental, P0).
//!
//! A fork of [`crate::schedule::scheduler`] that streams selected initializers
//! (`GpuResident` -> `HostStreamed`) from host memory into bounded GPU staging
//! chunks just before the consuming kernel, instead of keeping every weight
//! resident in VRAM. Kept as a separate orchestrator so the proven all-resident
//! scheduler stays untouched while this path churns through P0..P3.
//!
//! P0 scope: correctness only. Streamed weights are brought in on the *consuming
//! kernel's own stream* (serialized, no overlap) and the staging chunk is freed
//! immediately after the kernel reads it, so a single slot is reused for every
//! streamed weight (VRAM stays bounded). Copy-stream overlap + a real staging
//! pool with WAR-event slot recycling are P1.

use std::collections::HashMap;
use std::collections::HashSet;

use super::common::*;
use crate::graph::operator::Operator;
use crate::graph::ValueId;
use crate::schedule::placement::Placement;
use crate::schedule::*;
use crate::tensor::types::DataType;
use crate::tensor::types::SIntType;

/// Policy for deciding which initializers are streamed from host (`HostStreamed`)
/// vs kept resident in VRAM (`GpuResident`). P0 uses a crude size threshold; the
/// budget-greedy policy is P2.
#[derive(Debug, Clone, Copy)]
pub enum PrefetchPolicy {
    /// Stream every initializer consumed by a GPU kernel whose byte size is at
    /// least `min_bytes`. A non-trivial threshold (e.g. 1 MiB) naturally
    /// excludes scalar / metadata initializers.
    SizeThreshold { min_bytes: usize },
}

pub struct PrefetchSchedulePass {
    pub num_streams: usize,
    pub placement_strategy: PlacementStrategy,
    pub policy: PrefetchPolicy,
}

impl SchedulePass for PrefetchSchedulePass {
    fn summary(&self) -> &str {
        "Weight-prefetch schedule (experimental)"
    }

    fn run(&self, schedule: &mut Schedule) {
        let placement = match self.placement_strategy {
            PlacementStrategy::Uniform(d) => Placement::uniform(schedule, d),
            PlacementStrategy::StructuralKvTouch => Placement::structural_kv_touch(schedule),
        };
        let streamed = select_streamed(schedule, &placement, self.policy);
        let plan = build(schedule, self.num_streams, placement, streamed);
        schedule.execution_plan = Some(plan);
    }
}

/// Compute the `HostStreamed` initializer set from `policy`.
fn select_streamed(
    schedule: &Schedule,
    placement: &Placement,
    policy: PrefetchPolicy,
) -> HashSet<ValueId> {
    let initializers: HashSet<ValueId> = schedule.initializers.iter().copied().collect();
    let mut streamed = HashSet::new();
    match policy {
        PrefetchPolicy::SizeThreshold { min_bytes } => {
            for (kid, kernel) in schedule.kernels.iter() {
                if placement.device_of(kid) != Device::CUDA {
                    continue;
                }
                for input in kernel.inputs.iter().flatten().copied() {
                    if initializers.contains(&input) && schedule.value_byte_size(input) >= min_bytes
                    {
                        streamed.insert(input);
                    }
                }
            }
        }
    }
    streamed
}

struct PrefetchScheduler<'s> {
    schedule: &'s Schedule,
    deps: Deps,
    output_alias: HashMap<ValueId, ValueId>,
    initializers: HashSet<ValueId>,
    session_states: HashSet<ValueId>,
    inputs_set: HashSet<ValueId>,
    placement: Placement,
    session_state_tier: MemoryTier,

    /// Initializers brought in from host on demand (`HostStreamed`); all others
    /// are `GpuResident`.
    streamed: HashSet<ValueId>,

    num_streams: usize,

    allocator: ChunkAllocator,
    value_on_tier: HashMap<(ValueId, MemoryTier), TransferRecord>,
    next_event_id: usize,

    value2place: HashMap<ValueId, AllocPlace>,
    value_remaining_uses: HashMap<ValueId, usize>,
    kernel_info: HashMap<KernelId, KernelInfo>,
    stream_load: Vec<usize>,
    steps: Vec<Step>,
}

impl<'s> PrefetchScheduler<'s> {
    fn new(
        schedule: &'s Schedule,
        num_streams: usize,
        placement: Placement,
        streamed: HashSet<ValueId>,
    ) -> Self {
        let deps = Deps::new(schedule);
        let output_alias = schedule.output_aliases();
        let needs_cuda = placement.iter().any(|(_, d)| d == Device::CUDA);
        let num_streams = if needs_cuda { num_streams.max(1) } else { 1 };
        let session_state_tier = if needs_cuda {
            MemoryTier::GpuArena
        } else {
            MemoryTier::HostArena
        };

        let initializers: HashSet<ValueId> = schedule.initializers.iter().copied().collect();
        let session_states: HashSet<ValueId> = schedule.session_states.iter().copied().collect();
        let inputs_set: HashSet<ValueId> = schedule.inputs.iter().copied().collect();

        let mut value2place: HashMap<ValueId, AllocPlace> = HashMap::new();
        for v in &initializers {
            value2place.insert(*v, AllocPlace::Initializer(*v));
        }
        for v in &session_states {
            value2place.insert(*v, AllocPlace::SessionState(*v));
        }
        for v in &inputs_set {
            value2place.insert(*v, AllocPlace::Input(*v));
        }

        let value_remaining_uses = deps.value_uses_count.clone();

        Self {
            schedule,
            deps,
            output_alias,
            initializers,
            session_states,
            inputs_set,
            placement,
            session_state_tier,
            streamed,
            num_streams,
            allocator: ChunkAllocator::default(),
            value_on_tier: HashMap::new(),
            next_event_id: 0,
            value2place,
            value_remaining_uses,
            kernel_info: HashMap::new(),
            stream_load: vec![0; num_streams],
            steps: Vec::new(),
        }
    }

    fn alloc_chunk(
        &mut self,
        tier: MemoryTier,
        size: usize,
        stream: StreamId,
        uses: usize,
    ) -> ChunkId {
        let arena_id = self
            .allocator
            .arena2chunks
            .iter()
            .position(|a| a.tier == tier)
            .unwrap_or_else(|| self.allocator.new_arena(tier, self.num_streams));
        self.allocator.alloc(size, arena_id, stream, uses)
    }

    fn run(mut self) -> ExecutionPlan {
        // Kernels are built in topological order (KernelId order), so scheduling
        // them in that order already respects every dependency.
        let order: Vec<KernelId> = self.schedule.kernels.iter().map(|(kid, _)| kid).collect();
        for kid in order {
            self.schedule_kernel(kid);
        }
        self.finalize_outputs();
        self.into_plan()
    }

    fn finalize_outputs(&mut self) {
        let already: HashSet<ValueId> = self
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Transfer(t) if matches!(t.dst.place, AllocPlace::Output(_)) => {
                    Some(t.dst.value)
                }
                _ => None,
            })
            .collect();
        let outputs = self.schedule.outputs.clone();
        for v in outputs {
            if already.contains(&v) {
                continue;
            }
            let place = match self.value2place.get(&v).copied() {
                Some(p) => p,
                None => continue,
            };
            if matches!(place, AllocPlace::Output(_)) {
                continue;
            }
            let producer = self.deps.value2producer.get(&v).copied();
            let (stream, device) = match producer {
                Some(kid) => {
                    let info = &self.kernel_info[&kid];
                    (info.stream, info.device)
                }
                None => (StreamId(0), Device::CPU),
            };
            let event = self.fresh_event();
            self.steps.push(Step::Transfer(TransferStep {
                src: ValueBinding {
                    value: v,
                    role: BindingRole::Input,
                    place,
                    is_first_use: false,
                },
                dst: ValueBinding {
                    value: v,
                    role: BindingRole::Output,
                    place: AllocPlace::Output(v),
                    is_first_use: false,
                },
                context: ExecutionContext { device, stream },
                records_event: Some(event),
            }));
        }
    }

    fn pick_stream(&self, kid: KernelId) -> StreamId {
        if self.num_streams == 1 {
            return StreamId(0);
        }
        let latest = self.deps.kernel_preds[&kid]
            .iter()
            .map(|p| &self.kernel_info[p])
            .max_by_key(|info| info.order);
        match latest {
            Some(info) => info.stream,
            None => {
                let (idx, _) = self
                    .stream_load
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, l)| **l)
                    .unwrap();
                StreamId(idx)
            }
        }
    }

    fn pick_device(&self, kid: KernelId) -> Device {
        self.placement.device_of(kid)
    }

    fn fresh_event(&mut self) -> EventId {
        let id = EventId(self.next_event_id);
        self.next_event_id += 1;
        id
    }

    fn ensure_value_on_tier(
        &mut self,
        value: ValueId,
        original: AllocPlace,
        dst_tier: MemoryTier,
        stream: StreamId,
    ) -> AllocPlace {
        let needs_transfer = match original {
            AllocPlace::Chunk(cid) => self.allocator.chunk_tier(cid) != dst_tier,
            AllocPlace::Input(_) | AllocPlace::Output(_) => dst_tier != MemoryTier::HostArena,
            AllocPlace::SessionState(_) => self.session_state_tier != dst_tier,
            AllocPlace::Initializer(_) => false,
        };
        if !needs_transfer {
            return original;
        }
        if let Some(rty) = self.schedule.get_resolved_tensor_type(value) {
            if rty.dims.is_scalar() && matches!(rty.elem_type, DataType::SInt(SIntType::I64)) {
                return original;
            }
        }
        if let Some(cached) = self.value_on_tier.get(&(value, dst_tier)).copied() {
            if let AllocPlace::Chunk(cid) = cached.place {
                self.allocator.add_uses(cid, 1);
            }
            if cached.stream != stream {
                let device = match dst_tier {
                    MemoryTier::HostArena => Device::CPU,
                    MemoryTier::GpuArena => Device::CUDA,
                };
                self.steps.push(Step::SyncWait(SyncWaitStep {
                    context: ExecutionContext { device, stream },
                    event: cached.event,
                }));
            }
            return cached.place;
        }
        let size = self.schedule.value_byte_size(value);
        let dst_cid = self.alloc_chunk(dst_tier, size, stream, 1);
        let dst_place = AllocPlace::Chunk(dst_cid);
        let event = self.fresh_event();
        let device = match dst_tier {
            MemoryTier::HostArena => Device::CPU,
            MemoryTier::GpuArena => Device::CUDA,
        };
        let context = ExecutionContext { device, stream };
        let src_first_use = match original {
            AllocPlace::Chunk(cid) => self.allocator.mark_as_first_use(cid),
            _ => false,
        };
        let dst_first_use = self.allocator.mark_as_first_use(dst_cid);
        self.steps.push(Step::Transfer(TransferStep {
            src: ValueBinding {
                value,
                role: BindingRole::Input,
                place: original,
                is_first_use: src_first_use,
            },
            dst: ValueBinding {
                value,
                role: BindingRole::Output,
                place: dst_place,
                is_first_use: dst_first_use,
            },
            context,
            records_event: Some(event),
        }));
        self.value_on_tier.insert(
            (value, dst_tier),
            TransferRecord {
                place: dst_place,
                event,
                stream,
            },
        );
        dst_place
    }

    fn schedule_kernel(&mut self, kid: KernelId) {
        let device = self.pick_device(kid);
        let stream = self.pick_stream(kid);
        let order = self.kernel_info.len();
        let event = self.fresh_event();
        self.kernel_info.insert(
            kid,
            KernelInfo {
                order,
                stream,
                event,
                device,
            },
        );
        self.stream_load[stream.index()] += 1;

        let context = ExecutionContext { device, stream };

        for pred in &self.deps.kernel_preds[&kid] {
            let pred_info = &self.kernel_info[pred];
            if pred_info.stream != stream {
                self.steps.push(Step::SyncWait(SyncWaitStep {
                    context,
                    event: pred_info.event,
                }));
            }
        }

        let (bindings, post_transfers, staging_chunks) = self.bind_kernel(kid, stream);

        let is_transfer = matches!(
            &self.schedule.kernels[kid].body,
            KernelBody::Opaque(Opaque {
                op: Operator::Transfer(_)
            })
        );
        if is_transfer {
            let src = *bindings
                .iter()
                .find(|b| b.role == BindingRole::Input)
                .unwrap();
            let dst = *bindings
                .iter()
                .find(|b| b.role == BindingRole::Output)
                .unwrap();
            self.steps.push(Step::Transfer(TransferStep {
                src,
                dst,
                context,
                records_event: Some(event),
            }));
        } else {
            self.steps.push(Step::Kernel(KernelStep {
                kernel: kid,
                context,
                bindings,
                records_event: Some(event),
            }));
        }

        // Free streamed-weight staging chunks now that the kernel has read them.
        // The freed slot is returned to the per-stream free list so the next
        // streamed weight reuses it (P0 single-slot reuse).
        for cid in staging_chunks {
            self.allocator.consume(cid, stream);
        }

        for pt in post_transfers {
            let event = self.fresh_event();
            self.steps.push(Step::Transfer(TransferStep {
                src: pt.src,
                dst: pt.dst,
                context,
                records_event: Some(event),
            }));
            if let AllocPlace::Chunk(cid) = pt.src.place {
                self.allocator.consume(cid, stream);
            }
        }

        for input in self.schedule.kernels[kid].inputs.clone().iter().flatten() {
            if let Some(count) = self.value_remaining_uses.get_mut(input) {
                *count -= 1;
            }
            if let Some(AllocPlace::Chunk(cid)) = self.value2place.get(input).copied() {
                self.allocator.consume(cid, stream);
            }
        }
    }

    /// Returns `(bindings, post_transfers, staging_chunks)`. `staging_chunks` are
    /// GPU chunks holding streamed weights brought in for this kernel; the caller
    /// frees them after the kernel step.
    fn bind_kernel(
        &mut self,
        kid: KernelId,
        stream: StreamId,
    ) -> (Vec<ValueBinding>, Vec<PostTransfer>, Vec<ChunkId>) {
        let mut bindings = Vec::new();
        let mut post_transfers: Vec<PostTransfer> = Vec::new();
        let mut staging_chunks: Vec<ChunkId> = Vec::new();
        let inputs = self.schedule.kernels[kid].inputs.clone();
        let device = self.kernel_info[&kid].device;
        let tier = device.tier();

        for input in inputs.iter().flatten().copied() {
            // Streamed weight: bring it from host into a bounded GPU staging
            // chunk on this kernel's stream, bind the chunk, and queue the chunk
            // for release once the kernel has read it. Only on a GPU kernel;
            // a CPU kernel reading the weight would use the host pointer directly.
            if self.streamed.contains(&input) && tier == MemoryTier::GpuArena {
                let size = self.schedule.value_byte_size(input);
                let dst_cid = self.alloc_chunk(tier, size, stream, 1);
                let dst_place = AllocPlace::Chunk(dst_cid);
                let event = self.fresh_event();
                let dst_first_use = self.allocator.mark_as_first_use(dst_cid);
                self.steps.push(Step::Transfer(TransferStep {
                    src: ValueBinding {
                        value: input,
                        role: BindingRole::Input,
                        place: AllocPlace::Initializer(input),
                        is_first_use: false,
                    },
                    dst: ValueBinding {
                        value: input,
                        role: BindingRole::Output,
                        place: dst_place,
                        is_first_use: dst_first_use,
                    },
                    context: ExecutionContext { device, stream },
                    records_event: Some(event),
                }));
                bindings.push(ValueBinding {
                    value: input,
                    role: BindingRole::Input,
                    place: dst_place,
                    is_first_use: false,
                });
                staging_chunks.push(dst_cid);
                continue;
            }

            if self.initializers.contains(&input) {
                bindings.push(ValueBinding {
                    value: input,
                    role: BindingRole::Input,
                    place: AllocPlace::Initializer(input),
                    is_first_use: false,
                });
                continue;
            }
            let is_workspace = !self.deps.value2producer.contains_key(&input) &&
                !self.inputs_set.contains(&input) &&
                !self.session_states.contains(&input);

            let role = if is_workspace {
                BindingRole::Workspace
            } else {
                BindingRole::Input
            };

            let place = if is_workspace {
                let size = self.schedule.value_byte_size(input);
                let uses = self.value_remaining_uses.get(&input).copied().unwrap_or(0);
                let cid = self.alloc_chunk(tier, size, stream, uses);
                self.value2place.insert(input, AllocPlace::Chunk(cid));
                AllocPlace::Chunk(cid)
            } else {
                let original = *self
                    .value2place
                    .get(&input)
                    .unwrap_or_else(|| panic!("unresolved place for {input:?}"));
                self.ensure_value_on_tier(input, original, tier, stream)
            };

            let is_first_use =
                matches!(place, AllocPlace::Chunk(cid) if self.allocator.mark_as_first_use(cid));

            bindings.push(ValueBinding {
                value: input,
                role,
                place,
                is_first_use,
            });
        }

        let try_in_place_identity = match &self.schedule.kernels[kid].body {
            KernelBody::Opaque(Opaque {
                op: Operator::Identity | Operator::Reinterpret(_),
            }) => inputs[0].and_then(|in0| {
                let p = bindings.iter().find(|b| b.value == in0).map(|b| b.place)?;
                match p {
                    AllocPlace::Chunk(_) => {
                        if self.value_remaining_uses.get(&in0).copied().unwrap_or(0) == 1 {
                            Some(p)
                        } else {
                            None
                        }
                    }
                    _ => Some(p),
                }
            }),
            _ => None,
        };

        let must_in_place_place = self.schedule.must_in_place_input(kid).map(|idx| {
            let input_v = inputs[idx].expect("must_in_place input is None");
            bindings
                .iter()
                .find(|b| b.value == input_v)
                .expect("must_in_place input not in bindings")
                .place
        });

        let outputs = self.schedule.kernels[kid].outputs.clone();
        for output in outputs {
            let uses = self.value_remaining_uses.get(&output).copied().unwrap_or(0);
            let target_output_alias = self.output_alias.get(&output).copied();

            let (place, allocated) = if let Some(p) = try_in_place_identity {
                (p, false)
            } else if let Some(p) = must_in_place_place {
                (p, false)
            } else if let Some(target) = target_output_alias {
                if tier == MemoryTier::HostArena {
                    (AllocPlace::Output(target), false)
                } else {
                    let size = self.schedule.value_byte_size(output);
                    let cid = self.alloc_chunk(tier, size, stream, uses);
                    (AllocPlace::Chunk(cid), true)
                }
            } else {
                let size = self.schedule.value_byte_size(output);
                let cid = self.alloc_chunk(tier, size, stream, uses);
                (AllocPlace::Chunk(cid), true)
            };

            self.value2place.insert(output, place);

            if let (AllocPlace::Chunk(cid), false) = (place, allocated) {
                self.allocator.add_uses(cid, uses);
            }

            if let Some(target) = target_output_alias {
                if !matches!(place, AllocPlace::Output(_)) {
                    if let AllocPlace::Chunk(cid) = place {
                        self.allocator.add_uses(cid, 1);
                    }
                    post_transfers.push(PostTransfer {
                        src: ValueBinding {
                            value: output,
                            role: BindingRole::Input,
                            place,
                            is_first_use: false,
                        },
                        dst: ValueBinding {
                            value: output,
                            role: BindingRole::Output,
                            place: AllocPlace::Output(target),
                            is_first_use: false,
                        },
                    });
                }
            }

            let is_first_use =
                matches!(place, AllocPlace::Chunk(cid) if self.allocator.mark_as_first_use(cid));

            bindings.push(ValueBinding {
                value: output,
                role: BindingRole::Output,
                place,
                is_first_use,
            });

            if let AllocPlace::Chunk(cid) = place {
                self.allocator.release_if_dead(cid, stream);
            }
        }

        (bindings, post_transfers, staging_chunks)
    }

    fn into_plan(self) -> ExecutionPlan {
        let ChunkAllocator {
            arena2chunks,
            all_chunks,
        } = self.allocator;
        let mut chunks = Vec::new();
        let mut arenas = Vec::with_capacity(arena2chunks.len());
        for (arena_id, arena) in arena2chunks.into_iter().enumerate() {
            let mut arena_size = 0usize;
            for cid in arena.owned {
                let state = &all_chunks[cid];
                chunks.push(ChunkInfo {
                    id: cid,
                    arena: arena_id,
                    size: state.size,
                    offset: arena_size,
                });
                arena_size += align_up(state.size);
            }
            arenas.push(ArenaInfo {
                id: arena_id,
                tier: arena.tier,
                size: arena_size,
            });
        }

        let mut events: Vec<EventInfo> = self
            .kernel_info
            .values()
            .map(|info| EventInfo {
                id: info.event,
                stream: info.stream,
            })
            .collect();
        events.sort_by_key(|e| e.id.index());

        ExecutionPlan {
            steps: self.steps,
            chunks,
            arenas,
            events,
            stream_count: self.num_streams,
        }
    }
}

fn build(
    schedule: &Schedule,
    num_streams: usize,
    placement: Placement,
    streamed: HashSet<ValueId>,
) -> ExecutionPlan {
    PrefetchScheduler::new(schedule, num_streams, placement, streamed).run()
}
