use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::graph::operator::args;
use crate::graph::operator::Operator;
use crate::graph::ValueId;
use crate::schedule::*;

const ALIGNMENT: usize = 256;

fn align_up(size: usize) -> usize {
    size.next_multiple_of(ALIGNMENT)
}

pub struct MemoryAwareSchedulePass {
    pub num_streams: usize,
}

impl SchedulePass for MemoryAwareSchedulePass {
    fn summary(&self) -> &str {
        "Memory-aware schedule (experimental)"
    }

    fn run(&self, schedule: &mut Schedule) {
        let plan = build(schedule, self.num_streams);
        schedule.execution_plan = Some(plan);
    }
}

struct Deps {
    value2producer: HashMap<ValueId, KernelId>,
    value_uses_count: HashMap<ValueId, usize>,
    kernel_preds: HashMap<KernelId, HashSet<KernelId>>,
    kernel_consumers: HashMap<KernelId, HashSet<KernelId>>,
}

impl Deps {
    fn new(schedule: &Schedule) -> Self {
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

struct ChunkState {
    arena_id: ArenaId,
    size: usize,
    live_uses: usize,
    first_use: bool,
}

struct ArenaChunks {
    owned: Vec<ChunkId>,
    free_per_stream: Vec<Vec<ChunkId>>,
}

impl ArenaChunks {
    fn new(num_streams: usize) -> Self {
        Self {
            owned: Vec::new(),
            free_per_stream: vec![Vec::new(); num_streams],
        }
    }
}

#[derive(Default)]
struct ChunkAllocator {
    arena2chunks: Vec<ArenaChunks>,
    all_chunks: Vec<ChunkState>,
}

impl ChunkAllocator {
    fn new_arena(&mut self, num_streams: usize) -> ArenaId {
        let arena_id = self.arena2chunks.len();
        self.arena2chunks.push(ArenaChunks::new(num_streams));
        arena_id
    }

    fn alloc(&mut self, size: usize, arena_id: ArenaId, stream: StreamId, uses: usize) -> ChunkId {
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

    fn add_uses(&mut self, id: ChunkId, uses: usize) {
        self.all_chunks[id].live_uses += uses;
    }

    fn mark_as_first_use(&mut self, id: ChunkId) -> bool {
        let res = !self.all_chunks[id].first_use;
        self.all_chunks[id].first_use = true;
        res
    }

    fn consume(&mut self, id: ChunkId, stream: StreamId) {
        let entry = &mut self.all_chunks[id];
        entry.live_uses = entry.live_uses.saturating_sub(1);
        if entry.live_uses == 0 {
            self.free(id, stream);
        }
    }

    fn free(&mut self, id: ChunkId, stream: StreamId) {
        let arena_id = self.all_chunks[id].arena_id;
        self.arena2chunks[arena_id].free_per_stream[stream.index()].push(id);
    }
}

fn value_byte_size(schedule: &Schedule, v: ValueId) -> usize {
    let rty = schedule
        .get_resolved_tensor_type(v)
        .unwrap_or_else(|| panic!("unresolved tensor type for {v:?}"));
    rty.storage_num_elements() * (rty.elem_type.bit_width() / 8)
}

fn must_in_place_input(op: &Operator) -> Option<usize> {
    match op {
        Operator::KVCacheUpdate => Some(args::KVCACHE_UPDATE_CACHE),
        Operator::QuantizingKVCacheUpdate => Some(args::QKVCACHE_UPDATE_CACHE),
        _ => None,
    }
}

fn compute_output_aliases(schedule: &Schedule) -> HashMap<ValueId, ValueId> {
    use std::collections::hash_map::Entry;
    let mut alias: HashMap<ValueId, ValueId> = HashMap::new();
    for &out in &schedule.outputs {
        alias.insert(out, out);
    }

    loop {
        let mut changed = false;
        for (_, kernel) in schedule.kernels.iter() {
            let is_passthrough = match &kernel.body {
                KernelBody::Opaque(Opaque { op }) => {
                    matches!(op, Operator::Identity | Operator::Reinterpret(_))
                }
                _ => false,
            };
            if !is_passthrough {
                continue;
            }
            let Some(out0) = kernel.outputs.first().copied() else {
                continue;
            };
            let Some(in0) = kernel.inputs.first().and_then(|x| *x) else {
                continue;
            };
            if let Some(&target) = alias.get(&out0) {
                if let Entry::Vacant(e) = alias.entry(in0) {
                    e.insert(target);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    alias
}

struct KernelInfo {
    order: usize,
    stream: StreamId,
    event: EventId,
    device: Device,
}

struct Scheduler<'s> {
    schedule: &'s Schedule,
    deps: Deps,
    output_alias: HashMap<ValueId, ValueId>,
    initializers: HashSet<ValueId>,
    session_states: HashSet<ValueId>,
    inputs_set: HashSet<ValueId>,
    device: Device,

    // TODO: Per arena
    num_streams: usize,

    allocator: ChunkAllocator,
    tier2arena: BTreeMap<MemoryTier, ArenaId>,

    value2place: HashMap<ValueId, AllocPlace>,
    value_remaining_uses: HashMap<ValueId, usize>,
    kernel_pending: BTreeMap<KernelId, usize>,
    ready: BTreeSet<KernelId>,
    kernel_info: HashMap<KernelId, KernelInfo>,
    stream_load: Vec<usize>,
    steps: Vec<Step>,
}

impl<'s> Scheduler<'s> {
    fn new(schedule: &'s Schedule, num_streams: usize) -> Self {
        let deps = Deps::new(schedule);
        let output_alias = compute_output_aliases(schedule);
        let num_streams = match schedule.options.target {
            Target::CUDA => num_streams.max(1),
            Target::CPU => 1,
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

        let device = match schedule.options.target {
            Target::CUDA => Device::CUDA,
            Target::CPU => Device::CPU,
        };

        let kernel_pending: BTreeMap<KernelId, usize> = deps
            .kernel_preds
            .iter()
            .map(|(k, p)| (*k, p.len()))
            .collect();
        let ready: BTreeSet<KernelId> = kernel_pending
            .iter()
            .filter(|(_, c)| **c == 0)
            .map(|(k, _)| *k)
            .collect();

        Self {
            schedule,
            deps,
            output_alias,
            initializers,
            session_states,
            inputs_set,
            device,
            num_streams,
            allocator: ChunkAllocator::default(),
            tier2arena: BTreeMap::new(),
            value2place,
            value_remaining_uses,
            kernel_pending,
            ready,
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
        use std::collections::btree_map::Entry;
        let arena_id = match self.tier2arena.entry(tier) {
            Entry::Occupied(e) => *e.get(),
            Entry::Vacant(e) => {
                let arena_id = self.allocator.new_arena(self.num_streams);
                e.insert(arena_id);
                arena_id
            }
        };
        self.allocator.alloc(size, arena_id, stream, uses)
    }

    fn run(mut self) -> ExecutionPlan {
        while let Some(kid) = self.pick_best() {
            self.ready.remove(&kid);
            self.schedule_kernel(kid);
        }
        self.into_plan()
    }

    fn pick_best(&self) -> Option<KernelId> {
        // Pick the ready kernel that minimises the estimated change in arena occupancy.
        //
        //   score(k) = sum(output sizes) - sum(input sizes whose remaining_uses == 1)
        //
        // Tie-break by smallest KernelId for determinism.
        let score = |kid: KernelId| -> i64 {
            let kernel = &self.schedule.kernels[kid];
            let mut delta: i64 = 0;
            for o in &kernel.outputs {
                delta += value_byte_size(self.schedule, *o) as i64;
            }
            for input in kernel.inputs.iter().flatten() {
                if self.value_remaining_uses.get(input).copied().unwrap_or(0) == 1 {
                    delta -= value_byte_size(self.schedule, *input) as i64;
                }
            }
            delta
        };
        self.ready
            .iter()
            .copied()
            .min_by_key(|&kid| (score(kid), kid))
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

    /// Choose the device for `kid`. Currently every kernel runs on the
    /// target device; once hybrid placement lands this is where heuristics
    /// or per-kernel hints will plug in.
    fn pick_device(&self, _kid: KernelId) -> Device {
        self.device
    }

    fn schedule_kernel(&mut self, kid: KernelId) {
        let device = self.pick_device(kid);
        let stream = self.pick_stream(kid);
        let order = self.kernel_info.len();
        let event = EventId(order);
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

        // Cross-stream wait events.
        for pred in &self.deps.kernel_preds[&kid] {
            let pred_info = &self.kernel_info[pred];
            if pred_info.stream != stream {
                self.steps.push(Step::SyncWait(SyncWaitStep {
                    context,
                    event: pred_info.event,
                }));
            }
        }

        let bindings = self.bind_kernel(kid, stream);

        let is_transfer = matches!(
            &self.schedule.kernels[kid].body,
            KernelBody::Opaque(Opaque {
                op: Operator::Transfer(_)
            })
        );
        if is_transfer {
            // Operator::Transfer: 1 input + 1 output. Lower to Step::Transfer
            // so the codegen treats it as a memcpy rather than a kernel launch.
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

        // Decrement remaining uses; release chunks whose live count hits 0.
        for input in self.schedule.kernels[kid].inputs.clone().iter().flatten() {
            if let Some(count) = self.value_remaining_uses.get_mut(input) {
                *count -= 1;
            }
            if let Some(AllocPlace::Chunk(cid)) = self.value2place.get(input).copied() {
                self.allocator.consume(cid, stream);
            }
        }

        // Advance ready set.
        let consumers: Vec<KernelId> = self.deps.kernel_consumers[&kid].iter().copied().collect();
        for cons in consumers {
            let count = self.kernel_pending.get_mut(&cons).unwrap();
            *count -= 1;
            if *count == 0 {
                self.ready.insert(cons);
            }
        }
    }

    fn bind_kernel(&mut self, kid: KernelId, stream: StreamId) -> Vec<ValueBinding> {
        let mut bindings = Vec::new();
        let inputs = self.schedule.kernels[kid].inputs.clone();
        let tier = self.kernel_info[&kid].device.tier();

        for input in inputs.iter().flatten().copied() {
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
                let size = value_byte_size(self.schedule, input);
                let uses = self.value_remaining_uses.get(&input).copied().unwrap_or(0);
                let cid = self.alloc_chunk(tier, size, stream, uses);
                self.value2place.insert(input, AllocPlace::Chunk(cid));
                AllocPlace::Chunk(cid)
            } else {
                *self
                    .value2place
                    .get(&input)
                    .unwrap_or_else(|| panic!("unresolved place for {input:?}"))
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

        // Alias Identity/Reinterpret output to its input's place so codegen
        // elides the copy. Chunk inputs need single-use; persistent places
        // are always safe.
        let try_in_place_identity = match &self.schedule.kernels[kid].body {
            KernelBody::Opaque(Opaque {
                op: Operator::Identity | Operator::Reinterpret(_),
            }) => inputs[0].and_then(|in0| match self.value2place.get(&in0).copied() {
                Some(p @ AllocPlace::Chunk(_)) => {
                    if self.value_remaining_uses.get(&in0).copied().unwrap_or(0) == 1 {
                        Some(p)
                    } else {
                        None
                    }
                }
                place => place,
            }),
            _ => None,
        };

        let must_in_place = match &self.schedule.kernels[kid].body {
            KernelBody::Opaque(Opaque { op }) => must_in_place_input(op),
            _ => None,
        };

        let outputs = self.schedule.kernels[kid].outputs.clone();
        for output in outputs {
            let uses = self.value_remaining_uses.get(&output).copied().unwrap_or(0);
            let (place, allocated) = if let Some(&target) = self.output_alias.get(&output) {
                (AllocPlace::Output(target), false)
            } else if let Some(p) = try_in_place_identity {
                (p, false)
            } else if let Some(idx) = must_in_place {
                let input_v = inputs[idx].expect("must_in_place input is None");
                let p = *self
                    .value2place
                    .get(&input_v)
                    .expect("must_in_place input not yet placed");
                (p, false)
            } else {
                let size = value_byte_size(self.schedule, output);
                let cid = self.alloc_chunk(tier, size, stream, uses);
                (AllocPlace::Chunk(cid), true)
            };

            self.value2place.insert(output, place);

            // Output placed on an already-allocated chunk via aliasing: bump
            // the live count so the chunk survives until this output is also
            // fully consumed. `alloc` already accounted for the freshly-
            // allocated path.
            if let (AllocPlace::Chunk(cid), false) = (place, allocated) {
                self.allocator.add_uses(cid, uses);
            }

            let is_first_use =
                matches!(place, AllocPlace::Chunk(cid) if self.allocator.mark_as_first_use(cid));

            bindings.push(ValueBinding {
                value: output,
                role: BindingRole::Output,
                place,
                is_first_use,
            });
        }

        bindings
    }

    fn into_plan(self) -> ExecutionPlan {
        let mut chunks = Vec::new();
        let mut arenas = Vec::with_capacity(self.tier2arena.len());
        for (tier, arena_id) in self.tier2arena {
            let mut arena_size = 0usize;
            for id in &self.allocator.arena2chunks[arena_id].owned {
                let state = &self.allocator.all_chunks[*id];
                chunks.push(ChunkInfo {
                    id: *id,
                    arena: arena_id,
                    size: state.size,
                    offset: arena_size,
                });
                arena_size += align_up(state.size);
            }
            arenas.push(ArenaInfo {
                id: arena_id,
                tier,
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

fn build(schedule: &Schedule, num_streams: usize) -> ExecutionPlan {
    Scheduler::new(schedule, num_streams).run()
}
