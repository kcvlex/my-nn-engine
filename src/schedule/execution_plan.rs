use std::collections::HashMap;
use std::collections::HashSet;

use crate::graph::ValueId;
use crate::schedule::mem_alloc;
use crate::schedule::stream;
use crate::schedule::*;

const ALIGNMENT: usize = 256;

fn align_up(size: usize) -> usize {
    (size + ALIGNMENT - 1) & !(ALIGNMENT - 1)
}

pub struct BuildExecutionPlanPass;

impl SchedulePass for BuildExecutionPlanPass {
    fn summary(&self) -> &str {
        "Build ExecutionPlan IR"
    }

    fn run(&self, schedule: &mut Schedule) {
        let plan = build(schedule);
        schedule.execution_plan = Some(plan);
    }
}

fn build(schedule: &Schedule) -> ExecutionPlan {
    let mem_alloc_result = schedule.analysis.get::<mem_alloc::MemAllocResult>();

    let value2alloc: HashMap<ValueId, AllocateInfo> = mem_alloc_result
        .0
        .values()
        .flatten()
        .map(|info| (info.value_id, *info))
        .collect();

    let mut defined: HashSet<ValueId> = HashSet::new();
    for (_, k) in schedule.kernels.iter() {
        for &out in &k.outputs {
            defined.insert(out);
        }
    }

    let initializers: HashSet<ValueId> = schedule.initializers.iter().copied().collect();
    let session_states: HashSet<ValueId> = schedule.session_states.iter().copied().collect();
    let inputs_set: HashSet<ValueId> = schedule.inputs.iter().copied().collect();
    let outputs_set: HashSet<ValueId> = schedule.outputs.iter().copied().collect();

    let n_chunks = schedule.max_chunk_id().map(|id| id + 1).unwrap_or(0);
    let mut chunk_max_size = vec![0usize; n_chunks];
    for info in mem_alloc_result.0.values().flatten() {
        if let AllocateType::Chunk(cid) = info.ty {
            let rty = schedule.get_resolved_tensor_type(info.value_id).unwrap();
            let bytes = rty.storage_num_elements() * (rty.elem_type.bit_width() / 8);
            chunk_max_size[cid] = chunk_max_size[cid].max(bytes);
        }
    }

    let arena_id: ArenaId = 0;
    let mut chunks = Vec::with_capacity(n_chunks);
    let mut arena_size = 0usize;
    for (cid, &size) in chunk_max_size.iter().enumerate() {
        chunks.push(ChunkInfo {
            id: cid,
            arena: arena_id,
            size,
            offset: arena_size,
        });
        arena_size += align_up(size);
    }
    let arenas = vec![ArenaInfo {
        id: arena_id,
        tier: MemoryTier::DeviceArena,
        size: arena_size,
    }];

    let (stream_count, cuda_stream_data) = match schedule.options.target {
        Target::CUDA => {
            let r = schedule.analysis.get::<stream::StreamAllocResult>();
            let max_sid = r.0.values().map(|s| s.stream_id.index()).max().unwrap_or(0);
            (max_sid + 1, Some(r))
        }
        Target::CPU => (1, None),
    };

    let mut events: HashMap<EventId, EventInfo> = HashMap::new();
    let mut steps: Vec<Step> = Vec::new();

    for (kernel_id, kernel) in schedule.kernels.iter() {
        let (stream_id, event_id, to_wait): (StreamId, Option<EventId>, Vec<EventId>) =
            match cuda_stream_data {
                Some(r) => {
                    let a = &r.0[&kernel_id];
                    (a.stream_id, Some(a.event_id), a.to_wait.clone())
                }
                None => (StreamId(0), None, Vec::new()),
            };

        for evt in &to_wait {
            steps.push(Step::SyncWait {
                stream: stream_id,
                event: *evt,
            });
        }

        let kernel_alloc_infos: HashMap<ValueId, AllocateInfo> = mem_alloc_result
            .0
            .get(&kernel_id)
            .map(|infos| infos.iter().map(|i| (i.value_id, *i)).collect())
            .unwrap_or_default();

        let mut bindings = Vec::new();

        for input in kernel.inputs.iter().flatten() {
            if initializers.contains(input) {
                bindings.push(ValueBinding {
                    value: *input,
                    role: BindingRole::Input,
                    place: AllocPlace::Initializer(*input),
                    is_first_use: false,
                });
                continue;
            }
            let role = if defined.contains(input) || inputs_set.contains(input) {
                BindingRole::Input
            } else if session_states.contains(input) {
                BindingRole::Input
            } else {
                BindingRole::Workspace
            };
            let place = resolve_place(
                *input,
                &value2alloc,
                &session_states,
                &inputs_set,
                &outputs_set,
            );
            let is_first_use = kernel_alloc_infos
                .get(input)
                .map(|i| i.is_first_use)
                .unwrap_or(false);
            bindings.push(ValueBinding {
                value: *input,
                role,
                place,
                is_first_use,
            });
        }

        for output in kernel.outputs.iter() {
            let place = resolve_place(
                *output,
                &value2alloc,
                &session_states,
                &inputs_set,
                &outputs_set,
            );
            let is_first_use = kernel_alloc_infos
                .get(output)
                .map(|i| i.is_first_use)
                .unwrap_or(false);
            bindings.push(ValueBinding {
                value: *output,
                role: BindingRole::Output,
                place,
                is_first_use,
            });
        }

        if let Some(eid) = event_id {
            events.entry(eid).or_insert(EventInfo {
                id: eid,
                stream: stream_id,
            });
        }

        steps.push(Step::Kernel(KernelStep {
            kernel: kernel_id,
            stream: stream_id,
            bindings,
            records_event: event_id,
        }));
    }

    let mut events_v: Vec<EventInfo> = events.into_values().collect();
    events_v.sort_by_key(|e| e.id.index());

    ExecutionPlan {
        steps,
        chunks,
        arenas,
        events: events_v,
        stream_count,
    }
}

fn resolve_place(
    value: ValueId,
    value2alloc: &HashMap<ValueId, AllocateInfo>,
    session_states: &HashSet<ValueId>,
    inputs_set: &HashSet<ValueId>,
    outputs_set: &HashSet<ValueId>,
) -> AllocPlace {
    if let Some(info) = value2alloc.get(&value) {
        return match info.ty {
            AllocateType::Chunk(cid) => AllocPlace::Chunk(cid),
            AllocateType::SessionState(v) => AllocPlace::SessionState(v),
            AllocateType::Input(v) => AllocPlace::Input(v),
            AllocateType::Output(v) => AllocPlace::Output(v),
        };
    }
    if session_states.contains(&value) {
        return AllocPlace::SessionState(value);
    }
    if inputs_set.contains(&value) {
        return AllocPlace::Input(value);
    }
    if outputs_set.contains(&value) {
        return AllocPlace::Output(value);
    }
    panic!("unresolved place for value {value:?}");
}
