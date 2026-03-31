use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;

use indexmap::IndexSet;

use crate::onnx::model::ValueId;
use crate::schedule::*;

pub struct StreamAllocResult(pub HashMap<KernelId, KernelStreamAssignment>);

pub struct StreamAllocPass {
    pub num_streams: usize,
}

impl SchedulePass for StreamAllocPass {
    fn summary(&self) -> &str {
        "CUDA stream allocation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let result = StreamAllocator::new(schedule, self.num_streams).run();
        schedule.analysis.insert(StreamAllocResult(result));
    }
}

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug, Ord, PartialOrd)]
pub struct StreamId(usize);

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
pub struct EventId(usize);

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

fn build_chunk_deps(sched: &Schedule) -> HashMap<KernelId, Vec<KernelId>> {
    let value2used = {
        let mut res: HashMap<ValueId, Vec<KernelId>> = HashMap::new();
        let outputs = sched.outputs.iter().collect::<HashSet<_>>();
        for (kernel_id, kernel) in sched.kernels.iter() {
            for v in kernel
                .inputs
                .iter()
                .chain(kernel.outputs.iter().filter(|o| outputs.contains(o)))
            {
                res.entry(*v).or_default().push(kernel_id);
            }
        }
        res
    };

    let mut value2defined = {
        let mut res: HashMap<ValueId, KernelId> = HashMap::new();
        for (kernel_id, kernel) in sched.kernels.iter() {
            for output in kernel.outputs.iter() {
                match res.entry(*output) {
                    Entry::Occupied(_) => {
                        panic!("value defined multiple times");
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(kernel_id);
                    }
                }
            }
        }
        res
    };

    let value2chunk = {
        // TODO: Consolidate with the implementation in HostCodeGenerator.
        let mem_alloc_result = sched.analysis.get::<super::mem_alloc::MemAllocResult>();
        let mut res = HashMap::new();
        for mem in mem_alloc_result.0.values().flatten() {
            let chunk_id = if let AllocateType::Chunk(chunk_id) = mem.ty {
                chunk_id
            } else {
                unreachable!("non chunk");
            };

            match res.entry(mem.value_id) {
                Entry::Occupied(entry) => {
                    assert!(*entry.get() == chunk_id);
                }
                Entry::Vacant(entry) => {
                    entry.insert(chunk_id);
                }
            }
        }
        res
    };

    let mut from_host = sched
        .inputs
        .iter()
        .chain(sched.initializers.iter())
        .copied()
        .collect::<HashSet<_>>();
    let mut chunk2current_value: HashMap<ChunkId, ValueId> = HashMap::new();
    let mut res: HashMap<KernelId, Vec<KernelId>> = HashMap::new();

    for (kernel_id, kernel) in sched.kernels.iter() {
        let mut deps = Vec::new();

        for input in kernel.inputs.iter() {
            let chunk = value2chunk[input];
            if from_host.contains(input) {
                match chunk2current_value.entry(chunk) {
                    Entry::Occupied(mut entry) => {
                        let value_id = entry.get();
                        assert!(*input != *value_id);
                        if let Some(users) = value2used.get(value_id) {
                            for user in users.iter() {
                                deps.push(*user);
                            }
                        }
                        entry.insert(*input);
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(*input);
                    }
                }
                from_host.remove(input);
                value2defined.insert(*input, kernel_id);
            } else {
                assert!(chunk2current_value[&chunk] == *input);
                deps.push(value2defined[input]);
            }
        }

        for output in kernel.outputs.iter() {
            let chunk = value2chunk[output];

            if let Some(cur) = chunk2current_value.get(&chunk) {
                for user in value2used[cur].iter().filter(|u| **u != kernel_id) {
                    deps.push(*user);
                }
            }

            chunk2current_value.insert(chunk, *output);
        }

        // Assert.
        for dep in deps.iter() {
            assert!(*dep != kernel_id);
        }

        res.insert(kernel_id, deps);
    }

    res
}

struct EventTracker {
    num_streams: usize,

    event2stream: Vec<StreamId>,
    kernel2event: HashMap<KernelId, EventId>,

    waiting_events: Vec<Vec<Option<EventId>>>,

    event_slot: usize,
}

impl EventTracker {
    pub fn new(num_streams: usize) -> Self {
        Self {
            num_streams,
            event2stream: Vec::new(),
            kernel2event: HashMap::new(),
            waiting_events: Vec::new(),
            event_slot: 0,
        }
    }

    fn new_event(&mut self, kernel_id: KernelId, stream_id: StreamId) -> EventId {
        let event_id = EventId(self.event_slot);
        self.event_slot += 1;
        self.event2stream.push(stream_id);
        self.kernel2event.insert(kernel_id, event_id);
        self.waiting_events.push(vec![None; self.num_streams]);
        event_id
    }

    fn wait_kernel(&mut self, event_id: EventId, kernel_id: KernelId) {
        let wait = self.kernel2event[&kernel_id];
        let dst_stream = self.event2stream[event_id.index()];
        let src_stream = self.event2stream[wait.index()];
        if dst_stream == src_stream {
            assert!(wait < event_id);
            return;
        }
        let entry = &mut self.waiting_events[event_id.index()];
        let value = match entry[src_stream.index()] {
            Some(existing) => std::cmp::max(existing, wait),
            None => wait,
        };
        entry[src_stream.index()] = Some(value);
    }
}

struct StreamAllocator<'sched> {
    schedule: &'sched Schedule,

    streams: IndexSet<StreamId>,
    running: Vec<Option<KernelId>>,
    event_tracker: EventTracker,
    chunk_deps: HashMap<KernelId, Vec<KernelId>>,
}

pub struct KernelStreamAssignment {
    pub stream_id: StreamId,
    pub event_id: EventId,
    pub to_wait: Vec<EventId>,
}

impl<'sched> StreamAllocator<'sched> {
    fn new(schedule: &'sched Schedule, num_streams: usize) -> Self {
        let streams = (0..num_streams).map(StreamId).collect();
        Self {
            schedule,
            streams,
            running: vec![None; num_streams],
            event_tracker: EventTracker::new(num_streams),
            chunk_deps: build_chunk_deps(schedule),
        }
    }

    fn select_stream(&self, kernel_id: KernelId) -> StreamId {
        let kernel = &self.schedule.kernels[kernel_id];
        for stream_id in self.streams.iter().rev() {
            let Some(running) = self.running[stream_id.index()] else {
                continue;
            };
            let running = &self.schedule.kernels[running];
            if running
                .outputs
                .iter()
                .any(|output| kernel.inputs.contains(output))
            {
                return *stream_id;
            }
        }

        *self.streams.first().unwrap()
    }

    fn assign_stream(&mut self, kernel_id: KernelId, stream_id: StreamId) -> EventId {
        let event_id = self.event_tracker.new_event(kernel_id, stream_id);
        for dep in self.chunk_deps[&kernel_id].iter() {
            self.event_tracker.wait_kernel(event_id, *dep);
        }
        self.running[stream_id.index()] = Some(kernel_id);
        self.streams.shift_remove(&stream_id);
        self.streams.insert(stream_id);
        event_id
    }

    fn run(&mut self) -> HashMap<KernelId, KernelStreamAssignment> {
        self.schedule
            .kernels
            .iter()
            .map(|(kernel_id, _)| {
                let stream_id = self.select_stream(kernel_id);
                let event_id = self.assign_stream(kernel_id, stream_id);

                let to_wait = self.event_tracker.waiting_events[event_id.index()]
                    .iter()
                    .filter_map(|x| *x)
                    .collect::<Vec<_>>();

                let assign = KernelStreamAssignment {
                    stream_id,
                    event_id,
                    to_wait,
                };
                (kernel_id, assign)
            })
            .collect()
    }
}

pub fn allocate_streams(
    schedule: &Schedule,
    num_streams: usize,
) -> HashMap<KernelId, KernelStreamAssignment> {
    StreamAllocator::new(schedule, num_streams).run()
}
