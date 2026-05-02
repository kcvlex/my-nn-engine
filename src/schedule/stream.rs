use std::collections::HashMap;

use crate::graph::operator::Operator;
use crate::graph::operator::TransferKind;
use crate::graph::ValueId;
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
        let (result, execution_order) = StreamAllocator::new(schedule, self.num_streams).run();

        let old_kernels = std::mem::take(&mut schedule.kernels);
        let kernel_map: HashMap<KernelId, Kernel> =
            old_kernels.iter().map(|(id, k)| (id, k.clone())).collect();
        let mut new_kernels = Kernels::default();
        let mut id_remap: HashMap<KernelId, KernelId> = HashMap::new();
        for old_kid in &execution_order {
            let new_kid = new_kernels.0.alloc(kernel_map[old_kid].clone());
            id_remap.insert(*old_kid, new_kid);
        }
        schedule.kernels = new_kernels;

        let remapped_result = result
            .into_iter()
            .map(|(old_kid, assign)| (id_remap[&old_kid], assign))
            .collect();
        schedule.analysis.insert(StreamAllocResult(remapped_result));
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

struct EventTracker {
    num_streams: usize,
    event2stream: Vec<StreamId>,
    kernel2event: HashMap<KernelId, EventId>,
    waiting_events: Vec<Vec<Option<EventId>>>,
    event_slot: usize,
}

impl EventTracker {
    fn new(num_streams: usize) -> Self {
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

pub struct KernelStreamAssignment {
    pub stream_id: StreamId,
    pub event_id: EventId,
    pub to_wait: Vec<EventId>,
}

struct StreamAllocator<'sched> {
    schedule: &'sched Schedule,
    event_tracker: EventTracker,
    kernel2order: HashMap<KernelId, usize>,
    kernel2stream: HashMap<KernelId, StreamId>,
    stream_load: Vec<usize>,
    value2defined: HashMap<ValueId, KernelId>,
}

impl<'sched> StreamAllocator<'sched> {
    fn new(schedule: &'sched Schedule, num_streams: usize) -> Self {
        let mut value2defined = HashMap::new();
        for (kernel_id, kernel) in schedule.kernels.iter() {
            for output in &kernel.outputs {
                value2defined.insert(*output, kernel_id);
            }
        }
        Self {
            schedule,
            event_tracker: EventTracker::new(num_streams),
            kernel2order: HashMap::new(),
            kernel2stream: HashMap::new(),
            stream_load: vec![0; num_streams],
            value2defined,
        }
    }

    fn find_h2d_producer(&self, value: ValueId) -> Option<KernelId> {
        let kid = self.value2defined.get(&value)?;
        let kernel = &self.schedule.kernels[*kid];
        if matches_opaque!(kernel, Operator::Transfer(TransferKind::HostToDevice)) {
            Some(*kid)
        } else {
            None
        }
    }

    fn select_least_loaded_stream(&self) -> StreamId {
        let (idx, _) = self
            .stream_load
            .iter()
            .enumerate()
            .min_by_key(|(_, load)| **load)
            .unwrap();
        StreamId(idx)
    }

    fn assign(&mut self, kernel_id: KernelId, stream_id: StreamId, order: usize) {
        self.kernel2order.insert(kernel_id, order);
        self.kernel2stream.insert(kernel_id, stream_id);
        self.stream_load[stream_id.index()] += 1;
    }

    // for kernel in non_h2d_kernels (topological order):
    //     to_allocate = []
    //     depends_on = []
    //     for input in kernel.inputs:
    //         if input is produced by H2D (and not yet allocated):
    //             to_allocate.append(h2d_kernel)
    //         else:
    //             depends_on.append(producing_kernel)
    //     to_allocate.append(kernel)
    //
    //     if depends_on is not empty:
    //         stream = stream of the latest dependency (by topo order)
    //     else:
    //         stream = least loaded stream
    //
    //     for k in to_allocate:
    //         assign k to stream
    //
    // This places each H2D transfer on the same stream as its consumer, just before it, enabling H2D/compute overlap across streams.
    fn run(&mut self) -> (HashMap<KernelId, KernelStreamAssignment>, Vec<KernelId>) {
        let compute_kernels: Vec<KernelId> = self
            .schedule
            .kernels
            .iter()
            .filter(|(_, k)| !matches_opaque!(k, Operator::Transfer(TransferKind::HostToDevice)))
            .map(|(id, _)| id)
            .collect();

        let mut order_counter = 0;
        let mut execution_order: Vec<KernelId> = Vec::new();

        for &kernel_id in &compute_kernels {
            let kernel = &self.schedule.kernels[kernel_id];

            let mut to_allocate = Vec::new();
            let mut depends_on = Vec::new();

            for input in kernel.inputs.iter().flatten() {
                if let Some(h2d_kid) = self.find_h2d_producer(*input) {
                    if !self.kernel2order.contains_key(&h2d_kid) {
                        to_allocate.push(h2d_kid);
                    }
                } else if let Some(&dep_kid) = self.value2defined.get(input) {
                    if dep_kid != kernel_id {
                        depends_on.push(dep_kid);
                    }
                }
            }
            to_allocate.push(kernel_id);

            let stream = if !depends_on.is_empty() {
                let latest = depends_on
                    .iter()
                    .filter_map(|kid| self.kernel2order.get(kid).map(|order| (*order, *kid)))
                    .max_by_key(|(order, _)| *order);
                if let Some((_, latest_kid)) = latest {
                    self.kernel2stream[&latest_kid]
                } else {
                    self.select_least_loaded_stream()
                }
            } else {
                self.select_least_loaded_stream()
            };

            for kid in &to_allocate {
                self.assign(*kid, stream, order_counter);
                order_counter += 1;
                execution_order.push(*kid);
            }
        }

        let mut result = HashMap::new();
        for &kernel_id in &execution_order {
            let stream_id = self.kernel2stream[&kernel_id];
            let event_id = self.event_tracker.new_event(kernel_id, stream_id);

            let kernel = &self.schedule.kernels[kernel_id];
            for input in kernel.inputs.iter().flatten() {
                if let Some(&dep_kid) = self.value2defined.get(input) {
                    if dep_kid != kernel_id && self.kernel2order.contains_key(&dep_kid) {
                        self.event_tracker.wait_kernel(event_id, dep_kid);
                    }
                }
            }

            let to_wait = self.event_tracker.waiting_events[event_id.index()]
                .iter()
                .filter_map(|x| *x)
                .collect::<Vec<_>>();

            result.insert(
                kernel_id,
                KernelStreamAssignment {
                    stream_id,
                    event_id,
                    to_wait,
                },
            );
        }

        (result, execution_order)
    }
}
