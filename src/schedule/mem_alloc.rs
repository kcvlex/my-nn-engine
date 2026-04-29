use std::collections::BTreeSet;
use std::collections::HashMap;

use indexmap::IndexMap;
use indexmap::IndexSet;

use crate::onnx::model::ValueId;
use crate::onnx::operator::args;
use crate::onnx::operator::Operator;
use crate::schedule::*;

pub struct MemAllocResult(pub HashMap<KernelId, Vec<AllocateInfo>>);

pub struct MemAllocPass;

impl SchedulePass for MemAllocPass {
    fn summary(&self) -> &str {
        "Memory allocation"
    }

    fn run(&self, schedule: &mut Schedule) {
        let info_v = MemoryPlanner::new(schedule).run();
        let result = schedule
            .kernels
            .iter()
            .map(|(id, _)| id)
            .zip(info_v)
            .collect();
        schedule.analysis.insert(MemAllocResult(result));
    }
}

// TODO: Make the order deterministic
#[derive(Debug)]
struct DependencyGraph {
    value2defined: IndexMap<ValueId, KernelId>,
    value2used: IndexMap<ValueId, IndexSet<(KernelId, usize)>>,
    inputs_set: BTreeSet<ValueId>,
    outputs_set: BTreeSet<ValueId>,
}

impl DependencyGraph {
    fn new(schedule: &Schedule) -> Self {
        let mut value2defined = IndexMap::new();
        let mut value2used = IndexMap::new();
        let (inputs_set, outputs_set) = match schedule.options.target {
            Target::CPU => {
                let inputs_set = schedule
                    .inputs
                    .iter()
                    .chain(schedule.initializers.iter())
                    .copied()
                    .collect::<BTreeSet<_>>();
                let outputs_set = schedule.outputs.iter().copied().collect::<BTreeSet<_>>();
                (inputs_set, outputs_set)
            }
            Target::CUDA => (BTreeSet::new(), BTreeSet::new()),
        };
        let ignore = |x: &ValueId| inputs_set.contains(x) || outputs_set.contains(x);
        for (kernel_id, kernel) in schedule.kernels.0.iter() {
            // if node.is_dummy() {
            //     continue;
            // }

            for (i, input) in kernel.inputs.iter().enumerate() {
                let Some(input) = input else {
                    continue;
                };
                if ignore(input) {
                    continue;
                }
                value2used
                    .entry(*input)
                    .or_insert_with(IndexSet::new)
                    .insert((kernel_id, i));
            }
            for &output in kernel.outputs.iter().filter(|x| !ignore(x)) {
                value2defined.insert(output, kernel_id);
            }
        }
        Self {
            value2defined,
            value2used,
            inputs_set,
            outputs_set,
        }
    }
}

struct Chunks {
    free_lists: Vec<Vec<ChunkId>>,
    slot: usize,
}

impl Chunks {
    fn new(num_streams: usize) -> Self {
        Self {
            free_lists: vec![Vec::new(); num_streams.max(1)],
            slot: 0,
        }
    }

    fn allocate(&mut self) -> ChunkId {
        let id = self.slot;
        self.slot += 1;
        id
    }

    fn free(&mut self, id: ChunkId, stream: usize) {
        self.free_lists[stream].push(id);
    }

    fn reuse_or_new(&mut self, stream: usize) -> ChunkId {
        self.free_lists[stream]
            .pop()
            .unwrap_or_else(|| self.allocate())
    }
}

pub(super) struct MemoryPlanner<'sched> {
    schedule: &'sched Schedule,
    deps: DependencyGraph,
    liveness_counter: IndexMap<ChunkId, usize>,
    chunks: Chunks,
    allocations: HashMap<ValueId, AllocateType>,
    kernel2stream: HashMap<KernelId, usize>,
}

impl<'sched> MemoryPlanner<'sched> {
    pub(super) fn new(schedule: &'sched Schedule) -> Self {
        let deps = DependencyGraph::new(schedule);

        let (num_streams, kernel2stream) = match schedule.options.target {
            Target::CUDA => {
                let stream_result = schedule.analysis.get::<super::stream::StreamAllocResult>();
                let mut kernel2stream = HashMap::new();
                let mut max_stream = 0;
                for (kid, assign) in stream_result.0.iter() {
                    let s = assign.stream_id.index();
                    kernel2stream.insert(*kid, s);
                    max_stream = max_stream.max(s);
                }
                (max_stream + 1, kernel2stream)
            }
            Target::CPU => {
                let kernel2stream = schedule.kernels.iter().map(|(kid, _)| (kid, 0)).collect();
                (1, kernel2stream)
            }
        };

        MemoryPlanner {
            schedule,
            deps,
            liveness_counter: IndexMap::new(),
            chunks: Chunks::new(num_streams),
            allocations: HashMap::new(),
            kernel2stream,
        }
    }

    // ref: https://arxiv.org/pdf/1604.06174
    pub(super) fn run(&mut self) -> Vec<Vec<AllocateInfo>> {
        for input in self.deps.inputs_set.iter() {
            self.allocations.insert(*input, AllocateType::Input(*input));
        }
        for output in self.deps.outputs_set.iter() {
            self.allocations
                .insert(*output, AllocateType::Output(*output));
        }

        let mut extra_inputs: HashMap<KernelId, Vec<ValueId>> = HashMap::new();
        for (kernel_id, _) in self.schedule.kernels.iter() {
            let allocated = self.run_kernel(kernel_id);
            for (value_id, ref alloc) in &allocated {
                if matches!(alloc, AllocateType::Chunk(_)) {
                    // Track inputs that got newly allocated (e.g., workspace)
                    let kernel = &self.schedule.kernels[kernel_id];
                    if kernel.inputs.iter().flatten().any(|id| id == value_id) {
                        extra_inputs.entry(kernel_id).or_default().push(*value_id);
                    }
                }
            }
            for (value_id, allocated) in allocated {
                self.allocations.insert(value_id, allocated);
            }
        }

        self.coalesce_output();

        let mut used = vec![false; self.chunks.slot];
        let mut to_allocate_info = |s: &Self, id: &ValueId| {
            let chunk = s.allocations.get(id).unwrap();
            let is_first_use = if let AllocateType::Chunk(chunk) = chunk {
                let res = !used[*chunk];
                used[*chunk] = true;
                res
            } else {
                false
            };
            AllocateInfo {
                value_id: *id,
                ty: *chunk,
                is_first_use,
            }
        };
        self.schedule
            .kernels
            .iter()
            .map(|(kernel_id, kernel)| match self.schedule.options.target {
                Target::CPU => {
                    let extras = extra_inputs.get(&kernel_id).cloned().unwrap_or_default();
                    extras
                        .iter()
                        .chain(kernel.outputs.iter())
                        .map(|id| to_allocate_info(self, id))
                        .collect()
                }
                Target::CUDA => kernel
                    .inputs
                    .iter()
                    .flatten()
                    .filter(|id| !self.schedule.graph.has_initializer(**id))
                    .chain(kernel.outputs.iter())
                    .map(|id| to_allocate_info(self, id))
                    .collect(),
            })
            .collect::<Vec<_>>()
    }

    fn coalesce_output(&mut self) {
        let mut output_set: HashMap<_, _> =
            self.deps.outputs_set.iter().map(|v| (*v, *v)).collect();

        for (_, kernel) in self.schedule.kernels.0.iter().rev() {
            for (i, output) in kernel.outputs.iter().enumerate() {
                if let Some(v) = output_set.get(output) {
                    *self.allocations.get_mut(output).unwrap() = AllocateType::Output(*v);

                    // TODO: Support other patterns
                    if i == 0 &&
                        matches_opaque!(kernel, Operator::Identity | Operator::Reinterpret(_))
                    {
                        output_set.insert(kernel.inputs[0].unwrap(), *v);
                    }
                }
            }
        }
    }

    fn run_kernel(&mut self, kernel_id: KernelId) -> Vec<(ValueId, AllocateType)> {
        let mut res = Vec::new();
        let kernel = &self.schedule.kernels.0[kernel_id];
        let stream = self.kernel2stream.get(&kernel_id).copied().unwrap_or(0);

        for input in kernel.inputs.iter().flatten() {
            match self.allocations.get(input) {
                Some(_) => {}
                None => {
                    if self.deps.inputs_set.contains(input) ||
                        self.schedule.graph.has_initializer(*input)
                    {
                        continue;
                    }
                    let chunk_id = self.chunks.reuse_or_new(stream);
                    self.allocations
                        .insert(*input, AllocateType::Chunk(chunk_id));
                    *self.liveness_counter.entry(chunk_id).or_insert(0) +=
                        self.deps.value2used.get(input).map_or(1, |u| u.len());
                    res.push((*input, AllocateType::Chunk(chunk_id)));
                }
            };
        }

        for output in kernel.outputs.iter() {
            let chunk = if self.deps.outputs_set.contains(output) {
                AllocateType::Output(*output)
            } else {
                match self.try_in_place(*output) {
                    Some(prev) => *self.allocations.get(&prev).unwrap(),
                    None => AllocateType::Chunk(self.chunks.reuse_or_new(stream)),
                }
            };
            res.push((*output, chunk));
            if let AllocateType::Chunk(chunk) = chunk {
                if let Some(used) = self.deps.value2used.get(output) {
                    *self.liveness_counter.entry(chunk).or_insert(0) += used.len();
                }
            }
        }

        for input in kernel
            .inputs
            .iter()
            .flatten()
            .filter(|x| !self.deps.inputs_set.contains(x))
        {
            let chunk_id = match self.allocations.get(input).and_then(|x| x.chunk_id()) {
                Some(v) => v,
                None => continue,
            };
            let counter = self.liveness_counter.get_mut(&chunk_id).unwrap();
            *counter -= 1;
            if *counter == 0 {
                self.chunks.free(chunk_id, stream);
            }
        }
        res
    }

    // TODO: Identity and Reinterpret assume that the computation MUST be in-place. If the source
    // is not contiguous, we need to allocate a new chunk.
    fn try_in_place(&self, value_id: ValueId) -> Option<ValueId> {
        let kernel_id = self.deps.value2defined[&value_id];
        let kernel = &self.schedule.kernels.0[kernel_id];

        // Mandatory in-place: certain ops require the output to alias a specific input,
        // regardless of the input's liveness count. Ordering correctness is the graph
        // author's responsibility.
        if let KernelBody::Opaque(Opaque { op }) = &kernel.body {
            if let Some(idx) = must_in_place_input(op) {
                return kernel.inputs[idx];
            }
        }

        if !self.can_in_place(value_id) {
            return None;
        }
        for input in kernel.inputs.iter().flatten() {
            let is_input = match self.allocations.get(input) {
                Some(AllocateType::Input(_)) => true,
                Some(AllocateType::Chunk(chunk_id)) => {
                    if self.liveness_counter[chunk_id] != 1 {
                        continue;
                    }
                    false
                }
                Some(_) | None => continue,
            };

            if let KernelBody::Opaque(Opaque { op }) = &kernel.body {
                match op {
                    // TODO: Incorrect when the input is not contiguous for CUDA.
                    Operator::Identity | Operator::Reinterpret(_) => return Some(*input),
                    Operator::Gemm(_) => {
                        if !is_input &&
                            kernel
                                .inputs
                                .get(args::GEMM_C)
                                .and_then(|x| *x)
                                .map(|x| x == *input)
                                .unwrap_or(false)
                        {
                            return Some(*input);
                        } else {
                            continue;
                        }
                    }

                    // TODO: Unnecessary?
                    Operator::Conv(_) => continue,
                    _ => (),
                }
            }

            if is_input {
                continue;
            }

            if self
                .schedule
                .get_resolved_tensor_type(*input)
                .as_ref()
                .unwrap() ==
                self.schedule
                    .get_resolved_tensor_type(value_id)
                    .as_ref()
                    .unwrap()
            {
                return Some(*input);
            }
        }

        None
    }

    fn can_in_place(&self, value_id: ValueId) -> bool {
        let kernel_id = self.deps.value2defined[&value_id];
        match &self.schedule.kernels.0[kernel_id].body {
            KernelBody::Opaque(Opaque { op }) => match op {
                Operator::Identity | Operator::Reinterpret(_) => true,
                Operator::Gemm(_) => self.schedule.kernels.0[kernel_id]
                    .inputs
                    .get(args::GEMM_C)
                    .is_some(),
                _ => {
                    assert!(!op.is_elementwise());
                    false
                }
            },
            KernelBody::ElementWises(_) => true,
        }
    }
}

// Returns Some(input_index) if the op requires its output to alias the given input
// regardless of liveness count.
fn must_in_place_input(op: &Operator) -> Option<usize> {
    match op {
        Operator::KVCacheUpdate => Some(args::KVCACHE_UPDATE_CACHE),
        _ => None,
    }
}

#[cfg(test)]
mod test {
    use std::io::Error;
    use std::io::Result;
    use std::path::PathBuf;

    use itertools::Itertools;

    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::Model;
    use crate::options::Target;
    use crate::transform::lower::strides;
    use crate::transform::modify::SimpleGraphOp;
    use crate::transform::shape::*;
    use crate::transform::*;

    fn load_model(path: &str) -> Result<Model> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/schedule")
            .join(path);
        let mut model =
            Model::load_from_path(path).map_err(|e| Error::other(format!("{:?}", e)))?;
        let mut pass_manager = SimplePassManager::new("Shape".to_string());
        let target = Target::CPU;
        pass_manager.add_pass(Box::new(infer::ShapeInference { target }));
        pass_manager.add_pass(Box::new(strides::AssignStrides { target }));
        let mut modifier = SimpleGraphOp::new(&model.graph);
        pass_manager.run(&mut model.graph, &mut modifier);
        Ok(model)
    }

    // TODO: Insert Contiguous operator
    // Graph:
    //
    //               +-- 1.Sigmoid -- 3.Pool --+
    //              /                           \
    // Input -- 0.Sigmoid                      4.Add -- 5.Transpose -- Output
    //              \                           /
    //               +-- 2.Pool ---------------+
    //
    #[test]
    fn diamond() -> Result<()> {
        #[derive(Debug, PartialEq, Serialize)]
        struct Test {
            ty: AllocateType,
            is_first_use: bool,
            name: String,
        }

        let model = load_model("diamond.onnx")?;
        let schedule = Schedule::new(model.graph, Options::builder().build());
        let mem = MemoryPlanner::new(&schedule)
            .run()
            .iter()
            .zip_eq(schedule.kernels.iter().map(|(_, k)| k.name.clone()))
            .flat_map(|(allocs, name)| {
                allocs.iter().map(
                    move |AllocateInfo {
                              ty, is_first_use, ..
                          }| Test {
                        ty: *ty,
                        is_first_use: *is_first_use,
                        name: name.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        insta::assert_yaml_snapshot!(&mem);
        Ok(())
    }

    // Graph:
    //   inputs: cache [4,8] f32, src [4,4] f32, offset [] i64
    //   KVCacheUpdate(cache, src, offset) -> cache_new   ← intermediate
    //   Sqrt(cache_new)                   -> out         ← graph output
    //
    // KVCacheUpdate is mandatory-in-place on inputs[0] (cache), so cache_new
    // must share allocation with cache regardless of liveness. The snapshot
    // captures this: cache_new appears with `Input(cache)` allocation type.
    #[test]
    fn kv_cache_update_mandatory_in_place() -> Result<()> {
        #[derive(Debug, PartialEq, Serialize)]
        struct Test {
            ty: AllocateType,
            is_first_use: bool,
            name: String,
        }

        let model = load_model("kv_cache_update_in_place.onnx")?;
        let schedule = Schedule::new(model.graph, Options::builder().build());
        let mem = MemoryPlanner::new(&schedule)
            .run()
            .iter()
            .zip_eq(schedule.kernels.iter().map(|(_, k)| k.name.clone()))
            .flat_map(|(allocs, name)| {
                allocs.iter().map(
                    move |AllocateInfo {
                              ty, is_first_use, ..
                          }| Test {
                        ty: *ty,
                        is_first_use: *is_first_use,
                        name: name.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        insta::assert_yaml_snapshot!(&mem);
        Ok(())
    }
}
