use crate::onnx::model::ValueId;
use crate::onnx::operator::args;
use crate::onnx::operator::Operator;
use crate::schedule::*;
use indexmap::{IndexMap, IndexSet};
use std::collections::{HashMap, HashSet};

// TODO: Make the order deterministic
#[derive(Debug)]
struct DependencyGraph {
    value2defined: IndexMap<ValueId, KernelId>,
    value2used: IndexMap<ValueId, IndexSet<(KernelId, usize)>>,
    inputs_set: HashSet<ValueId>,
    outputs_set: HashSet<ValueId>,
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
                    .collect::<HashSet<_>>();
                let outputs_set = schedule.outputs.iter().copied().collect::<HashSet<_>>();
                (inputs_set, outputs_set)
            }
            Target::CUDA => (HashSet::new(), HashSet::new()),
        };
        let ignore = |x| inputs_set.contains(x) || outputs_set.contains(x);
        for (kernel_id, kernel) in schedule.kernels.0.iter() {
            // if node.is_dummy() {
            //     continue;
            // }

            for (i, &input) in kernel.inputs.iter().enumerate().filter(|(_, x)| !ignore(x)) {
                value2used
                    .entry(input)
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

#[derive(Default)]
struct Chunks {
    free_list: Vec<ChunkId>,
    slot: usize,
}

impl Chunks {
    fn allocate(&mut self) -> ChunkId {
        let id = self.slot;
        self.slot += 1;
        id
    }

    fn free(&mut self, id: ChunkId) {
        self.free_list.push(id);
    }

    fn reuse_or_new(&mut self) -> ChunkId {
        self.free_list.pop().unwrap_or_else(|| self.allocate())
    }
}

pub(super) struct MemoryPlanner<'sched> {
    schedule: &'sched Schedule,
    deps: DependencyGraph,
    liveness_counter: IndexMap<ChunkId, usize>,
    chunks: Chunks,
    allocations: HashMap<ValueId, AllocateType>,
}

impl<'sched> MemoryPlanner<'sched> {
    pub(super) fn new(schedule: &'sched Schedule) -> Self {
        let deps = DependencyGraph::new(schedule);

        MemoryPlanner {
            schedule,
            deps,
            liveness_counter: IndexMap::new(),
            chunks: Chunks::default(),
            allocations: HashMap::new(),
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

        for (kernel_id, kernel) in self.schedule.kernels.iter() {
            let allocated = self.run_kernel(kernel_id);

            // Verify
            // if self.schedule.options.target == Target::CUDA {
            //     zip_eq(
            //         allocated.iter(),
            //         kernel.inputs.iter().chain(kernel.outputs.iter()),
            //     )
            //     .for_each(|(allocated, value_id)| assert!(allocated.0 == *value_id))
            // }

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
            .map(|(_, kernel)| match self.schedule.options.target {
                Target::CPU => kernel
                    .outputs
                    .iter()
                    .map(|id| to_allocate_info(self, id))
                    .collect(),
                Target::CUDA => kernel
                    .inputs
                    .iter()
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
                    if i == 0 && matches_single_kernel!(kernel, Operator::Identity) {
                        output_set.insert(kernel.inputs[0], *v);
                    }
                }
            }
        }
    }

    fn run_kernel(&mut self, kernel_id: KernelId) -> Vec<(ValueId, AllocateType)> {
        let mut res = Vec::new();
        let kernel = &self.schedule.kernels.0[kernel_id];

        if self.schedule.options.target != Target::CPU {
            for input in kernel.inputs.iter() {
                match self.allocations.get(input) {
                    Some(info) => {
                        info.chunk_id().to_owned().expect("Input must be allocated");
                    }
                    None => {
                        let chunk_id = self.chunks.reuse_or_new();
                        self.allocations
                            .insert(*input, AllocateType::Chunk(chunk_id));
                        *self.liveness_counter.entry(chunk_id).or_insert(0) += 1;
                        res.push((*input, AllocateType::Chunk(chunk_id)));
                    }
                };
            }
        }

        for output in kernel.outputs.iter() {
            let chunk = match kernel.body {
                // Split is a special case.
                // TODO: When the input is `Input` or initializer.
                KernelBody::SingleKernel(SingleKernel {
                    op: Operator::Split(_),
                }) if self.schedule.options.target == Target::CPU => {
                    let res = *self.allocations.get(&kernel.inputs[0]).unwrap();
                    // assert!(matches!(res, AllocateType::Chunk(_)));
                    res
                }
                _ => {
                    if self.deps.outputs_set.contains(output) {
                        AllocateType::Output(*output)
                    } else {
                        match self.try_in_place(*output) {
                            Some(prev) => *self.allocations.get(&prev).unwrap(),
                            None => AllocateType::Chunk(self.chunks.reuse_or_new()),
                        }
                    }
                }
            };
            res.push((*output, chunk));
            if let AllocateType::Chunk(chunk) = chunk {
                // It is possible that the value is not used by any other kernels, e.g., the output
                // of splitted one.
                if let Some(used) = self.deps.value2used.get(output) {
                    *self.liveness_counter.entry(chunk).or_insert(0) += used.len();
                }
            }
        }

        for input in kernel
            .inputs
            .iter()
            .filter(|x| !self.deps.inputs_set.contains(x))
        {
            let chunk_id = match self.allocations.get(input).and_then(|x| x.chunk_id()) {
                Some(v) => v,
                None => continue,
            };
            let counter = self.liveness_counter.get_mut(&chunk_id).unwrap();
            *counter -= 1;
            if *counter == 0 {
                self.chunks.free(chunk_id);
            }
        }
        res
    }

    // TODO: Identity assumes that the computation MUST be in-place.
    // If the source is not contiguous, we need to allocate a new chunk.
    fn try_in_place(&self, value_id: ValueId) -> Option<ValueId> {
        if !self.can_in_place(value_id) {
            return None;
        }

        let kernel_id = self.deps.value2defined[&value_id];
        let kernel = &self.schedule.kernels.0[kernel_id];
        for input in kernel.inputs.iter() {
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

            if let KernelBody::SingleKernel(SingleKernel { op }) = &kernel.body {
                match op {
                    // TODO: correct?
                    Operator::Identity => return Some(*input),
                    Operator::Gemm(_) => {
                        if !is_input &&
                            kernel
                                .inputs
                                .get(args::GEMM_C)
                                .map(|x| x == input)
                                .unwrap_or(false)
                        {
                            return Some(*input);
                        }
                    }
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
            KernelBody::SingleKernel(SingleKernel { op }) => match op {
                Operator::Identity => true,
                Operator::Gemm(_) => self.schedule.kernels.0[kernel_id]
                    .inputs
                    .get(args::GEMM_C)
                    .is_some(),
                _ => op.is_elementwise(),
            },
            KernelBody::FusedElementWises(_) => true,
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;
    use crate::onnx::load::*;
    use crate::onnx::model::Model;
    use crate::transform::modify::SimpleGraphOp;
    use crate::transform::shape::*;
    use crate::transform::*;
    use std::io::{Error, Result};
    use std::path::PathBuf;

    fn load_model(path: &str) -> Result<Model> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/plan")
            .join(path);
        let mut model =
            Model::load_from_path(path).map_err(|e| Error::other(format!("{:?}", e)))?;
        let mut pass_manager = SimplePassManager::new("Shape".to_string());
        pass_manager.add_pass(Box::new(infer::ShapeInference::default()));
        pass_manager.add_pass(Box::new(strides::AssignStrides::default()));
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
            .flatten()
            .zip_eq(schedule.kernels.iter().map(|(_, k)| k.name.clone()))
            .map(
                |(
                    AllocateInfo {
                        ty, is_first_use, ..
                    },
                    name,
                )| Test {
                    ty: *ty,
                    is_first_use: *is_first_use,
                    name,
                },
            )
            .collect::<Vec<_>>();
        insta::assert_yaml_snapshot!(&mem);
        Ok(())
    }
}
