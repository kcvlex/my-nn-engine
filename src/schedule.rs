pub mod ir;
pub mod kernel;
pub mod omp;
pub mod placement;
pub mod scheduler;

use std::any::Any;
use std::any::TypeId;
use std::collections::HashMap;
use std::ops::Index;
use std::ops::IndexMut;

use id_arena::Arena;
use id_arena::Id;
pub use ir::*;
use log::info;

use crate::graph::operator::args;
use crate::graph::operator::Operator;
use crate::graph::Graph;
use crate::graph::ValueId;
use crate::graph::ValueInfo;
use crate::options::*;
use crate::schedule::scheduler::PlacementStrategy;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::modify::SimpleGraphOp;

#[derive(Default)]
pub struct AnalysisResults {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl AnalysisResults {
    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
    }

    pub fn get<T: Send + Sync + 'static>(&self) -> &T {
        self.map
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("Analysis result not found: {}", std::any::type_name::<T>()))
            .downcast_ref()
            .unwrap()
    }
}

pub trait SchedulePass {
    fn summary(&self) -> &str;
    fn run(&self, schedule: &mut Schedule);
}

pub struct SchedulePassManager {
    name: String,
    passes: Vec<Box<dyn SchedulePass>>,
}

impl SchedulePassManager {
    pub fn new(name: String) -> Self {
        Self {
            name,
            passes: Vec::new(),
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn SchedulePass>) {
        self.passes.push(pass);
    }

    pub fn run(&self, schedule: &mut Schedule) {
        info!("SchedulePassManager: {}", self.name);
        for pass in self.passes.iter() {
            info!("-- Running pass: {}", pass.summary());
            pass.run(schedule);
        }
    }
}

pub fn create_schedule_passes(options: &Options) -> SchedulePassManager {
    let mut manager = SchedulePassManager::new("Schedule".to_string());
    let placement_strategy = match options.placement_strategy {
        Some(s) => s,
        None => PlacementStrategy::Uniform(match options.target {
            Target::CUDA => ir::Device::CUDA,
            Target::CPU => ir::Device::CPU,
        }),
    };
    manager.add_pass(Box::new(scheduler::MemoryAwareSchedulePass {
        num_streams: options.num_cuda_streams,
        placement_strategy,
    }));
    let needs_cpu_omp = options.target == Target::CPU ||
        matches!(
            placement_strategy,
            PlacementStrategy::StructuralKvTouch | PlacementStrategy::Uniform(ir::Device::CPU)
        );
    if needs_cpu_omp {
        manager.add_pass(Box::new(omp::OmpAnnotatePass {
            elementwise_threshold: options.omp_elementwise_threshold,
            softmax_threshold: options.omp_softmax_threshold,
        }));
    }
    manager
}

#[derive(Default)]
pub struct Kernels(Arena<Kernel>);
pub type KernelId = Id<Kernel>;

pub struct Schedule {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: Vec<ValueId>,
    pub session_states: Vec<ValueId>,

    pub kernels: Kernels,
    pub options: Options,
    pub analysis: AnalysisResults,
    pub execution_plan: Option<ExecutionPlan>,

    graph: Graph,
}

#[derive(Debug, Clone)]
pub struct Kernel {
    pub inputs: Vec<Option<ValueId>>,
    pub outputs: Vec<ValueId>,
    pub body: KernelBody,
    pub name: String,
}

impl Kernel {
    /// Maps each `inputs` slot to its index in the flattened argument list
    /// (outputs followed by `Some` inputs). Returns `None` for `None` inputs.
    pub fn input_ptr_map(&self) -> Vec<Option<usize>> {
        let mut map = vec![None; self.inputs.len()];
        let mut ptr_idx = self.outputs.len();
        for (i, inp) in self.inputs.iter().enumerate() {
            if inp.is_some() {
                map[i] = Some(ptr_idx);
                ptr_idx += 1;
            }
        }
        map
    }
}

#[derive(Debug, Clone)]
pub enum KernelBody {
    Opaque(Opaque),
    ElementWises(ElementWises),
}

#[derive(Debug, Clone)]
pub struct Opaque {
    pub op: Operator,
}

#[derive(Debug, Clone)]
pub enum ElementwiseOpArg {
    Input(usize),
    NthResult(usize),
}

#[derive(Debug, Clone)]
pub struct ElementWises {
    pub ops: Vec<(Operator, Vec<ElementwiseOpArg>)>,
}

pub type ChunkId = usize;

impl Kernels {
    pub fn iter(&self) -> impl Iterator<Item = (KernelId, &Kernel)> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (KernelId, &mut Kernel)> {
        self.0.iter_mut()
    }
}

impl Index<KernelId> for Kernels {
    type Output = Kernel;
    fn index(&self, index: KernelId) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<KernelId> for Kernels {
    fn index_mut(&mut self, index: KernelId) -> &mut Self::Output {
        &mut self.0[index]
    }
}

macro_rules! matches_opaque {
    ($kernel:expr, $pat:pat) => {{
        match &$kernel.body {
            KernelBody::Opaque(Opaque { op }) => matches!(op, $pat),
            KernelBody::ElementWises(_) => false,
        }
    }};
}

pub(crate) use matches_opaque;

impl Schedule {
    pub fn new(mut graph: Graph, options: Options) -> Self {
        let mut graph_op = SimpleGraphOp::new(&graph);
        let mut inputs = Vec::new();
        let mut session_states = Vec::new();
        for &node_id in graph.inputs.iter() {
            match graph.nodes[node_id].op {
                Operator::Input(v) => inputs.push(v),
                Operator::SessionState(v) => session_states.push(v),
                _ => unreachable!(),
            }
        }
        let outputs = graph
            .outputs
            .iter()
            .map(|x| match graph.nodes[*x].op {
                Operator::Output(v) => v,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let initializers = graph.initializer_ids();
        let include_cpu_workspaces =
            options.target == Target::CPU || options.placement_strategy.is_some();
        let kernels = kernel::build_kernels(&mut graph, &mut graph_op, include_cpu_workspaces);
        Self {
            inputs,
            outputs,
            initializers,
            session_states,

            kernels,
            options,
            analysis: AnalysisResults::default(),
            execution_plan: None,

            graph,
        }
    }

    pub fn get_resolved_tensor_type(&self, id: ValueId) -> Option<&ResolvedTensorType> {
        self.graph.values[id].ty.as_ref()?.as_resolved()
    }

    pub fn get_value(&self, id: ValueId) -> &ValueInfo {
        &self.graph.values[id]
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn value_byte_size(&self, v: ValueId) -> usize {
        let rty = self
            .get_resolved_tensor_type(v)
            .unwrap_or_else(|| panic!("unresolved tensor type for {v:?}"));
        rty.storage_num_elements() * (rty.elem_type.bit_width() / 8)
    }

    /// For a kernel that must write its output in place over an input (the KV
    /// cache), the index of that in-place input; `None` otherwise.
    pub(crate) fn must_in_place_input(&self, kid: KernelId) -> Option<usize> {
        let KernelBody::Opaque(Opaque { op }) = &self.kernels[kid].body else {
            return None;
        };
        match op {
            Operator::KVCacheUpdate => Some(args::KVCACHE_UPDATE_CACHE),
            Operator::QuantizingKVCacheUpdate => Some(args::QKVCACHE_UPDATE_CACHE),
            _ => None,
        }
    }

    /// Map each value forwarded to a graph output through a chain of
    /// Identity / Reinterpret kernels to that output, so codegen can alias the
    /// producer's buffer to the output buffer.
    pub(crate) fn output_aliases(&self) -> HashMap<ValueId, ValueId> {
        let mut passthrough_input: HashMap<ValueId, ValueId> = HashMap::new();
        for (_, kernel) in self.kernels.iter() {
            if !matches_opaque!(kernel, Operator::Identity | Operator::Reinterpret(_)) {
                continue;
            }
            let Some(out0) = kernel.outputs.first().copied() else {
                continue;
            };
            let Some(in0) = kernel.inputs.first().and_then(|x| *x) else {
                continue;
            };
            passthrough_input.insert(out0, in0);
        }

        let mut alias: HashMap<ValueId, ValueId> = HashMap::new();
        let mut queue: Vec<ValueId> = Vec::with_capacity(self.outputs.len());
        for &out in &self.outputs {
            alias.insert(out, out);
            queue.push(out);
        }
        while let Some(v) = queue.pop() {
            let target = alias[&v];
            if let Some(&pred) = passthrough_input.get(&v) {
                if alias.insert(pred, target).is_none() {
                    queue.push(pred);
                }
            }
        }
        alias
    }
}
