pub mod ir;
pub mod kernel;
pub mod omp;
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

use crate::graph::operator::Operator;
use crate::graph::Graph;
use crate::graph::ValueId;
use crate::graph::ValueInfo;
use crate::options::*;
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
    manager.add_pass(Box::new(scheduler::MemoryAwareSchedulePass {
        num_streams: options.num_cuda_streams,
    }));
    if options.target == Target::CPU {
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
        let kernels = kernel::build_kernels(&mut graph, &mut graph_op, options.target);
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
