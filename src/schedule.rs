pub mod kernel;
pub mod mem_alloc;
pub mod omp;

use std::ops::Index;
use std::ops::IndexMut;

use id_arena::Arena;
use id_arena::Id;
use log::info;
use serde::Serialize;
use serde_derive::Serialize;

use crate::onnx::model::Graph;
use crate::onnx::model::ValueId;
use crate::onnx::model::ValueInfo;
use crate::onnx::operator::Operator;
use crate::options::*;
use crate::tensor::types::ResolvedTensorType;
use crate::transform::modify::SimpleGraphOp;

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
    manager.add_pass(Box::new(mem_alloc::MemAllocPass));
    manager.add_pass(Box::new(omp::OmpAnnotatePass {
        threshold: options.omp_threshold,
    }));
    manager
}

#[derive(Default)]
pub struct Kernels(Arena<Kernel>);
pub type KernelId = Id<Kernel>;

pub struct Schedule {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: Vec<ValueId>,

    pub kernels: Kernels,
    pub options: Options,

    graph: Graph,
}

#[derive(Debug, Clone, Default)]
pub struct OmpInfo {
    pub omp_parallel: Option<usize>,
    pub omp_for: Option<usize>,
}

#[derive(Debug)]
pub struct Kernel {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub body: KernelBody,
    pub name: String,

    pub mem_alloc: Option<Vec<AllocateInfo>>,
    pub omp_info: OmpInfo,
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

#[derive(Debug, PartialEq, Eq, Clone, Copy, Serialize)]
pub struct AllocateInfo {
    #[serde(serialize_with = "serialize_value_id")]
    pub value_id: ValueId,
    pub ty: AllocateType,
    pub is_first_use: bool,
}

fn serialize_value_id<S>(value_id: &ValueId, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_u64(value_id.index() as u64)
}

pub type ChunkId = usize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocateType {
    Chunk(ChunkId),
    Input(ValueId),
    Output(ValueId),
}

impl Serialize for AllocateType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            AllocateType::Chunk(id) => {
                serializer.serialize_newtype_variant("AllocateType", 0, "Chunk", id)
            }
            AllocateType::Input(id) => {
                serializer.serialize_newtype_variant("AllocateType", 1, "Input", &id.index())
            }
            AllocateType::Output(id) => {
                serializer.serialize_newtype_variant("AllocateType", 2, "Output", &id.index())
            }
        }
    }
}

impl AllocateType {
    pub fn chunk_id(&self) -> Option<ChunkId> {
        match self {
            AllocateType::Chunk(id) => Some(*id),
            AllocateType::Input(_) | AllocateType::Output(_) => None,
        }
    }
}

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
    pub fn new(graph: Graph, options: Options) -> Self {
        let graph_op = SimpleGraphOp::new(&graph);
        let inputs = graph
            .inputs
            .iter()
            .map(|x| match graph.nodes[*x].op {
                Operator::Input(v) => v,
                _ => unreachable!(),
            })
            //.chain(graph.initializer.keys().copied())
            .collect::<Vec<_>>();
        let outputs = graph
            .outputs
            .iter()
            .map(|x| match graph.nodes[*x].op {
                Operator::Output(v) => v,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>();
        let initializers = graph.initializer.keys().copied().collect::<Vec<_>>();
        let kernels = kernel::build_kernels(&graph, &graph_op);
        Self {
            inputs,
            outputs,
            initializers,

            kernels,
            options,

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

    pub fn max_chunk_id(&self) -> Option<usize> {
        self.kernels
            .iter()
            .filter_map(|(_, kernel)| kernel.mem_alloc.as_ref())
            .flatten()
            .filter_map(|info| info.ty.chunk_id())
            .max()
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
