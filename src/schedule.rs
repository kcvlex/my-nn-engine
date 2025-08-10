pub mod kernel;
pub mod mem_alloc;

use crate::onnx::model::{Graph, NodeId, ValueId};
use crate::onnx::operator::Operator;
use crate::onnx::utils;
use crate::schedule::kernel::*;
use crate::transform::modify::GraphOp;
use id_arena::{Arena, Id};
use indexmap::{IndexMap, IndexSet};
use serde::Serialize;
use serde_derive::Serialize;
use std::collections::{HashMap, HashSet};

#[derive(Default)]
pub struct Kernels(Arena<Kernel>);
pub type KernelId = Id<Kernel>;

pub struct Schedule<'graph> {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub initializers: Vec<ValueId>,

    pub kernels: Kernels,

    graph: &'graph Graph,
}

#[derive(Debug)]
pub struct Kernel {
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
    pub body: KernelBody,
    pub name: String,

    pub mem_alloc: Option<Vec<AllocateInfo>>,
}

#[derive(Debug, Clone)]
pub enum KernelBody {
    Single(Single),
    ElementWises(ElementWises),
}

#[derive(Debug, Clone)]
pub struct Single {
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

pub fn build_init_schedule<'graph>(
    graph: &'graph Graph,
    graph_op: &mut impl GraphOp,
) -> Schedule<'graph> {
    let inputs = graph
        .inputs
        .iter()
        .map(|x| match graph.nodes[*x].op {
            Operator::Input(v) => v,
            _ => unreachable!(),
        })
        .chain(graph.initializer.keys().copied())
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
    let kernels = kernel::build_kernels(graph, graph_op);
    Schedule {
        inputs,
        outputs,
        initializers,

        kernels,

        graph,
    }
}
