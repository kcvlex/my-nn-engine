use crate::codegen::*;
use crate::onnx::model::ValueId;
use crate::onnx::operator;
use crate::onnx::operator::Operator;
use crate::schedule::*;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::{DataType, FloatType, SIntType, UIntType};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::targets::FileType;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use std::collections::HashMap;
use std::path::Path;
use indexmap::IndexMap;
use indexmap::map::Entry;

struct MemSize {
    map: IndexMap<DataType, u64>,
}

impl MemSize {
    fn append(&mut self, k: DataType, v: u64) {
        match self.map.entry(k) {
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = (*entry.get()).max(v);
            }
            Entry::Vacant(entry) => {
                entry.insert(v);
            }
        }
    }
}

pub struct CodeGenContext {
    pub schedule: Schedule,
    mem_sizes: Vec<MemSize>,
}


