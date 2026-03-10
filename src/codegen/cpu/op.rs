use inkwell::values::*;
use smallvec::SmallVec;

use crate::onnx::operator::BatchNormalization;
use crate::onnx::operator::GeLU;
use crate::onnx::operator::LeakyReLU;
use crate::schedule::ElementwiseOpArg;
use crate::tensor::types::DataType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;

// TODO: Change `ty` to reference
#[derive(Debug, Clone)]
pub struct TensorPtr<'ctx> {
    pub ptr: PointerValue<'ctx>,
    pub ty: ResolvedTensorType,
    pub offset: IntValue<'ctx>,
    pub name: String,
}

impl<'ctx> TensorPtr<'ctx> {
    pub fn new(
        ptr: PointerValue<'ctx>,
        ty: ResolvedTensorType,
        offset: IntValue<'ctx>,
        name: String,
    ) -> Self {
        Self {
            ptr,
            ty,
            offset,
            name,
        }
    }

    pub fn new_with_index(
        ptr: PointerValue<'ctx>,
        ty: ResolvedTensorType,
        offset: IntValue<'ctx>,
        prefix: &str,
        index: usize,
    ) -> Self {
        Self::new(ptr, ty, offset, format!("{}.{}", prefix, index))
    }

    pub fn set_offset(mut self, offset: IntValue<'ctx>) -> Self {
        self.offset = offset;
        self
    }

    pub fn set_type(mut self, ty: ResolvedTensorType) -> Self {
        self.ty = ty;
        self
    }

    pub fn set_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    // TODO: Remove
    pub fn stride(&self, i: usize) -> usize {
        self.ty.stride(i)
    }
}

pub struct Operation<'ctx> {
    pub opcode: Opcode,
    pub operands: SmallVec<[TensorPtr<'ctx>; 4]>,
}

#[derive(Debug, Clone, Copy)]
pub enum SingleOpcode {
    Add,
    BatchNorm(BatchNormalization),
    Cast(DataType, DataType),
    Div,
    Exp,
    GeLU(GeLU),
    LeakyReLU(LeakyReLU),
    Log,
    Mul,
    Pow(DataType, DataType),
    Reciprocal,
    ReLU,
    Sigmoid,
    Sqrt,
    Sub,
    Tanh,
    Transfer,
}

#[derive(Debug, Clone)]
pub enum Opcode {
    Single(SingleOpcode),
    Fused(Vec<(SingleOpcode, Vec<ElementwiseOpArg>)>),
}

impl From<SingleOpcode> for Opcode {
    fn from(val: SingleOpcode) -> Self {
        Opcode::Single(val)
    }
}

#[derive(Clone)]
pub struct OMPContext<'ctx> {
    pub global_tid: PointerValue<'ctx>,
    pub is_last: PointerValue<'ctx>,
    pub lb: PointerValue<'ctx>,
    pub ub: PointerValue<'ctx>,
    pub stride: PointerValue<'ctx>,
}

pub struct OperationContext<'ctx> {
    pub operation: Operation<'ctx>,
    pub omp_ctx: Option<OMPContext<'ctx>>,
    pub omp_parallel: Option<usize>,
    pub omp_for: Option<usize>,
}

impl OperationContext<'_> {
    pub fn to_paralleize(&self, nest: usize) -> bool {
        self.omp_parallel == Some(nest)
    }

    pub fn to_for(&self, nest: usize) -> bool {
        self.omp_for == Some(nest)
    }
}

impl<'ctx> Operation<'ctx> {
    pub fn result_dims(&self) -> &ResolvedTensorDims {
        &self.dst_operand().ty.dims
    }

    pub fn result_type(&self) -> DataType {
        self.dst_operand().ty.elem_type
    }

    pub fn dst_operand(&self) -> &TensorPtr<'ctx> {
        &self.operands[0]
    }

    pub fn src_operands(&self) -> &[TensorPtr<'ctx>] {
        &self.operands[1..]
    }
}
