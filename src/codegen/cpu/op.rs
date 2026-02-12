use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use smallvec::SmallVec;

use crate::codegen::cpu::translator::FunctionTranslator;
use crate::onnx::operator::BatchNormalization;
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

    pub fn with_offset(mut self, offset: IntValue<'ctx>) -> Self {
        self.offset = offset;
        self
    }

    pub fn with_type(mut self, ty: ResolvedTensorType) -> Self {
        self.ty = ty;
        self
    }

    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    // TODO: Remove
    pub fn stride(&self, i: usize) -> usize {
        self.ty.stride(i)
    }

    pub fn to_outlined_nth_tensor(
        &self,
        translator: &FunctionTranslator<'_, 'ctx>,
        n: u32,
    ) -> Result<Self, BuilderError> {
        // global_tid, bound_tid, ...
        let begin = 2 + n * 2;
        let ptr_type = translator.context.ptr_type(AddressSpace::default());
        let i64_type = translator.context.i64_type();

        let ptr = translator
            .func
            .get_nth_param(begin)
            .unwrap()
            .into_pointer_value();
        let ptr = translator
            .builder
            .build_load(ptr_type, ptr, "")?
            .into_pointer_value();
        let offset = translator
            .func
            .get_nth_param(begin + 1)
            .unwrap()
            .into_pointer_value();
        let offset = translator
            .builder
            .build_load(i64_type, offset, "")?
            .into_int_value();

        Ok(Self::new(ptr, self.ty.clone(), offset, self.name.clone()))
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
    Exp,
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

impl Into<Opcode> for SingleOpcode {
    fn into(self) -> Opcode {
        Opcode::Single(self)
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
        self.omp_parallel.map_or(false, |n| n == nest)
    }

    pub fn to_for(&self, nest: usize) -> bool {
        self.omp_for.map_or(false, |n| n == nest)
    }
}

impl<'ctx> Operation<'ctx> {
    pub fn result_dims(&self) -> &ResolvedTensorDims {
        &self.dst_operand().ty.dims
    }

    pub fn to_outlined(
        &self,
        translator: &FunctionTranslator<'_, 'ctx>,
    ) -> Result<Self, BuilderError> {
        let operands = self
            .operands
            .iter()
            .enumerate()
            .map(|(i, tensor)| tensor.to_outlined_nth_tensor(translator, i as u32))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            operands,
            opcode: self.opcode.clone(),
        })
    }

    pub fn outlined_type(&self, context: &'ctx Context) -> FunctionType<'ctx> {
        let void_type = context.void_type();
        let ptr_type = context.ptr_type(AddressSpace::default());
        let argc = self.operands.len();
        let mut vec = Vec::with_capacity(2 + argc * 2);

        // global_tid
        vec.push(ptr_type.into());

        // bound_tid
        vec.push(ptr_type.into());

        for _ in 0..argc {
            vec.push(ptr_type.into());
            vec.push(ptr_type.into());
        }

        void_type.fn_type(&vec, false)
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

impl<'ctx> OperationContext<'ctx> {
    pub fn to_outlined(
        &self,
        translator: &FunctionTranslator<'_, 'ctx>,
        omp_ctx: OMPContext<'ctx>,
    ) -> Result<Self, BuilderError> {
        let operation = self.operation.to_outlined(translator)?;
        assert!(self.omp_ctx.is_none());
        Ok(Self {
            operation,
            omp_ctx: Some(omp_ctx),
            omp_parallel: None,
            omp_for: self.omp_for,
        })
    }
}
