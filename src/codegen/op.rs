use crate::codegen::translator::FunctionTranslator;
use crate::onnx::operator::LeakyReLU;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::{DataType, ResolvedTensorType};
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use smallvec::SmallVec;

// TODO: Change `ty` to reference
#[derive(Debug, Clone)]
pub struct TensorPtr<'ctx> {
    pub ptr: PointerValue<'ctx>,
    pub ty: ResolvedTensorType,
    pub offset: IntValue<'ctx>,
    pub name: String,
}

impl<'ctx> TensorPtr<'ctx> {
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
        let ty = self.ty.clone();
        let name = self.name.clone();
        Ok(Self {
            ptr,
            ty,
            offset,
            name,
        })
    }
}

pub struct Operation<'ctx> {
    pub opcode: Opcode,
    pub operands: SmallVec<[TensorPtr<'ctx>; 4]>,
}

#[derive(Debug, Clone, Copy)]
pub enum Opcode {
    Add,
    Exp,
    LeakyReLU(LeakyReLU),
    Log,
    Mul,
    ReLU,
    Sigmoid,
    Tanh,
    Transfer,
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
        &self.operands[0].ty.dims
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
            opcode: self.opcode,
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

    pub fn operands_as_vec(&self) -> Vec<TensorPtr<'ctx>> {
        self.operands.to_vec()
    }

    pub fn dst_operand(&self) -> &TensorPtr<'ctx> {
        &self.operands[0]
    }

    pub fn unary_operand(&self) -> &TensorPtr<'ctx> {
        assert!(self.operands.len() == 2);
        &self.operands[1]
    }

    pub fn binary_operands(&self) -> (&TensorPtr<'ctx>, &TensorPtr<'ctx>) {
        assert!(self.operands.len() == 3);
        (&self.operands[1], &self.operands[2])
    }

    pub fn src_operands(&self) -> &[TensorPtr<'ctx>] {
        &self.operands[1..]
    }

    pub fn result_type(&self) -> DataType {
        self.dst_operand().ty.elem_type
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
