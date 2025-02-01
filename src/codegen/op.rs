use crate::codegen::blas::Precision;
use crate::codegen::translator::FunctionTranslator;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;

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

pub enum Operation<'ctx> {
    UnaryOp(UnaryOps<'ctx>, UnaryOpcode),
    BinaryOp(BinaryOps<'ctx>, BinaryOpcode),
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

#[derive(Debug, Clone, Copy)]
pub enum BinaryArithmeticOpcode {
    Add,
    Mul,
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryArithmetic {
    pub opcode: BinaryArithmeticOpcode,
    pub is_float: bool,
}

#[derive(Debug, Clone)]
pub enum BinaryOpcode {
    BinaryArithmetic(BinaryArithmetic),
    Gemm(Gemm),
}

#[derive(Debug, Clone, Copy)]
pub struct Gemm {
    pub prec: Precision,
    pub trans_a: bool,
    pub trans_b: bool,
    pub trans_c: bool,
    pub alpha: f64,
    pub beta: f64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOpcode {
    ReLU,
    Transfer,
}

pub struct UnaryOps<'ctx> {
    pub dst: TensorPtr<'ctx>,
    pub src: TensorPtr<'ctx>,
}

pub struct BinaryOps<'ctx> {
    pub dst: TensorPtr<'ctx>,
    pub lhs: TensorPtr<'ctx>,
    pub rhs: TensorPtr<'ctx>,
}

impl<'ctx> Operation<'ctx> {
    pub fn result_dims(&self) -> &ResolvedTensorDims {
        match self {
            Operation::UnaryOp(op, _) => &op.dst.ty.dims,
            Operation::BinaryOp(op, _) => &op.dst.ty.dims,
        }
    }

    pub fn to_outlined(
        &self,
        translator: &FunctionTranslator<'_, 'ctx>,
    ) -> Result<Self, BuilderError> {
        match self {
            Operation::UnaryOp(op, opcode) => {
                let dst = op.dst.to_outlined_nth_tensor(translator, 0)?;
                let src = op.src.to_outlined_nth_tensor(translator, 1)?;
                Ok(Operation::UnaryOp(UnaryOps { dst, src }, *opcode))
            }
            Operation::BinaryOp(op, opcode) => {
                let dst = op.dst.to_outlined_nth_tensor(translator, 0)?;
                let lhs = op.lhs.to_outlined_nth_tensor(translator, 1)?;
                let rhs = op.rhs.to_outlined_nth_tensor(translator, 2)?;
                Ok(Operation::BinaryOp(
                    BinaryOps { dst, lhs, rhs },
                    opcode.clone(),
                ))
            }
        }
    }

    pub fn outlined_type(&self, context: &'ctx Context) -> FunctionType<'ctx> {
        let void_type = context.void_type();
        let ptr_type = context.ptr_type(AddressSpace::default());
        let argc = match self {
            Operation::UnaryOp(_, _) => 2,
            Operation::BinaryOp(_, _) => 3,
        };
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
        match self {
            Operation::UnaryOp(UnaryOps { dst, src }, _) => vec![dst.clone(), src.clone()],
            Operation::BinaryOp(BinaryOps { dst, lhs, rhs }, _) => {
                vec![dst.clone(), lhs.clone(), rhs.clone()]
            }
        }
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
