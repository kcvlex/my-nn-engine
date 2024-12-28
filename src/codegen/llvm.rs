use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use crate::tensor::tensor::{DataType, ResolvedTensorType};

use inkwell::attributes::*;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::OptimizationLevel;
use inkwell::{
    passes::PassBuilderOptions,
    targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine},
};
use inkwell::AddressSpace;

enum LLVMPass {
    LoopVectorize,
    SLPVectorize,
    InstCombine,
    Reassociate,
    GVN,
    SimplifyCFG,
    Mem2Reg,
}

impl LLVMPass {
    fn to_llvm_pass(&self) -> &'static str {
        match self {
            LLVMPass::LoopVectorize => "loop-vectorize",
            LLVMPass::SLPVectorize => "slp-vectorize",
            LLVMPass::InstCombine => "instcombine",
            LLVMPass::Reassociate => "reassociate",
            LLVMPass::GVN => "gvn",
            LLVMPass::SimplifyCFG => "simplifycfg",
            LLVMPass::Mem2Reg => "mem2reg",
        }
    }
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    target_machine: TargetMachine,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Self {
        let module = context.create_module("main");
        let builder = context.create_builder();
        Target::initialize_native(&InitializationConfig::default()).unwrap();
        let target_triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple).unwrap();
        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                OptimizationLevel::Aggressive,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .unwrap();

        CodeGen {
            context,
            module,
            builder,
            target_machine,
        }
    }
}

struct TensorPtr<'ctx> {
    context: &'ctx Context,
    ptr: PointerValue<'ctx>,
    ty: ResolvedTensorType,
    offset: IntValue<'ctx>,
}

struct LoopBB<'ctx> {
    preheader: BasicBlock<'ctx>,
    header: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,
}

struct Operators<'ctx> {
    lhs: TensorPtr<'ctx>,
    rhs: TensorPtr<'ctx>,
}

struct FunctionTranslator<'ctx> {
    context: &'ctx Context,
    builder: &'ctx Builder<'ctx>,
    function: FunctionValue<'ctx>,
}

impl<'ctx> FunctionTranslator<'ctx> {
    fn gen_nested_loop_rec(
        &self,
        res: TensorPtr<'ctx>,
        ops: Operators<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
    ) -> Result<(), BuilderError> {
        macro_rules! update_offset {
            ($ptr: expr, $offset_phi: expr, $name: expr, $exiting: expr) => {{
                let offset_int = $offset_phi.as_basic_value().into_int_value();
                self.builder.position_at_end($exiting);
                let stride = self
                    .context
                    .i64_type()
                    .const_int($ptr.ty.stride(nest).try_into().unwrap(), false);
                let offset_next = self.builder.build_int_add(
                    offset_int,
                    stride,
                    format!("offset.{}.{}.next", $name, nest).as_str(),
                )?;
                self.builder.position_at_end(loop_bb.header);
                $offset_phi.add_incoming(&[
                    (
                        &self.context.i64_type().const_int(0, false),
                        loop_bb.preheader,
                    ),
                    (&offset_next, $exiting),
                ]);
                let offset_sum = self.builder.build_int_add(
                    $ptr.offset,
                    offset_int,
                    format!("offset.sum.{}.{}", $name, nest).as_str(),
                )?;
                TensorPtr {
                    context: self.context,
                    ptr: $ptr.ptr,
                    ty: $ptr.ty,
                    offset: offset_sum,
                }
            }};
        }

        macro_rules! gep {
            ($ptr: expr, $ty: expr, $name: expr) => {{
                unsafe {
                    self.builder.build_in_bounds_gep(
                        $ty,
                        $ptr.ptr,
                        &[$ptr.offset],
                        format!("gep.{}", $name).as_str(),
                    )?
                }
            }};
        }
        macro_rules! load {
            ($ptr: expr, $ty: expr, $name: expr) => {{
                let gep = gep!($ptr, $ty, $name);
                self.builder
                    .build_load($ty, gep, format!("load.{}", $name).as_str())?
            }};
            ($ptr: expr, $name: expr) => {{
                match $ptr.ty.elem_type {
                    DataType::F32 => load!($ptr, self.context.f32_type(), $name),
                    DataType::F64 => load!($ptr, self.context.f64_type(), $name),
                    DataType::I64 => load!($ptr, self.context.i64_type(), $name),
                }
            }};
        }
        macro_rules! store {
            ($val: expr, $ptr: expr, $ty: expr, $name: expr) => {{
                let gep = gep!($ptr, $ty, $name);
                self.builder.build_store(gep, $val)?
            }};
            ($val: expr, $ptr: expr, $name: expr) => {{
                match $ptr.ty.elem_type {
                    DataType::F32 => store!($val, $ptr, self.context.f32_type(), $name),
                    DataType::F64 => store!($val, $ptr, self.context.f64_type(), $name),
                    DataType::I64 => store!($val, $ptr, self.context.i64_type(), $name),
                }
            }};
        }

        let is_last = nest + 1 == res.ty.dims.ndim();
        let ind = self
            .builder
            .build_phi(self.context.i64_type(), format!("ind.{}", nest).as_str())?;
        let bound: u64 = res.ty.dims[nest].try_into().unwrap();
        let exiting_bb = if is_last {
            loop_bb.header
        } else {
            self.context
                .append_basic_block(self.function, format!("exit.{}", nest).as_str())
        };
        let bound = self.context.i64_type().const_int(bound, false);
        let offset_phi_res = self.builder.build_phi(
            self.context.i64_type(),
            format!("offset.res.{}", nest).as_str(),
        )?;
        let offset_phi_lhs = self.builder.build_phi(
            self.context.i64_type(),
            format!("offset.lhs.{}", nest).as_str(),
        )?;
        let offset_phi_rhs = self.builder.build_phi(
            self.context.i64_type(),
            format!("offset.rhs.{}", nest).as_str(),
        )?;
        let next_ops = Operators {
            lhs: update_offset!(ops.lhs, offset_phi_lhs, "lhs", exiting_bb),
            rhs: update_offset!(ops.rhs, offset_phi_rhs, "rhs", exiting_bb),
        };
        let next_res = update_offset!(res, offset_phi_res, "res", exiting_bb);
        if is_last {
            // TODO: remove `into_float_value`
            let lhs = load!(next_ops.lhs, "lhs").into_float_value();
            let rhs = load!(next_ops.rhs, "rhs").into_float_value();
            let res = self.builder.build_float_add(lhs, rhs, "res")?;
            store!(res, next_res, "res");
            let ind_next = self.builder.build_int_add(
                ind.as_basic_value().into_int_value(),
                self.context.i64_type().const_int(1, false),
                format!("ind.{}.next", nest).as_str(),
            )?;
            let cond = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ind_next,
                bound,
                format!("cond.{}", nest).as_str(),
            )?;
            ind.add_incoming(&[
                (
                    &self.context.i64_type().const_int(0, false),
                    loop_bb.preheader,
                ),
                (&ind_next, exiting_bb),
            ]);
            self.builder
                .build_conditional_branch(cond, loop_bb.header, loop_bb.exit)?;
            Ok(())
        } else {
            let next_bb = self
                .context
                .append_basic_block(self.function, format!("loop.{}", nest).as_str());
            self.builder.build_unconditional_branch(next_bb)?;
            self.builder.position_at_end(exiting_bb);
            let ind_next = self.builder.build_int_add(
                ind.as_basic_value().into_int_value(),
                self.context.i64_type().const_int(1, false),
                format!("ind.{}.next", nest).as_str(),
            )?;
            let cond = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ind_next,
                bound,
                format!("cond.{}", nest).as_str(),
            )?;
            self.builder
                .build_conditional_branch(cond, loop_bb.header, loop_bb.exit)?;

            ind.add_incoming(&[
                (
                    &self.context.i64_type().const_int(0, false),
                    loop_bb.preheader,
                ),
                (&ind_next, exiting_bb),
            ]);
            self.builder.position_at_end(next_bb);
            let next_loop_bb = LoopBB {
                preheader: loop_bb.header,
                header: next_bb,
                exit: exiting_bb,
            };
            self.gen_nested_loop_rec(next_res, next_ops, next_loop_bb, nest + 1)
        }
    }

    fn gen_nested_loop(
        &self,
        res: TensorPtr<'ctx>,
        ops: Operators<'ctx>,
        preheader: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let header = self.context.append_basic_block(self.function, "header");
        let exit = self.context.append_basic_block(self.function, "exit");
        self.builder.build_unconditional_branch(header)?;
        self.builder.position_at_end(header);
        let loop_bb = LoopBB {
            preheader,
            header,
            exit,
        };
        self.gen_nested_loop_rec(res, ops, loop_bb, 0)?;
        self.builder.position_at_end(exit);
        Ok(exit)
    }
}

#[test] 
fn test_add() -> std::io::Result<()> {
    use crate::tensor::tensor::Tensor;
    use std::io::Error;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};

    let context = Context::create();
    let codegen = CodeGen::new(&context);
    let ptr_type = codegen.context.ptr_type(AddressSpace::default());
    let fn_type = codegen.context.void_type().fn_type(&[ptr_type.into(), ptr_type.into(), ptr_type.into()], false);
    let function = codegen.module.add_function("add", fn_type, None);
    let translator = FunctionTranslator {
        context: &context,
        builder: &codegen.builder,
        function,
    };
    
    macro_rules! make_tensor {
        ($ty: ty, $($expr: expr,)*) => {{
            let orig: ndarray::Array<$ty, _> = ndarray::array!($($expr,)*);
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .try_into()
                .map(|t| (t, orig.clone()))
                .map_err(|e| Error::other(format!("{:?}", e)));
            res
        }};
    }

    let entry = context.append_basic_block(function, "entry");
    translator.builder.position_at_end(entry);

    let (input0, orig0) = make_tensor!(f32, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], )?;
    let (input1, orig1) = make_tensor!(f32, [[1.0, 2.0, 3.0], [-4.0, -5.0, -7.0]],[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], )?;
    let mut buf = vec![0f32; input0.ty.dims.size()];
    let res = TensorPtr {
        context: &context,
        ptr: function.get_nth_param(0).unwrap().into_pointer_value(),
        ty: input0.ty.clone(),
        offset: context.i64_type().const_int(0, false),
    };
    let lhs = TensorPtr {
        context: &context,
        ptr: function.get_nth_param(1).unwrap().into_pointer_value(),
        ty: input0.ty.clone(),
        offset: context.i64_type().const_int(0, false),
    };
    let rhs = TensorPtr {
        context: &context,
        ptr: function.get_nth_param(2).unwrap().into_pointer_value(),
        ty: input0.ty.clone(),
        offset: context.i64_type().const_int(0, false),
    };
    translator.gen_nested_loop(res, Operators { lhs, rhs }, entry).map_err(|e| Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    codegen.builder.build_return(None).map_err(|e| Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

    function.print_to_stderr();

    let execution_engine = codegen.module
        .create_jit_execution_engine(OptimizationLevel::Default)
        .map_err(|e| Error::other(format!("{:?}", e)))?;
    type CodeType = unsafe extern "C" fn(*mut f32, *const u8, *const u8);
    let func: JitFunction<CodeType> = unsafe { execution_engine.get_function("add").ok() }
        .ok_or(Error::other("Unable to JIT compile `sum` function"))?;
    unsafe {
        func.call(
            buf.as_mut_ptr(),
            input0.data.raw_vec().as_ptr(),
            input1.data.raw_vec().as_ptr(),
        )
    };

    let res = ndarray::Array::from_vec(buf)
        .to_shape(orig0.shape())
        .map_err(|e| Error::other(format!("{:?}", e)))?
        .into_owned();
    let expected = (orig0 + orig1).into_dyn();
    assert_eq!(res, expected);

    Ok(())
}
