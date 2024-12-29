use crate::load::ModelLoadError;
use crate::model::{Graph, Model, Node, ValueId};
use crate::operator::Operator;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use crate::tensor::tensor::{DataType, ResolvedTensorType, Tensor, TensorData, TypeError};

use crate::codegen::memory;
use inkwell::attributes::*;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, FunctionLookupError, JitFunction};
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum CodeGenError {
    BuilderError(BuilderError),
    LLVMError(inkwell::support::LLVMString),
    TargetMachineError(String),
    IntrinsicNotFound(String),
}

pub enum LLVMPass {
    LoopUnroll,
    LoopVectorize,
    SLPVectorize,
    InstCombine,
    Reassociate,
    GlobalValueNumbering,
    SimplifyCFG,
    Mem2Reg,
}

impl LLVMPass {
    pub fn to_llvm_pass(&self) -> &'static str {
        match self {
            LLVMPass::LoopUnroll => "loop-unroll",
            LLVMPass::LoopVectorize => "loop-vectorize",
            LLVMPass::SLPVectorize => "slp-vectorizer",
            LLVMPass::InstCombine => "instcombine",
            LLVMPass::Reassociate => "reassociate",
            LLVMPass::GlobalValueNumbering => "gvn",
            LLVMPass::SimplifyCFG => "simplifycfg",
            LLVMPass::Mem2Reg => "mem2reg",
        }
    }

    pub fn passes(passes: &[LLVMPass]) -> String {
        passes
            .iter()
            .map(|p| p.to_llvm_pass())
            .collect::<Vec<_>>()
            .join(",")
    }
}

struct Intrinsics<'ctx> {
    fmax_f32: FunctionValue<'ctx>,
    fmax_f64: FunctionValue<'ctx>,
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,

    allocator: memory::Allocator<GlobalValue<'ctx>>,
    id2value: HashMap<ValueId, PointerValue<'ctx>>,
    main: FunctionValue<'ctx>,
    main_entry: BasicBlock<'ctx>,

    target_machine: TargetMachine,

    noalias: Attribute,
    noundef: Attribute,

    intrinsics: Intrinsics<'ctx>,
}

struct FunctionTranslator<'a, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    builder: &'a Builder<'ctx>,
    function: &'a FunctionValue<'ctx>,
    intrinsics: &'a Intrinsics<'ctx>,
}

fn target_machine() -> Result<TargetMachine, CodeGenError> {
    Target::initialize_native(&InitializationConfig::default())
        .map_err(CodeGenError::TargetMachineError)?;
    let target_triple = TargetMachine::get_default_triple();
    let cpu = TargetMachine::get_host_cpu_name().to_string();
    let features = TargetMachine::get_host_cpu_features().to_string();
    Target::from_triple(&target_triple)
        .map_err(CodeGenError::LLVMError)?
        .create_target_machine(
            &target_triple,
            &cpu,
            &features,
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| {
            CodeGenError::TargetMachineError("Unable to create target machine".to_string())
        })
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context) -> Result<Self, CodeGenError> {
        let module = context.create_module("main");
        let builder = context.create_builder();

        let ptr_type = context.ptr_type(AddressSpace::default());
        let fn_type = context
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let main = module.add_function("main", fn_type, None);
        let main_entry = context.append_basic_block(main, "entry");
        builder.position_at_end(main_entry);

        let get_attr = |name: &str| {
            let kind_id = Attribute::get_named_enum_kind_id(name);
            context.create_enum_attribute(kind_id, 0)
        };

        let noalias = get_attr("noalias");
        let noundef = get_attr("noundef");

        main.add_attribute(AttributeLoc::Param(0), noalias);
        main.add_attribute(AttributeLoc::Param(0), noundef);
        main.add_attribute(AttributeLoc::Param(1), noalias);
        main.add_attribute(AttributeLoc::Param(1), noundef);

        let target_machine = target_machine()?;

        macro_rules! get_intrinsic {
            ($name: expr, $args: expr) => {{
                Intrinsic::find($name)
                    .and_then(|intrinsic| intrinsic.get_declaration(&module, $args))
                    .ok_or_else(|| CodeGenError::IntrinsicNotFound($name.to_string()))
            }};
        }

        let f32_ty = context.f32_type().into();
        let f64_ty = context.f64_type().into();

        let fmax_f32 = get_intrinsic!("llvm.maximum.f32", &[f32_ty, f32_ty])?;
        let fmax_f64 = get_intrinsic!("llvm.maximum.f64", &[f64_ty, f64_ty])?;

        let intrinsics = Intrinsics { fmax_f32, fmax_f64 };

        Ok(CodeGen {
            context,
            module,
            builder,

            allocator: memory::Allocator::new(),
            id2value: HashMap::new(),
            main,
            main_entry,

            target_machine,

            noalias,
            noundef,

            intrinsics,
        })
    }

    pub fn compile(&mut self, graph: &Graph, passes: &[LLVMPass]) -> Result<(), CodeGenError> {
        self.compile_graph(graph)
            .map_err(CodeGenError::BuilderError)?;
        if !passes.is_empty() {
            self.module
                .run_passes(
                    LLVMPass::passes(passes).as_str(),
                    &self.target_machine,
                    inkwell::passes::PassBuilderOptions::create(),
                )
                .map_err(CodeGenError::LLVMError)?;
        }
        Ok(())
    }

    // TODO: Adjust attributes
    fn create_function(&self, name: &str, argc: u32) -> FunctionValue<'ctx> {
        let mut vec = Vec::with_capacity(argc as usize);
        for _ in 0..argc {
            vec.push(self.context.ptr_type(AddressSpace::default()).into());
        }
        let fn_type = self.context.void_type().fn_type(&vec, false);
        let func = self.module.add_function(name, fn_type, None);
        for i in 0..argc {
            func.add_attribute(AttributeLoc::Param(i), self.noalias);
            func.add_attribute(AttributeLoc::Param(i), self.noundef);
        }
        func
    }

    fn init_data(&mut self, graph: &Graph) -> Result<(), BuilderError> {
        macro_rules! define_gv {
            ($name: expr, $data: expr, $ty: expr, $convert: expr) => {{
                let len = $data.len();
                let gv = self
                    .module
                    .add_global($ty.array_type(len as u32), None, $name.as_str());
                let arr = $data.iter().map($convert).collect::<Vec<_>>();
                let arr = $ty.const_array(&arr);
                gv.set_initializer(&arr);
                gv
            }};
        }
        for (id, value) in graph.initializer.iter() {
            let name = format!("gv.{}", graph.values[*id].name);
            let gv = match value.data {
                TensorData::F32(ref data) => {
                    let ty = self.context.f32_type();
                    define_gv!(name, data, ty, |&x| ty.const_float(x.into()))
                }
                TensorData::F64(ref data) => {
                    let ty = self.context.f64_type();
                    define_gv!(name, data, ty, |&x| ty.const_float(x))
                }
                TensorData::I64(ref data) => {
                    let ty = self.context.i64_type();
                    define_gv!(name, data, ty, |&x| ty.const_int(x as u64, false))
                }
            };
            self.id2value.insert(*id, gv.as_pointer_value());
        }
        Ok(())
    }

    fn init_main_args(&mut self, graph: &Graph) -> Result<(), BuilderError> {
        for (i, arr) in [&graph.outputs, &graph.inputs].iter().enumerate() {
            let ptr = self
                .main
                .get_nth_param(i as u32)
                .unwrap()
                .into_pointer_value();
            for (i, node_id) in arr.iter().enumerate() {
                let value_id = match graph.nodes[*node_id].op {
                    Operator::Input(v) | Operator::Output(v) => v,
                    _ => unreachable!(),
                };
                let value = &graph.values[value_id];
                let ptr = unsafe {
                    self.builder.build_in_bounds_gep(
                        self.context.ptr_type(AddressSpace::default()),
                        ptr,
                        &[self.context.i64_type().const_int(i as u64, false)],
                        value.name.as_str(),
                    )
                }?;
                let ptr = self
                    .builder
                    .build_load(
                        self.context.ptr_type(AddressSpace::default()),
                        ptr,
                        value.name.as_str(),
                    )?
                    .into_pointer_value();
                self.id2value.insert(value_id, ptr);
            }
        }
        Ok(())
    }

    pub fn compile_graph(&mut self, graph: &Graph) -> Result<(), BuilderError> {
        self.init_data(graph)?;
        self.init_main_args(graph)?;
        for (_, node) in graph.nodes.iter() {
            if matches!(node.op, Operator::Input(_) | Operator::Output(_)) {
                continue;
            }
            let function = self.compile_node(node, graph)?;
            self.builder.position_at_end(self.main_entry);
            macro_rules! malloc {
                ($ty: expr, $len: expr, $name: expr) => {{
                    self.builder.build_array_malloc($ty, $len, $name)
                }};
            }

            // TODO: malloc
            for &id in node.outputs.iter() {
                let ty = graph.get_resolved_tensor_type(id).unwrap();
                let len = self
                    .context
                    .i64_type()
                    .const_int(ty.dims.size() as u64, false);
                let name = graph.values[id].name.as_str();
                if let std::collections::hash_map::Entry::Vacant(e) = self.id2value.entry(id) {
                    let ptr = match ty.elem_type {
                        DataType::F32 => malloc!(self.context.f32_type(), len, name),
                        DataType::F64 => malloc!(self.context.f64_type(), len, name),
                        DataType::I64 => malloc!(self.context.i64_type(), len, name),
                    }?;
                    e.insert(ptr);
                }
            }

            let mut args = node.outputs.clone();
            args.extend(node.inputs.clone());
            let args = args
                .iter()
                .map(|&id| self.id2value[&id])
                .map(|ptr| ptr.into())
                .collect::<Vec<_>>();
            let call = self.builder.build_call(function, &args[..], "")?;
            call.set_tail_call(true);
        }

        self.builder.position_at_end(self.main_entry);
        self.builder.build_return(None)?;
        Ok(())
    }

    fn compile_node(
        &self,
        node: &Node,
        graph: &Graph,
    ) -> Result<FunctionValue<'ctx>, BuilderError> {
        let mut args = node.outputs.clone();
        args.extend(node.inputs.clone());

        let function = self.create_function(node.name.as_str(), args.len() as u32);
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let translator = FunctionTranslator {
            context: self.context,
            module: &self.module,
            builder: &self.builder,
            function: &function,
            intrinsics: &self.intrinsics,
        };

        let mut ptrs = args
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let ptr = function
                    .get_nth_param(i as u32)
                    .unwrap()
                    .into_pointer_value();
                let ty = graph.get_resolved_tensor_type(id).unwrap().clone();
                let name = format!("ptr.{}", i);
                let offset = self.context.i64_type().const_int(0, false);
                TensorPtr {
                    ptr,
                    ty,
                    offset,
                    name,
                    perms: None,
                }
            })
            .collect::<Vec<_>>();

        macro_rules! gen_binaryop {
            ($op: expr) => {{
                let binop = BinaryOps {
                    dst: ptrs[0].clone(),
                    lhs: ptrs[1].clone(),
                    rhs: ptrs[2].clone(),
                };
                let op = Operation::BinaryOp(binop, $op);
                translator.gen_nested_loop(op, entry)
            }};
        }

        macro_rules! gen_unaryop {
            ($op: expr) => {{
                let op = Operation::UnaryOp(
                    UnaryOps {
                        dst: ptrs[0].clone(),
                        src: ptrs[1].clone(),
                    },
                    $op,
                );
                translator.gen_nested_loop(op, entry)
            }};
        }

        let exit = match node.op {
            Operator::Add => gen_binaryop!(BinaryOpcode::FloatAdd),
            Operator::ReLU => gen_unaryop!(UnaryOpcode::ReLU),
            Operator::Transpose(ref perm) => {
                ptrs[1].perms = Some(perm.clone());
                gen_unaryop!(UnaryOpcode::Transpose)
            }
            _ => todo!(),
        }?;

        self.builder.position_at_end(exit);
        self.builder.build_return(None)?;
        Ok(function)
    }
}

#[derive(Debug, Clone)]
struct TensorPtr<'ctx> {
    ptr: PointerValue<'ctx>,
    ty: ResolvedTensorType,
    offset: IntValue<'ctx>,
    name: String,
    perms: Option<Vec<usize>>,
}

impl TensorPtr<'_> {
    fn stride(&self, i: usize) -> usize {
        let i = match self.perms {
            Some(ref perms) => perms[i],
            None => i,
        };
        self.ty.stride(i)
    }
}

struct LoopBB<'ctx> {
    preheader: BasicBlock<'ctx>,
    header: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,
}

enum LLVMScalarType<'ctx> {
    LLVMInt(IntType<'ctx>),
    LLVMFloat(FloatType<'ctx>),
}

impl<'ctx> LLVMScalarType<'ctx> {
    fn from_data_type(context: &'ctx Context, data_type: DataType) -> Self {
        match data_type {
            DataType::F32 => LLVMScalarType::LLVMFloat(context.f32_type()),
            DataType::F64 => LLVMScalarType::LLVMFloat(context.f64_type()),
            DataType::I64 => LLVMScalarType::LLVMInt(context.i64_type()),
        }
    }
}

enum Operation<'ctx> {
    UnaryOp(UnaryOps<'ctx>, UnaryOpcode),
    BinaryOp(BinaryOps<'ctx>, BinaryOpcode),
}

#[derive(Debug, Clone, Copy)]
enum BinaryOpcode {
    IntAdd,
    FloatAdd,
}

#[derive(Debug, Clone, Copy)]
enum UnaryOpcode {
    ReLU,
    Transpose,
}

struct UnaryOps<'ctx> {
    dst: TensorPtr<'ctx>,
    src: TensorPtr<'ctx>,
}

struct BinaryOps<'ctx> {
    dst: TensorPtr<'ctx>,
    lhs: TensorPtr<'ctx>,
    rhs: TensorPtr<'ctx>,
}

impl Operation<'_> {
    fn result_dims(&self) -> &ResolvedTensorDims {
        match self {
            Operation::UnaryOp(op, _) => &op.dst.ty.dims,
            Operation::BinaryOp(op, _) => &op.dst.ty.dims,
        }
    }
}

impl<'ctx> FunctionTranslator<'_, 'ctx> {
    fn build_gep(&self, ptr: &TensorPtr<'ctx>) -> Result<PointerValue<'ctx>, BuilderError> {
        macro_rules! gep {
            ($ty: expr) => {{
                unsafe {
                    self.builder.build_in_bounds_gep(
                        $ty,
                        ptr.ptr,
                        &[ptr.offset],
                        format!("gep.{}", ptr.name).as_str(),
                    )
                }
            }};
        }
        let ty = LLVMScalarType::from_data_type(self.context, ptr.ty.elem_type);
        match ty {
            LLVMScalarType::LLVMFloat(ty) => gep!(ty),
            LLVMScalarType::LLVMInt(ty) => gep!(ty),
        }
    }

    fn build_load(&self, ptr: &TensorPtr<'ctx>) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        macro_rules! load {
            ($ty: expr) => {{
                let res: Result<_, BuilderError> = {
                    let gep = self.build_gep(ptr)?;
                    self.builder
                        .build_load($ty, gep, format!("load.{}", ptr.name).as_str())
                };
                res
            }};
        }
        let ty = LLVMScalarType::from_data_type(self.context, ptr.ty.elem_type);
        match ty {
            LLVMScalarType::LLVMFloat(ty) => load!(ty),
            LLVMScalarType::LLVMInt(ty) => load!(ty),
        }
    }

    fn build_store<V: BasicValue<'ctx>>(
        &self,
        ptr: &TensorPtr<'ctx>,
        val: V,
    ) -> Result<(), BuilderError> {
        let gep = self.build_gep(ptr)?;
        self.builder.build_store(gep, val).map(|_| ())
    }

    fn build_tail_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let call = self.builder.build_call(function, args, name)?;
        call.set_tail_call(true);
        Ok(call)
    }

    fn build_operation(&self, op: &Operation<'ctx>) -> Result<(), BuilderError> {
        match op {
            Operation::UnaryOp(op, opcode) => {
                let res = match opcode {
                    UnaryOpcode::ReLU => {
                        let (fmax, zero) = match op.dst.ty.elem_type {
                            DataType::F32 => (
                                self.intrinsics.fmax_f32,
                                self.context.f32_type().const_float(0.0),
                            ),
                            DataType::F64 => (
                                self.intrinsics.fmax_f64,
                                self.context.f64_type().const_float(0.0),
                            ),
                            _ => todo!(),
                        };
                        let src = self.build_load(&op.src)?.into_float_value();
                        self.build_tail_call(fmax, &[src.into(), zero.into()], "res")?
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                    }
                    UnaryOpcode::Transpose => self.build_load(&op.src)?,
                };
                self.build_store(&op.dst, res)
            }
            Operation::BinaryOp(op, opcode) => {
                let res = match opcode {
                    BinaryOpcode::FloatAdd => {
                        let lhs = self.build_load(&op.lhs)?.into_float_value();
                        let rhs = self.build_load(&op.rhs)?.into_float_value();
                        self.builder.build_float_add(lhs, rhs, "res")?
                    }
                    BinaryOpcode::IntAdd => todo!(),
                };
                self.build_store(&op.dst, res)
            }
        }
    }

    fn gen_nested_loop_rec(
        &self,
        ops: Operation<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
    ) -> Result<(), BuilderError> {
        macro_rules! update_offset {
            ($ptr: expr, $offset_phi: expr, $exiting: expr) => {{
                let offset_int = $offset_phi.as_basic_value().into_int_value();
                self.builder.position_at_end($exiting);
                let stride = self
                    .context
                    .i64_type()
                    .const_int($ptr.stride(nest).try_into().unwrap(), false);
                let offset_next = self.builder.build_int_add(
                    offset_int,
                    stride,
                    format!("offset.{}.{}.next", $ptr.name, nest).as_str(),
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
                    format!("offset.sum.{}.{}", $ptr.name, nest).as_str(),
                )?;
                TensorPtr {
                    ptr: $ptr.ptr,
                    ty: $ptr.ty,
                    offset: offset_sum,
                    name: $ptr.name,
                    perms: $ptr.perms.clone(),
                }
            }};
        }

        let is_last = nest + 1 == ops.result_dims().ndim();
        let ind = self
            .builder
            .build_phi(self.context.i64_type(), format!("ind.{}", nest).as_str())?;
        let bound: u64 = ops.result_dims()[nest].try_into().unwrap();
        let exiting_bb = if is_last {
            loop_bb.header
        } else {
            self.context
                .append_basic_block(*self.function, format!("exit.{}", nest).as_str())
        };
        let bound = self.context.i64_type().const_int(bound, false);
        let next_ops = match ops {
            Operation::UnaryOp(op, opcode) => {
                let offset_phi_dst = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.dst.{}", nest).as_str(),
                )?;
                let offset_phi_src = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.src.{}", nest).as_str(),
                )?;
                let next_dst = update_offset!(op.dst, offset_phi_dst, exiting_bb);
                let next_src = update_offset!(op.src, offset_phi_src, exiting_bb);
                Operation::UnaryOp(
                    UnaryOps {
                        dst: next_dst,
                        src: next_src,
                    },
                    opcode,
                )
            }
            Operation::BinaryOp(op, opcode) => {
                let offset_phi_dst = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.dst.{}", nest).as_str(),
                )?;
                let offset_phi_lhs = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.lhs.{}", nest).as_str(),
                )?;
                let offset_phi_rhs = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.rhs.{}", nest).as_str(),
                )?;
                let next_dst = update_offset!(op.dst, offset_phi_dst, exiting_bb);
                let next_lhs = update_offset!(op.lhs, offset_phi_lhs, exiting_bb);
                let next_rhs = update_offset!(op.rhs, offset_phi_rhs, exiting_bb);
                Operation::BinaryOp(
                    BinaryOps {
                        dst: next_dst,
                        lhs: next_lhs,
                        rhs: next_rhs,
                    },
                    opcode,
                )
            }
        };
        if is_last {
            self.build_operation(&next_ops)?;
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
                .append_basic_block(*self.function, format!("loop.{}", nest).as_str());
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
            self.gen_nested_loop_rec(next_ops, next_loop_bb, nest + 1)
        }
    }

    fn gen_nested_loop(
        &self,
        op: Operation<'ctx>,
        preheader: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let header = self.context.append_basic_block(*self.function, "header");
        let exit = self.context.append_basic_block(*self.function, "exit");
        self.builder.build_unconditional_branch(header)?;
        self.builder.position_at_end(header);
        let loop_bb = LoopBB {
            preheader,
            header,
            exit,
        };
        self.gen_nested_loop_rec(op, loop_bb, 0)?;
        self.builder.position_at_end(exit);
        Ok(exit)
    }
}

type CodeType = unsafe extern "C" fn(*const *mut u8, *const *const u8);

#[derive(Debug)]
pub enum LLVMSessionError {
    CodeGenError(CodeGenError),
    ModelLoadError(ModelLoadError),
    TypeError(TypeError),
    FunctionLookupError(FunctionLookupError),
    OtherError(String),
}

pub struct LLVMSession<'ctx> {
    input_ty: Vec<ResolvedTensorType>,
    output_ty: Vec<ResolvedTensorType>,

    llvm_ctx: &'ctx Context,
    codegen: CodeGen<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
}

fn get_argument_types(
    graph: &Graph,
    values: &[ValueId],
) -> Result<Vec<ResolvedTensorType>, LLVMSessionError> {
    values
        .iter()
        .map(|&id| graph.get_resolved_tensor_type(id).cloned())
        .collect::<Option<Vec<_>>>()
        .ok_or(LLVMSessionError::TypeError(TypeError::UnresolvedInput))
}

impl<'ctx> LLVMSession<'ctx> {
    pub fn new<P: AsRef<Path>>(
        ctx: &'ctx Context,
        p: P,
        llvm_passes: &[LLVMPass],
    ) -> Result<Self, LLVMSessionError> {
        let mut model = Model::load_from_path(p).map_err(LLVMSessionError::ModelLoadError)?;
        model.graph.infer().map_err(LLVMSessionError::TypeError)?;
        let inputs_ty = get_argument_types(&model.graph, &model.graph.input_values())?;
        let outputs_ty = get_argument_types(&model.graph, &model.graph.output_values())?;
        let mut codegen = CodeGen::new(ctx).map_err(LLVMSessionError::CodeGenError)?;
        codegen
            .compile(&model.graph, llvm_passes)
            .map_err(LLVMSessionError::CodeGenError)?;

        let execution_engine = codegen
            .module
            .create_jit_execution_engine(OptimizationLevel::Aggressive)
            .map_err(CodeGenError::LLVMError)
            .map_err(LLVMSessionError::CodeGenError)?;

        Ok(LLVMSession {
            input_ty: inputs_ty,
            output_ty: outputs_ty,
            llvm_ctx: ctx,
            codegen,
            execution_engine,
        })
    }

    // TODO: Type check
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>, LLVMSessionError> {
        let mut outputs = self
            .output_ty
            .iter()
            .map(|ty| Tensor::zeros(ty.elem_type, ty.dims.clone()))
            .collect::<Vec<_>>();
        let output_ptrs = outputs
            .iter_mut()
            .map(|t| t.data.as_mut_ptr())
            .collect::<Vec<_>>();
        let input_ptrs = inputs.iter().map(|t| t.data.as_ptr()).collect::<Vec<_>>();
        let func: JitFunction<CodeType> = unsafe { self.execution_engine.get_function("main") }
            .map_err(LLVMSessionError::FunctionLookupError)?;
        unsafe { func.call(output_ptrs.as_ptr(), input_ptrs.as_ptr()) };
        Ok(outputs)
    }
}

#[cfg(test)]
macro_rules! make_tensor {
    ($ty: ty, $($expr: expr,)*) => {{
        let orig: ndarray::Array<$ty, _> = ndarray::array!($($expr,)*);
        let res: Result<(Tensor, _), _> = orig
            .clone()
            .try_into()
            .map(|t| (t, orig.clone()))
            .map_err(LLVMSessionError::TypeError);
        res
    }};
}

#[cfg(test)]
macro_rules! make_range_tensor {
    ($ty: ty, $($dim: expr),*) => {{
        let len = [$($dim),*].iter().product();
        let orig = ndarray::Array::from_iter((0..len).map(|x| x as $ty))
            .into_shape_with_order(($($dim),*))
            .map_err(|e| LLVMSessionError::OtherError(format!("{:?}", e)))?;
        let res: Result<(Tensor, _), _> = orig
            .clone()
            .try_into()
            .map(|t| (t, orig.clone()))
            .map_err(LLVMSessionError::TypeError);
        res
    }};
}

#[cfg(test)]
macro_rules! tensor_assert_eq {
    ($left: expr, $right: expr) => {{
        let right = Tensor::try_from($right).map_err(LLVMSessionError::TypeError)?;
        assert_eq!($left, right);
    }};
}

#[cfg(test)]
fn make_session<'ctx, P: AsRef<std::path::Path>>(
    ctx: &'ctx Context,
    path: P,
    passes: &[LLVMPass],
) -> Result<LLVMSession<'ctx>, LLVMSessionError> {
    use std::path::PathBuf;
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path);
    LLVMSession::new(ctx, path, passes)
}

#[cfg(test)]
fn with_session<P, F>(path: P, f: F) -> TestResult
where
    P: AsRef<std::path::Path>,
    F: FnOnce(LLVMSession) -> TestResult,
{
    let context = Context::create();
    let session = make_session(&context, path, &[])?;
    f(session)?;
    Ok(())
}

#[cfg(test)]
type TestResult = Result<(), LLVMSessionError>;

#[test]
fn test_add() -> TestResult {
    with_session("models/test/add.onnx", |session| {
        let (input0, orig0) = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],)?;
        let (input1, orig1) = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0],)?;
        let output = session.run(&[input0, input1])?;
        tensor_assert_eq!(output[0], orig0 + orig1);
        Ok(())
    })
}

#[test]
fn test_add_large() -> TestResult {
    with_session("models/test/add_large.onnx", |session| {
        let (input0, orig0) = make_tensor!(
            f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
        )?;
        let (input1, orig1) = make_tensor!(
            f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
        )?;
        let outputs = session.run(&[input0, input1])?;
        // session.codegen.target_machine.write_to_file(
        //     &session.codegen.module,
        //     inkwell::targets::FileType::Object,
        //     "test_add_large.o".as_ref(),
        // ).unwrap();
        tensor_assert_eq!(outputs[0], orig0 + orig1);
        Ok(())
    })
}

#[test]
fn test_relu() -> TestResult {
    with_session("models/test/relu.onnx", |session| {
        let (input, orig) =
            make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],)?;
        let output = session.run(&[input])?;
        tensor_assert_eq!(output[0], orig.mapv(|x| x.max(0.0)));
        Ok(())
    })
}

#[test]
fn transpose() -> TestResult {
    with_session("models/test/transpose.onnx", |session| {
        let (input, orig) = make_range_tensor!(f32, 1, 7, 5, 1)?;
        let output = session.run(&[input])?;
        let expected = orig.view().permuted_axes([2, 3, 1, 0]).to_owned();
        tensor_assert_eq!(output[0], expected);
        Ok(())
    })
}
