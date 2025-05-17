mod blas;
mod llvm;
mod omp;
mod op;
mod plan;
mod translator;

use crate::codegen::blas::*;
use crate::codegen::llvm::*;
use crate::codegen::omp::*;
use crate::codegen::op::*;
use crate::codegen::plan::*;
use crate::codegen::translator::*;
use crate::onnx::model::{Graph, Node, NodeId, ValueId};
use crate::onnx::operator;
use crate::onnx::operator::Operator;
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
use smallvec::smallvec;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum CodeGenError {
    BuilderError(BuilderError),
    LLVMError(inkwell::support::LLVMString),
    TargetMachineError(String),
    IntrinsicNotFound(String),
}

struct UnitInfo<'ll> {
    ty: UnitType,
    module: Module<'ll>,
    func: FunctionValue<'ll>,
    entry: BasicBlock<'ll>,
}

enum UnitType {
    Main,
    Node(NodeId),
}

pub struct CodeGenContext {
    pub graph: Graph,
    order: Vec<(NodeId, Vec<AllocateInfo>)>,
    value2alloc: HashMap<ValueId, AllocateInfo>,
    mem_size: Vec<u64>,
}

pub struct CodeGen<'ll, 'gen> {
    ll_ctx: &'ll Context,
    gen_ctx: &'gen CodeGenContext,
    unit: UnitInfo<'ll>,
    attrs: llvm::Attributes,
    intrinsics: llvm::Intrinsics<'ll>,
    blas: BLAS<'ll>,
    omp: OMP<'ll>,
    debug_stuff: llvm::DebugStuff<'ll>,
    target_machine: TargetMachine,
}

// TODO: correct?
unsafe impl Send for CodeGen<'_, '_> {}
unsafe impl Sync for CodeGen<'_, '_> {}

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

impl CodeGenContext {
    pub fn new(graph: Graph) -> Result<Self, CodeGenError> {
        let order = plan::plan(&graph);
        let value2alloc = order
            .iter()
            .flat_map(|(_, v)| v)
            .map(|info| (info.value_id, *info))
            .collect::<HashMap<_, _>>();

        let mem_size = calc_memsize(&graph, &order);

        Ok(CodeGenContext {
            graph,
            order,
            value2alloc,
            mem_size,
        })
    }

    fn need_to_generate(&self, node_id: NodeId) -> bool {
        let node = &self.graph.nodes[node_id];
        if node.is_dummy() {
            return false;
        }
        if matches!(node.op, Operator::Split(_)) {
            return false;
        }
        if let Operator::Identity = node.op {
            let chunk_in = self.value2alloc.get(&node.inputs[0]).map(|info| &info.ty);
            let chunk_out = self.value2alloc.get(&node.outputs[0]).map(|info| &info.ty);
            // TODO: correct?
            let res = match (chunk_in, chunk_out) {
                (Some(AllocateType::Chunk(in_chunk)), Some(AllocateType::Chunk(out_chunk))) => {
                    in_chunk != out_chunk
                }
                _ => false,
            };
            return res;
        }
        true
    }

    pub fn all_necessary_nodes(&self) -> Vec<NodeId> {
        self.order
            .iter()
            .map(|(id, _)| *id)
            .filter(|&id| self.need_to_generate(id))
            .collect()
    }
}

// TODO: Target dependent value
fn memory_usage(graph: &Graph, value: ValueId) -> u64 {
    let result_ty = graph.get_resolved_tensor_type(value).unwrap();
    let data_size = match result_ty.elem_type {
        DataType::SInt(SIntType::I32) => 4,
        DataType::SInt(SIntType::I64) => 8,
        DataType::UInt(UIntType::U64) => 8,
        DataType::Float(FloatType::F32) => 4,
        DataType::Float(FloatType::F64) => 8,
    };
    (result_ty.dims.size() * data_size).try_into().unwrap()
}

fn calc_memsize(graph: &Graph, order: &[(NodeId, Vec<AllocateInfo>)]) -> Vec<u64> {
    let mut mem_size = vec![
        0;
        order
            .iter()
            .flat_map(|(_, info)| info)
            .filter_map(|info| info.ty.chunk_id())
            .max()
            .map(|x| x + 1)
            .unwrap_or(0)
    ];
    for (_, vec) in order.iter() {
        for info in vec.iter() {
            if let Some(chunk_id) = info.ty.chunk_id() {
                mem_size[chunk_id] = mem_size[chunk_id].max(memory_usage(graph, info.value_id));
            }
        }
    }
    mem_size
}

impl CodeGenContext {
    pub fn new_codegen_for_node<'ll>(
        &self,
        node_id: NodeId,
        ll_ctx: &'ll Context,
    ) -> Result<CodeGen<'ll, '_>, CodeGenError> {
        let node = &self.graph.nodes[node_id];
        let target_machine = target_machine()?;
        let attrs = Attributes::new(ll_ctx, &target_machine);
        let module = ll_ctx.create_module(get_node_name_or(node, node_id).as_str());
        let func = self.declare_node_func(node_id, ll_ctx, &module, &attrs);
        let entry = ll_ctx.append_basic_block(func, "entry");
        let ty = UnitType::Node(node_id);
        let unit = UnitInfo {
            ty,
            module,
            func,
            entry,
        };

        self.new_codegen_with_func(ll_ctx, unit, attrs, target_machine)
    }

    pub fn new_codegen_for_main<'ll>(
        &self,
        ll_ctx: &'ll Context,
    ) -> Result<CodeGen<'ll, '_>, CodeGenError> {
        let module = ll_ctx.create_module("main");
        let builder = ll_ctx.create_builder();

        let ptr_ty = ll_ctx
            .ptr_type(AddressSpace::default())
            .as_basic_type_enum();
        let fn_type = ll_ctx
            .void_type()
            .fn_type(&[ptr_ty.into(), ptr_ty.into(), ptr_ty.into()], false);
        let main = module.add_function("main", fn_type, None);
        let entry = ll_ctx.append_basic_block(main, "entry");
        builder.position_at_end(entry);

        let target_machine = target_machine()?;
        let attrs = Attributes::new(ll_ctx, &target_machine);
        attrs.add_default_attributes(&main, |_| true);
        let unit = UnitInfo {
            ty: UnitType::Main,
            module,
            func: main,
            entry,
        };
        self.new_codegen_with_func(ll_ctx, unit, attrs, target_machine)
    }

    fn new_codegen_with_func<'ll>(
        &self,
        ll_ctx: &'ll Context,
        unit: UnitInfo<'ll>,
        attrs: Attributes,
        target_machine: TargetMachine,
    ) -> Result<CodeGen<'ll, '_>, CodeGenError> {
        let entry = unit.entry;
        let f32_ty = ll_ctx.f32_type().into();
        let f64_ty = ll_ctx.f64_type().into();
        let i32_ty = ll_ctx.i32_type().into();
        let i64_ty = ll_ctx.i64_type().into();
        let builder = ll_ctx.create_builder();
        builder.position_at_end(entry);

        macro_rules! get_intrinsic {
            ($name: expr, $args: expr) => {{
                Intrinsic::find($name)
                    .and_then(|intrinsic| intrinsic.get_declaration(&unit.module, $args))
                    .ok_or_else(|| CodeGenError::IntrinsicNotFound($name.to_string()))
            }};
        }

        let ceil = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.ceil", &[f32_ty])?,
            f_f64: get_intrinsic!("llvm.ceil", &[f64_ty])?,
        };
        let exp = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.exp", &[f32_ty])?,
            f_f64: get_intrinsic!("llvm.exp", &[f64_ty])?,
        };
        let fma = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.fma", &[f32_ty, f32_ty, f32_ty])?,
            f_f64: get_intrinsic!("llvm.fma", &[f64_ty, f64_ty, f64_ty])?,
        };
        let fmax = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.maxnum", &[f32_ty, f32_ty])?,
            f_f64: get_intrinsic!("llvm.maxnum", &[f64_ty, f64_ty])?,
        };
        let floor = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.floor", &[f32_ty])?,
            f_f64: get_intrinsic!("llvm.floor", &[f64_ty])?,
        };
        let log = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.log", &[f32_ty])?,
            f_f64: get_intrinsic!("llvm.log", &[f64_ty])?,
        };
        let sqrt = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.sqrt", &[f32_ty])?,
            f_f64: get_intrinsic!("llvm.sqrt", &[f64_ty])?,
        };
        let smin_i32 = get_intrinsic!("llvm.smin", &[i32_ty, i32_ty])?;
        let smin_i64 = get_intrinsic!("llvm.smin", &[i64_ty, i64_ty])?;
        let smax_i32 = get_intrinsic!("llvm.smax", &[i32_ty, i32_ty])?;
        let smax_i64 = get_intrinsic!("llvm.smax", &[i64_ty, i64_ty])?;
        let tanh = f64_ty.fn_type(&[f64_ty.into()], false);
        let tanh = unit.module.add_function("tanh", tanh, None);
        // let lifetime_start = get_intrinsic!("llvm.lifetime.start", &[i64_ty, ptr_ty])?;
        // let lifetime_end = get_intrinsic!("llvm.lifetime.end", &[i64_ty, ptr_ty])?;

        let intrinsics = Intrinsics {
            ceil,
            exp,
            floor,
            fma,
            fmax,
            log,
            sqrt,
            smin_i32,
            smin_i64,
            smax_i32,
            smax_i64,
            tanh,
            // lifetime_start,
            // lifetime_end,
        };

        let blas = BLAS::new(ll_ctx, &unit.module);
        let omp = OMP::new(ll_ctx, &unit.module, &builder).map_err(CodeGenError::BuilderError)?;

        let debug_stuff = DebugStuff::new(ll_ctx, &unit.module, &builder);

        Ok(CodeGen {
            ll_ctx,
            gen_ctx: self,
            unit,
            attrs,
            intrinsics,
            blas,
            omp,
            debug_stuff,
            target_machine,
        })
    }

    fn declare_node_func<'ctx>(
        &self,
        node_id: NodeId,
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        attrs: &Attributes,
    ) -> FunctionValue<'ctx> {
        let node = &self.graph.nodes[node_id];
        let allocs = node
            .outputs
            .iter()
            .chain(node.inputs.iter())
            .map(|&id| self.value2alloc.get(&id))
            .collect::<Vec<_>>();
        let mut is_noalias = vec![true; allocs.len()];
        for (i, alloc) in allocs.iter().enumerate() {
            // If it is None, it is an input or an initializer
            if let Some(info) = alloc {
                let cnt = allocs
                    .iter()
                    .filter_map(|x| *x)
                    .map(|&a| a.ty == info.ty)
                    .filter(|x| *x)
                    .count();
                is_noalias[i] = cnt == 1;
            }
        }

        let args = vec![ctx.ptr_type(AddressSpace::default()).into(); allocs.len()];
        let fn_type = ctx.void_type().fn_type(&args, false);
        let func = module.add_function(get_node_name_or(node, node_id).as_str(), fn_type, None);
        attrs.add_default_attributes(&func, |i| is_noalias[i]);
        func
    }
}

// TODO
fn get_node_name_or(node: &Node, node_id: NodeId) -> String {
    if node.name.is_empty() {
        format!("node.{}", node_id.index())
    } else {
        node.name.clone()
    }
}

impl<'ll> CodeGen<'ll, '_> {
    pub fn compile(&self) -> Result<(), CodeGenError> {
        (match self.unit.ty {
            UnitType::Main => self.compile_main(),
            UnitType::Node(node_id) => self.compile_node(node_id),
        })
        .map_err(CodeGenError::BuilderError)
    }

    pub fn run_opt_aggressive(&self) -> Result<(), CodeGenError> {
        let opt = inkwell::passes::PassBuilderOptions::create();
        self.unit
            .module
            .run_passes("default<O3>", &self.target_machine, opt)
            .map_err(CodeGenError::LLVMError)?;
        Ok(())
    }

    pub fn module(&self) -> &Module<'ll> {
        &self.unit.module
    }

    pub fn write_to_file<P: AsRef<Path>>(&self, ty: FileType, path: P) -> Result<(), CodeGenError> {
        self.target_machine
            .write_to_file(&self.unit.module, ty, path.as_ref())
            .map_err(CodeGenError::LLVMError)
    }

    fn init_main_args(&self) -> Result<HashMap<ValueId, PointerValue<'ll>>, BuilderError> {
        let mut ptr_values = HashMap::new();
        let builder = self.ll_ctx.create_builder();
        builder.position_at_end(self.unit.entry);
        macro_rules! init_ptr {
            ($value_id: expr, $ptr: expr, $i: expr) => {{
                let value = &self.gen_ctx.graph.values[$value_id];
                let ptr = unsafe {
                    builder.build_in_bounds_gep(
                        self.ll_ctx.ptr_type(AddressSpace::default()),
                        $ptr,
                        &[self.ll_ctx.i64_type().const_int($i as u64, false)],
                        value.name.as_str(),
                    )
                }?;
                let ptr = builder
                    .build_load(
                        self.ll_ctx.ptr_type(AddressSpace::default()),
                        ptr,
                        value.name.as_str(),
                    )?
                    .into_pointer_value();
                ptr_values.insert($value_id, ptr);
            }};
        }

        for (i, arr) in [&self.gen_ctx.graph.outputs, &self.gen_ctx.graph.inputs]
            .iter()
            .enumerate()
        {
            let ptr = self
                .unit
                .func
                .get_nth_param(i as u32)
                .unwrap()
                .into_pointer_value();
            for (i, node_id) in arr.iter().enumerate() {
                let value_id = match self.gen_ctx.graph.nodes[*node_id].op {
                    Operator::Input(v) | Operator::Output(v) => v,
                    _ => unreachable!(),
                };

                // TODO: necessary?
                if self.gen_ctx.graph.initializer.contains_key(&value_id) {
                    continue;
                }

                init_ptr!(value_id, ptr, i);
            }
        }

        {
            let ptr = self
                .unit
                .func
                .get_nth_param(2)
                .unwrap()
                .into_pointer_value();
            for (i, value_id) in self.gen_ctx.graph.initializer.keys().enumerate() {
                init_ptr!(*value_id, ptr, i);
            }
        }

        Ok(ptr_values)
    }

    fn compile_main(&self) -> Result<(), BuilderError> {
        let mut ptr_values = self.init_main_args()?;
        let mut chunk2ptr = HashMap::new();
        let builder = self.ll_ctx.create_builder();
        for (node_id, alloc) in self.gen_ctx.order.iter() {
            let function = if !self.gen_ctx.need_to_generate(*node_id) {
                None
            } else {
                Some(self.gen_ctx.declare_node_func(
                    *node_id,
                    self.ll_ctx,
                    &self.unit.module,
                    &self.attrs,
                ))
            };

            builder.position_at_end(self.unit.entry);
            let node = &self.gen_ctx.graph.nodes[*node_id];

            if let Operator::Split(ref split) = node.op {
                let src_ty = self
                    .gen_ctx
                    .graph
                    .get_resolved_tensor_type(node.inputs[0])
                    .unwrap();
                dbg!(&src_ty);
                let src = *ptr_values.get(&node.inputs[0]).unwrap();
                let axis = split.axis.index(src_ty.dims.ndim());
                let mut acc = 0;
                let elem_ty = src_ty.elem_type.llvm_type(self.ll_ctx);
                for output in node.outputs.iter() {
                    let ptr = unsafe {
                        builder.build_in_bounds_gep(
                            elem_ty,
                            src,
                            &[self.ll_ctx.i64_type().const_int(acc, false)],
                            format!("split.{}", output.index()).as_str(),
                        )
                    }?;
                    ptr_values.insert(*output, ptr);
                    let len = self
                        .gen_ctx
                        .graph
                        .get_resolved_tensor_type(*output)
                        .unwrap()
                        .dims[axis];
                    acc += (src_ty.stride(axis) * len) as u64;
                }
            } else {
                for alloc in alloc.iter() {
                    let dst_ptr = match alloc.ty {
                        AllocateType::Chunk(chunk) => {
                            if alloc.is_first_use {
                                // TODO: type
                                let ptr = builder.build_array_malloc(
                                    self.ll_ctx.i128_type(),
                                    self.ll_ctx
                                        .i64_type()
                                        .const_int(self.gen_ctx.mem_size[chunk], false),
                                    format!("chunk.{}", chunk).as_str(),
                                )?;
                                chunk2ptr.insert(chunk, ptr);
                                ptr
                            } else {
                                *chunk2ptr.get(&chunk).unwrap()
                            }
                        }
                        AllocateType::Input(v) | AllocateType::Output(v) => {
                            *ptr_values.get(&v).unwrap()
                        }
                    };
                    ptr_values.insert(alloc.value_id, dst_ptr);
                }
            }

            if let Some(function) = function {
                let args = node
                    .outputs
                    .iter()
                    .chain(node.inputs.iter())
                    .map(|&id| ptr_values.get(&id).unwrap())
                    .map(|ptr| (*ptr).into())
                    .collect::<Vec<_>>();
                let call = builder.build_call(function, &args[..], "")?;
                //call.set_tail_call(true);
            }
        }
        builder.position_at_end(self.unit.entry);
        builder.build_return(None)?;
        Ok(())
    }

    fn compile_node(&self, node_id: NodeId) -> Result<(), BuilderError> {
        let node = &self.gen_ctx.graph.nodes[node_id];
        // dbg!(&node);
        let args = node
            .outputs
            .iter()
            .chain(node.inputs.iter())
            .collect::<Vec<_>>();
        let builder = self.ll_ctx.create_builder();
        let entry = self.unit.entry;
        builder.position_at_end(entry);
        let translator = FunctionTranslator {
            context: self.ll_ctx,
            module: &self.unit.module,
            builder: &builder,
            func: &self.unit.func,
            intrinsics: &self.intrinsics,
            blas: &self.blas,
            omp: &self.omp,
            debug_stuff: &self.debug_stuff,
        };

        let ptrs = args
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let ptr = self
                    .unit
                    .func
                    .get_nth_param(i as u32)
                    .unwrap()
                    .into_pointer_value();
                let ty = self
                    .gen_ctx
                    .graph
                    .get_resolved_tensor_type(*id)
                    .unwrap()
                    .clone();
                let name = format!("ptr.{}", i);
                let offset = self.ll_ctx.i64_type().const_int(0, false);
                TensorPtr {
                    ptr,
                    ty,
                    offset,
                    name,
                }
            })
            .collect::<Vec<_>>();

        // TODO
        if let Operator::Identity = node.op {
            builder.position_at_end(entry);
            let len = ptrs[0]
                .ty
                .elem_type
                .llvm_type(self.ll_ctx)
                .size_of()
                .unwrap();
            let len = builder.build_int_mul(
                len,
                self.ll_ctx
                    .i64_type()
                    .const_int(ptrs[0].ty.dims.size().try_into().unwrap(), false),
                "len",
            )?;
            builder.build_memcpy(ptrs[0].ptr, 1, ptrs[1].ptr, 1, len)?;
            builder.build_return(None)?;
            return Ok(());
        }

        // TODO
        let omp_ctx = None;
        let omp_parallel = node.meta.omp_parallel;
        let omp_for = node.meta.omp_for;

        let mut ptrs = ptrs;

        macro_rules! nested_loop {
            ($op: expr, $nest: expr) => {{
                let op = OperationContext {
                    operation: $op,
                    omp_ctx,
                    omp_parallel,
                    omp_for,
                };
                translator.build_nested_loop(op, entry, $nest)
            }};
        }

        let convert_op = |op: &Operator| -> SingleOpcode {
            match op {
                Operator::Add => SingleOpcode::Add,
                Operator::BatchNormalization(bn) => SingleOpcode::BatchNorm(*bn),
                Operator::Contiguous => SingleOpcode::Transfer,
                Operator::Exp => SingleOpcode::Exp,
                Operator::LeakyReLU(v) => SingleOpcode::LeakyReLU(*v),
                Operator::Log => SingleOpcode::Log,
                Operator::Mul => SingleOpcode::Mul,
                Operator::ReLU => SingleOpcode::ReLU,
                Operator::Sigmoid => SingleOpcode::Sigmoid,
                Operator::Tanh => SingleOpcode::Tanh,
                _ => unreachable!(),
            }
        };

        // TODO: When same Input is used in multiple nodes
        let adjust_ptrs = |op: &Operator,
                           ptrs: &mut [TensorPtr<'_>],
                           operands: &[Option<usize>],
                           target_dim: &ResolvedTensorDims| {
            match op {
                Operator::Add | Operator::Mul => {
                    assert!(operands.len() == 2);
                    for i in operands.iter().filter_map(|x| *x) {
                        ptrs[i].ty = ptrs[i].ty.broadcast(target_dim);
                    }
                }

                Operator::BatchNormalization(_) => {
                    assert!(operands.len() == 5);
                    for i in operands.iter().skip(1).filter_map(|x| *x) {
                        ptrs[i].ty = ptrs[i].ty.extend_per_channel_params(&ptrs[0].ty.dims);
                    }
                    for i in operands.iter().filter_map(|x| *x) {
                        ptrs[i].ty = ptrs[i].ty.broadcast(target_dim);
                    }
                }

                Operator::Contiguous |
                Operator::Exp |
                Operator::LeakyReLU(_) |
                Operator::Log |
                Operator::ReLU |
                Operator::Sigmoid |
                Operator::Tanh => (),

                _ => unreachable!(),
            }
        };

        let exit = match &node.op {
            operator @ (Operator::Add |
            Operator::BatchNormalization(_) |
            Operator::Contiguous |
            Operator::Exp |
            Operator::LeakyReLU(_) |
            Operator::Log |
            Operator::Mul |
            Operator::ReLU |
            Operator::Sigmoid |
            Operator::Tanh) => {
                let operands: &'static [Option<usize>] = match operator {
                    Operator::Add | Operator::Mul => &[Some(0), Some(1)],
                    Operator::BatchNormalization(_) => {
                        &[Some(0), Some(1), Some(2), Some(3), Some(4)]
                    }
                    Operator::Exp |
                    Operator::LeakyReLU(_) |
                    Operator::Log |
                    Operator::ReLU |
                    Operator::Sigmoid |
                    Operator::Tanh |
                    Operator::Contiguous => &[],
                    _ => unreachable!(),
                };
                let target_dim = ptrs[0].ty.dims.clone();
                let nest = target_dim.ndim();
                adjust_ptrs(operator, &mut ptrs[1..], operands, &target_dim);
                let operator = convert_op(operator);
                let op = Operation {
                    opcode: operator.into(),
                    operands: ptrs.into(),
                };
                nested_loop!(op, nest)
            }

            Operator::Concat(ref concat) => {
                let dst = ptrs[0].clone();
                let axis = concat.axis.index(dst.ty.dims.ndim());
                translator.build_concat(dst, &ptrs[1..], entry, axis)
            }
            // Operator::Transpose(ref perm) => {
            //     ptrs[1].perms = Some(perm.clone());
            //     gen_unaryop!(UnaryOpcode::Transpose)
            // }
            // Operator::MatMul => {
            //     let nest = ptrs[0].ty.dims.ndim() - 2;
            //     let gemm = gen_gemm!(
            //         &operator::Gemm {
            //             trans_a: false,
            //             trans_b: false,
            //             alpha: 1.0,
            //             beta: 0.0,
            //         },
            //         nest
            //     );
            //     translator.build_nested_loop(gemm, entry, nest)
            // }
            Operator::MatMul => todo!(),
            Operator::Gemm(ref gemm) => {
                translator.build_gemm(&ptrs[0], &ptrs[1], &ptrs[2], ptrs.get(3), entry, gemm)
            }
            Operator::Im2Col(ref im2col) => {
                translator.build_im2col(&ptrs[0], &ptrs[1], im2col, entry)
            }
            Operator::ReduceMatrix(op) => {
                let m = ptrs[1].ty.dims[0] as u64;
                let n = ptrs[1].ty.dims[1] as u64;
                let elem_type = ptrs[0].ty.elem_type;
                translator.build_matrix_reduce(&ptrs, elem_type, (m, n), *op, entry)
            }
            Operator::Resize(ref resize) => translator.build_resize(
                ptrs[0].clone(),
                ptrs[1].clone(),
                ptrs.get(1 + operator::args::RESIZE_SCALES),
                ptrs.get(1 + operator::args::RESIZE_SIZES),
                entry,
                resize,
            ),
            Operator::ElementwiseOps(operator::ElementwiseOps { ops }) => {
                let target_dim = ptrs[0].ty.dims.clone();
                let nest = target_dim.ndim();
                let ops: Vec<_> = ops
                    .iter()
                    .inspect(|(op, args)| {
                        let operands = args
                            .iter()
                            .map(|arg| match arg {
                                operator::ElementwiseOpArg::Input(i) => Some(*i),
                                operator::ElementwiseOpArg::NthResult(_) => None,
                            })
                            .collect::<Vec<_>>();
                        adjust_ptrs(op, &mut ptrs[1..], &operands, &target_dim);
                    })
                    .map(|(op, args)| (convert_op(op), args.clone()))
                    .collect();
                let op = Operation {
                    opcode: Opcode::Fused(ops),
                    operands: ptrs.clone().into(),
                };
                nested_loop!(op, nest)
            }
            _ => todo!("{:?}", node.op),
        }?;

        builder.position_at_end(exit);
        builder.build_return(None)?;

        Ok(())
    }
}
