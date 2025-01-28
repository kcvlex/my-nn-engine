use crate::codegen::blas::{GemmArgs, Precision, BLAS};
use crate::codegen::omp::{ForkCallArgs, ScheduleType, StaticFiniArgs, StaticInitArgs, OMP};
use crate::codegen::plan;
use crate::codegen::plan::{AllocateInfo, AllocateType};
use crate::onnx::model::{Graph, Node, NodeId, ValueId};
use crate::onnx::operator;
use crate::onnx::operator::Operator;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::{DataType, ResolvedTensorType};
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::OptimizationLevel;

use inkwell::attributes::*;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::targets::FileType;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug)]
pub enum CodeGenError {
    BuilderError(BuilderError),
    LLVMError(inkwell::support::LLVMString),
    TargetMachineError(String),
    IntrinsicNotFound(String),
}

struct Intrinsics<'ll> {
    fmax_f32: FunctionValue<'ll>,
    fmax_f64: FunctionValue<'ll>,
    sqrt_f32: FunctionValue<'ll>,
    sqrt_f64: FunctionValue<'ll>,
    fma_f32: FunctionValue<'ll>,
    fma_f64: FunctionValue<'ll>,
    smin_i32: FunctionValue<'ll>,
    // lifetime_start: FunctionValue<'ctx>,
    // lifetime_end: FunctionValue<'ctx>,
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
    attrs: Attributes,
    intrinsics: Intrinsics<'ll>,
    blas: BLAS<'ll>,
    omp: OMP<'ll>,
    debug_stuff: DebugStuff<'ll>,
    target_machine: TargetMachine,
}

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

#[derive(Clone)]
struct FunctionTranslator<'a, 'ctx> {
    context: &'ctx Context,
    module: &'a Module<'ctx>,
    builder: &'a Builder<'ctx>,
    func: &'a FunctionValue<'ctx>,
    intrinsics: &'a Intrinsics<'ctx>,
    blas: &'a BLAS<'a>,
    omp: &'a OMP<'ctx>,

    #[allow(dead_code)]
    debug_stuff: &'a DebugStuff<'ctx>,
}

struct Attributes {
    noalias: Attribute,
    noundef: Attribute,
    cpu: Attribute,
    features: Attribute,
}

impl Attributes {
    fn new(context: &Context, target_machine: &TargetMachine) -> Self {
        let get_attr = |name: &str| {
            let kind_id = Attribute::get_named_enum_kind_id(name);
            context.create_enum_attribute(kind_id, 0)
        };

        let noalias = get_attr("noalias");
        let noundef = get_attr("noundef");
        let cpu = context
            .create_string_attribute("target-cpu", target_machine.get_cpu().to_str().unwrap());
        let features = context.create_string_attribute(
            "target-features",
            target_machine.get_feature_string().to_str().unwrap(),
        );

        Self {
            noalias,
            noundef,
            cpu,
            features,
        }
    }
    fn add_default_attributes<P>(&self, func: &FunctionValue<'_>, is_noalias: P)
    where
        P: Fn(usize) -> bool,
    {
        for i in 0..func.count_params() {
            if is_noalias(i as usize) {
                func.add_attribute(AttributeLoc::Param(i), self.noalias);
            }
            func.add_attribute(AttributeLoc::Param(i), self.noundef);
        }
        func.add_attribute(AttributeLoc::Function, self.cpu);
        func.add_attribute(AttributeLoc::Function, self.features);
    }
}

#[allow(dead_code)]
struct DebugStuff<'ll> {
    printf: FunctionValue<'ll>,
    fflush: FunctionValue<'ll>,
    float_fmt: GlobalValue<'ll>,
    i64_fmt: GlobalValue<'ll>,
    i64_i64_fmt: GlobalValue<'ll>,
    stdout: GlobalValue<'ll>,
}

impl<'ll> DebugStuff<'ll> {
    fn new(ctx: &'ll Context, module: &Module<'ll>, builder: &Builder<'ll>) -> Self {
        let i32_type = ctx.i32_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let printf = i32_type.fn_type(&[ptr_type.into()], true);
        let printf =
            module.add_function("printf", printf, Some(inkwell::module::Linkage::External));
        let fflush = i32_type.fn_type(&[ptr_type.into()], false);
        let fflush =
            module.add_function("fflush", fflush, Some(inkwell::module::Linkage::External));
        let stdout = module.add_global(ptr_type, None, "stdout");
        stdout.set_externally_initialized(true);
        let float_fmt = builder
            .build_global_string_ptr("%f\n", "float_fmt")
            .unwrap();
        let i64_fmt = builder.build_global_string_ptr("%ld\n", "i64_fmt").unwrap();
        let i64_i64_fmt = builder
            .build_global_string_ptr("%ld %ld\n", "i64_i64_fmt")
            .unwrap();
        Self {
            printf,
            fflush,
            float_fmt,
            i64_fmt,
            i64_i64_fmt,
            stdout,
        }
    }
}

// TODO: Target dependent value
fn memory_usage(graph: &Graph, value: ValueId) -> u64 {
    let result_ty = graph.get_resolved_tensor_type(value).unwrap();
    let data_size = match result_ty.elem_type {
        DataType::I64 => 8,
        DataType::U64 => 8,
        DataType::F32 => 4,
        DataType::F64 => 8,
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
        let builder = ll_ctx.create_builder();
        builder.position_at_end(entry);

        macro_rules! get_intrinsic {
            ($name: expr, $args: expr) => {{
                Intrinsic::find($name)
                    .and_then(|intrinsic| intrinsic.get_declaration(&unit.module, $args))
                    .ok_or_else(|| CodeGenError::IntrinsicNotFound($name.to_string()))
            }};
        }

        let fmax_f32 = get_intrinsic!("llvm.maximum", &[f32_ty, f32_ty])?;
        let fmax_f64 = get_intrinsic!("llvm.maximum", &[f64_ty, f64_ty])?;
        let sqrt_f32 = get_intrinsic!("llvm.sqrt", &[f32_ty])?;
        let sqrt_f64 = get_intrinsic!("llvm.sqrt", &[f64_ty])?;
        let fma_f32 = get_intrinsic!("llvm.fma", &[f32_ty, f32_ty, f32_ty])?;
        let fma_f64 = get_intrinsic!("llvm.fma", &[f64_ty, f64_ty, f64_ty])?;
        let smin_i32 = get_intrinsic!("llvm.smin", &[i32_ty, i32_ty])?;
        // let lifetime_start = get_intrinsic!("llvm.lifetime.start", &[i64_ty, ptr_ty])?;
        // let lifetime_end = get_intrinsic!("llvm.lifetime.end", &[i64_ty, ptr_ty])?;

        let intrinsics = Intrinsics {
            fmax_f32,
            fmax_f64,
            sqrt_f32,
            sqrt_f64,
            fma_f32,
            fma_f64,
            smin_i32,
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
            for alloc in alloc.iter() {
                let dst_ptr = match alloc.ty {
                    AllocateType::Chunk(chunk) => {
                        if alloc.is_first_use {
                            // TODO: type
                            let ptr = builder.build_array_malloc(
                                self.ll_ctx.f32_type(),
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

            let node = &self.gen_ctx.graph.nodes[*node_id];
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
            let len = ptrs[0].ty.dims.size();
            let len = len *
                (match ptrs[0].ty.elem_type {
                    DataType::F32 => 4,
                    DataType::F64 => 8,
                    DataType::I64 => 8,
                    DataType::U64 => 8,
                });
            builder.build_memcpy(
                ptrs[0].ptr,
                1,
                ptrs[1].ptr,
                1,
                self.ll_ctx
                    .i64_type()
                    .const_int(len.try_into().unwrap(), false),
            )?;
            builder.build_return(None)?;
            return Ok(());
        }

        // TODO
        let omp_ctx = None;
        let omp_parallel = node.meta.omp_parallel;
        let omp_for = node.meta.omp_for;

        macro_rules! gen_binaryop {
            ($op: expr) => {{
                let mut lhs = ptrs[1].clone();
                lhs.ty = lhs.ty.broadcast(&ptrs[0].ty.dims);
                let mut rhs = ptrs[2].clone();
                rhs.ty = rhs.ty.broadcast(&ptrs[0].ty.dims);
                let binop = BinaryOps {
                    dst: ptrs[0].clone(),
                    lhs,
                    rhs,
                };
                let op = Operation::BinaryOp(binop, $op);
                let op = OperationContext {
                    operation: op,
                    omp_ctx,
                    omp_parallel,
                    omp_for,
                };
                translator.build_nested_loop(op, entry, ptrs[0].ty.dims.ndim())
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
                let op = OperationContext {
                    operation: op,
                    omp_ctx,
                    omp_parallel,
                    omp_for,
                };
                translator.build_nested_loop(op, entry, ptrs[0].ty.dims.ndim())
            }};
        }

        macro_rules! gen_gemm {
            ($gemm: expr, $nest: expr) => {{
                let prec = match ptrs[0].ty.elem_type {
                    DataType::F32 => Precision::Single,
                    DataType::F64 => Precision::Double,
                    _ => unreachable!(),
                };
                let m = ptrs[0].ty.dims[$nest] as u32;
                let n = ptrs[0].ty.dims[$nest + 1] as u32;
                let k = ptrs[1].ty.dims[$nest + 1] as u32;
                let gemm = Gemm {
                    prec,
                    alpha: $gemm.alpha,
                    beta: $gemm.beta,
                    trans_a: $gemm.trans_a,
                    trans_b: $gemm.trans_b,
                    trans_c: $gemm.trans_c,
                    m,
                    n,
                    k,
                };
                let op = Operation::BinaryOp(
                    BinaryOps {
                        dst: ptrs[0].clone(),
                        lhs: ptrs[1].clone(),
                        rhs: ptrs[2].clone(),
                    },
                    BinaryOpcode::Gemm(gemm),
                );
                OperationContext {
                    operation: op,
                    omp_ctx,
                    omp_parallel,
                    omp_for,
                }
            }};
        }

        let exit = match node.op {
            Operator::Add => gen_binaryop!(BinaryOpcode::FloatAdd),
            Operator::ReLU => gen_unaryop!(UnaryOpcode::ReLU),
            // Operator::Transpose(ref perm) => {
            //     ptrs[1].perms = Some(perm.clone());
            //     gen_unaryop!(UnaryOpcode::Transpose)
            // }
            Operator::Contiguous => gen_unaryop!(UnaryOpcode::Transfer),
            Operator::MatMul => {
                let nest = ptrs[0].ty.dims.ndim() - 2;
                let gemm = gen_gemm!(
                    &operator::BLASGemm {
                        trans_a: false,
                        trans_b: false,
                        trans_c: false,
                        alpha: 1.0,
                        beta: 0.0,
                    },
                    nest
                );
                translator.build_nested_loop(gemm, entry, nest)
            }
            Operator::BLASGemm(ref gemm) => {
                let gemm = gen_gemm!(gemm, 0);
                translator.build_nested_loop(gemm, entry, 0)
            }
            Operator::Im2Col(ref im2col) => translator.build_im2col(
                (ptrs[0].ptr, &ptrs[0].ty.dims),
                (ptrs[1].ptr, &ptrs[1].ty.dims),
                ptrs[0].ty.elem_type,
                im2col,
                entry,
            ),
            Operator::ReduceMatrix(op) => {
                let m = ptrs[1].ty.dims[0] as u64;
                let n = ptrs[1].ty.dims[1] as u64;
                let elem_type = ptrs[0].ty.elem_type;
                translator.build_matrix_reduce(&ptrs, elem_type, (m, n), op, entry)
            }
            Operator::BatchNormalizationPerChannel(ref batchnorm) => {
                let dst = ptrs[0].ptr;
                let inputs = ptrs.iter().skip(1).map(|ptr| ptr.ptr).collect::<Vec<_>>();
                let m = ptrs[1].ty.dims[0] as u64;
                let n = (ptrs[1].ty.dims.size() as u64) / m;
                let elem_type = ptrs[0].ty.elem_type;
                translator.build_batchnorm_by_channel(
                    dst,
                    inputs.as_slice(),
                    elem_type,
                    (m, n),
                    entry,
                    batchnorm,
                )
            }
            _ => todo!("{:?}", node.op),
        }?;

        builder.position_at_end(exit);
        builder.build_return(None)?;

        Ok(())
    }
}

// TODO: Change `ty` to reference
#[derive(Debug, Clone)]
struct TensorPtr<'ctx> {
    ptr: PointerValue<'ctx>,
    ty: ResolvedTensorType,
    offset: IntValue<'ctx>,
    name: String,
}

impl<'ctx> TensorPtr<'ctx> {
    // TODO: Remove
    fn stride(&self, i: usize) -> usize {
        self.ty.stride(i)
    }

    fn to_outlined_nth_tensor(
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
            DataType::I64 | DataType::U64 => LLVMScalarType::LLVMInt(context.i64_type()),
        }
    }
}

enum Operation<'ctx> {
    UnaryOp(UnaryOps<'ctx>, UnaryOpcode),
    BinaryOp(BinaryOps<'ctx>, BinaryOpcode),
}

#[derive(Clone)]
struct OMPContext<'ctx> {
    global_tid: PointerValue<'ctx>,
    is_last: PointerValue<'ctx>,
    lb: PointerValue<'ctx>,
    ub: PointerValue<'ctx>,
    stride: PointerValue<'ctx>,
}

struct OperationContext<'ctx> {
    operation: Operation<'ctx>,
    omp_ctx: Option<OMPContext<'ctx>>,
    omp_parallel: Option<usize>,
    omp_for: Option<usize>,
}

impl OperationContext<'_> {
    fn to_paralleize(&self, nest: usize) -> bool {
        self.omp_parallel.map_or(false, |n| n == nest)
    }

    fn to_for(&self, nest: usize) -> bool {
        self.omp_for.map_or(false, |n| n == nest)
    }
}

#[derive(Debug, Clone)]
enum BinaryOpcode {
    // IntAdd,
    FloatAdd,
    Gemm(Gemm),
}

#[derive(Debug, Clone, Copy)]
struct Gemm {
    prec: Precision,
    trans_a: bool,
    trans_b: bool,
    trans_c: bool,
    alpha: f64,
    beta: f64,
    m: u32,
    n: u32,
    k: u32,
}

#[derive(Debug, Clone, Copy)]
enum UnaryOpcode {
    ReLU,
    Transfer,
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

impl<'ctx> Operation<'ctx> {
    fn result_dims(&self) -> &ResolvedTensorDims {
        match self {
            Operation::UnaryOp(op, _) => &op.dst.ty.dims,
            Operation::BinaryOp(op, _) => &op.dst.ty.dims,
        }
    }

    fn to_outlined(&self, translator: &FunctionTranslator<'_, 'ctx>) -> Result<Self, BuilderError> {
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

    fn outlined_type(&self, context: &'ctx Context) -> FunctionType<'ctx> {
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

    fn operands_as_vec(&self) -> Vec<TensorPtr<'ctx>> {
        match self {
            Operation::UnaryOp(UnaryOps { dst, src }, _) => vec![dst.clone(), src.clone()],
            Operation::BinaryOp(BinaryOps { dst, lhs, rhs }, _) => {
                vec![dst.clone(), lhs.clone(), rhs.clone()]
            }
        }
    }
}

impl<'ctx> OperationContext<'ctx> {
    fn to_outlined(
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

#[derive(Debug)]
struct Im2ColsInnerLoop<'a, 'ctx: 'a> {
    preheader: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,

    dst_ptr: PointerValue<'ctx>,
    dst_offset: IntValue<'ctx>,

    src_ptr: TensorPtr<'ctx>,
    is_pad: IntValue<'ctx>,
    pads: &'a [u64],
    outer_offsets: &'a [IntValue<'ctx>],

    elem_ty: FloatType<'ctx>,
    nest: u64,
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

    fn build_raw_load<T: BasicType<'ctx> + Copy>(
        &self,
        ty: T,
        ptr: PointerValue<'ctx>,
        offset: IntValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let gep = unsafe { self.builder.build_in_bounds_gep(ty, ptr, &[offset], "gep") }?;
        self.builder.build_load(ty, gep, "load")
    }

    fn build_raw_store<T: BasicType<'ctx>, V: BasicValue<'ctx>>(
        &self,
        ty: T,
        ptr: PointerValue<'ctx>,
        offset: IntValue<'ctx>,
        val: V,
    ) -> Result<(), BuilderError> {
        let gep = unsafe { self.builder.build_in_bounds_gep(ty, ptr, &[offset], "gep") }?;
        self.builder.build_store(gep, val).map(|_| ())
    }

    fn build_tail_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let call = self.builder.build_call(function, args, name)?;
        //call.set_tail_call(true);
        Ok(call)
    }

    fn build_im2col_by_channel_inner(
        &self,
        inner_loops: Im2ColsInnerLoop<'_, 'ctx>,
        im2col: &operator::Im2Col,
    ) -> Result<IntValue<'ctx>, BuilderError> {
        if inner_loops.nest as usize == inner_loops.outer_offsets.len() {
            let prolog = self.context.append_basic_block(*self.func, "inner.prolog");
            let normal = self.context.append_basic_block(*self.func, "inner.normal");
            let pad = self.context.append_basic_block(*self.func, "inner.pad");
            let epilog = self.context.append_basic_block(*self.func, "inner.epilog");

            self.builder.build_unconditional_branch(prolog)?;
            self.builder.position_at_end(prolog);
            self.builder
                .build_conditional_branch(inner_loops.is_pad, pad, normal)?;

            self.builder.position_at_end(pad);
            self.builder.build_unconditional_branch(epilog)?;

            self.builder.position_at_end(normal);
            let load_v = self.build_load(&inner_loops.src_ptr)?;
            self.builder.build_unconditional_branch(epilog)?;

            self.builder.position_at_end(epilog);
            let store_v = self.builder.build_phi(inner_loops.elem_ty, "store.v")?;
            store_v.add_incoming(&[(&load_v, normal), (&inner_loops.elem_ty.const_zero(), pad)]);
            self.build_raw_store(
                inner_loops.elem_ty,
                inner_loops.dst_ptr,
                inner_loops.dst_offset,
                store_v.as_basic_value(),
            )?;
            let next_dst_offset = self.builder.build_int_add(
                inner_loops.dst_offset,
                self.context.i64_type().const_int(1, false),
                "next.dst.offset",
            )?;
            self.builder.build_unconditional_branch(inner_loops.exit)?;
            return Ok(next_dst_offset);
        }

        let head = self.context.append_basic_block(*self.func, "inner.head");
        let exit = self.context.append_basic_block(*self.func, "inner.exit");
        let nest = inner_loops.nest;

        self.builder.position_at_end(inner_loops.preheader);
        self.builder.build_unconditional_branch(head)?;

        self.builder.position_at_end(head);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let dst_offset = self
            .builder
            .build_phi(self.context.i64_type(), "dst.offset")?;
        let src_inner_offset = self
            .builder
            .build_phi(self.context.i64_type(), "src.inner.offset")?;
        let dst_offset_int = dst_offset.as_basic_value().into_int_value();
        let src_inner_offset_int = src_inner_offset.as_basic_value().into_int_value();
        let src_offset = self.builder.build_int_add(
            inner_loops.outer_offsets[nest as usize],
            src_inner_offset_int,
            "src.offset",
        )?;
        let is_pad_left = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            src_offset,
            self.context
                .i64_type()
                .const_int(inner_loops.pads[nest as usize], false),
            "is.pad.left",
        )?;
        let is_pad_right = self.builder.build_int_compare(
            inkwell::IntPredicate::SLE,
            self.context.i64_type().const_int(
                u64::try_from(inner_loops.src_ptr.ty.dims[nest as usize + 2]).unwrap() +
                    inner_loops.pads[nest as usize],
                false,
            ),
            src_offset,
            "is.pad.right",
        )?;
        let is_pad_i = self
            .builder
            .build_or(is_pad_left, is_pad_right, "is.pad.i")?;
        let is_pad = self
            .builder
            .build_or(inner_loops.is_pad, is_pad_i, "is.pad")?;

        let src_offset = self.builder.build_int_sub(
            src_offset,
            self.context
                .i64_type()
                .const_int(inner_loops.pads[nest as usize], false),
            "src.offset",
        )?;
        let src_offset = self.builder.build_int_mul(
            src_offset,
            self.context.i64_type().const_int(
                inner_loops
                    .src_ptr
                    .stride(nest as usize + 2)
                    .try_into()
                    .unwrap(),
                false,
            ),
            "src.inner.offset.mul",
        )?;
        let src_offset =
            self.builder
                .build_int_add(inner_loops.src_ptr.offset, src_offset, "src.offset")?;
        let src_ptr = TensorPtr {
            ptr: inner_loops.src_ptr.ptr,
            ty: inner_loops.src_ptr.ty.clone(),
            offset: src_offset,
            name: format!("src.{}", nest),
        };
        let next_inner_loops = Im2ColsInnerLoop {
            preheader: head,
            exit,
            dst_ptr: inner_loops.dst_ptr,
            dst_offset: dst_offset_int,
            src_ptr,
            is_pad,
            pads: inner_loops.pads,
            outer_offsets: inner_loops.outer_offsets,
            elem_ty: inner_loops.elem_ty,
            nest: nest + 1,
        };

        let next_dst_offset = self.build_im2col_by_channel_inner(next_inner_loops, im2col)?;

        self.builder.position_at_end(exit);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let dilation = self
            .context
            .i64_type()
            .const_int(im2col.dilations[nest as usize].try_into().unwrap(), false);
        let src_inner_offset_next = self.builder.build_int_add(
            src_inner_offset.as_basic_value().into_int_value(),
            dilation,
            "src.inner.offset.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind_next,
            self.context.i64_type().const_int(
                im2col.one_kernel_shape[nest as usize].try_into().unwrap(),
                false,
            ),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, head, inner_loops.exit)?;
        ind.add_incoming(&[
            (&ind_next, exit),
            (&self.context.i64_type().const_zero(), inner_loops.preheader),
        ]);
        src_inner_offset.add_incoming(&[
            (&src_inner_offset_next, exit),
            (&self.context.i64_type().const_zero(), inner_loops.preheader),
        ]);
        dst_offset.add_incoming(&[
            (&next_dst_offset, exit),
            (&inner_loops.dst_offset, inner_loops.preheader),
        ]);
        Ok(next_dst_offset)
    }

    fn build_im2col_by_channel_outer(
        &self,
        dst_info: (PointerValue<'ctx>, IntValue<'ctx>),
        src_ptr: TensorPtr<'ctx>,
        offsets: Vec<IntValue<'ctx>>,
        nest: usize,
        blocks: (BasicBlock<'ctx>, BasicBlock<'ctx>),
        im2col: &operator::Im2Col,
    ) -> Result<IntValue<'ctx>, BuilderError> {
        let (dst_ptr, dst_offset) = dst_info;
        let (preheader, exit) = blocks;
        let max_nest = im2col.one_fm_shape.ndim();
        if nest == max_nest {
            let is_pad = self.context.bool_type().const_int(0, false);
            let pads = (0..max_nest)
                .map(|i| match &im2col.pad {
                    operator::ConvPad::NotSet(pad) => pad[i].0,
                    operator::ConvPad::Valid => 0,
                    operator::ConvPad::SameLower | operator::ConvPad::SameUpper => {
                        let padded_len = im2col.padded_len(i);
                        let pad_len = padded_len - src_ptr.ty.dims[i + 2];
                        let pad_left = pad_len / 2;
                        let add_left =
                            pad_len % 2 == 1 && matches!(im2col.pad, operator::ConvPad::SameLower);
                        pad_left + add_left as usize
                    }
                })
                .map(|x| x.try_into().unwrap())
                .collect::<Vec<u64>>();
            let elem_ty = match src_ptr.ty.elem_type {
                DataType::F32 => self.context.f32_type(),
                DataType::F64 => self.context.f64_type(),
                _ => todo!(),
            };

            let inner_loops = Im2ColsInnerLoop {
                preheader,
                exit,

                dst_ptr,
                dst_offset,

                src_ptr,
                is_pad,
                pads: &pads,
                outer_offsets: &offsets,

                elem_ty,
                nest: 0,
            };

            return self.build_im2col_by_channel_inner(inner_loops, im2col);
        }

        let head = self.context.append_basic_block(*self.func, "outer.head");
        let exiting = self.context.append_basic_block(*self.func, "outer.exit");

        self.builder.build_unconditional_branch(head)?;

        self.builder.position_at_end(head);
        let dst_offset_init = dst_offset;
        let src_offset = self
            .builder
            .build_phi(self.context.i64_type(), "src.offset")?;
        let dst_offset = self
            .builder
            .build_phi(self.context.i64_type(), "dst.offset")?;
        let src_offset_int = src_offset.as_basic_value().into_int_value();
        let dst_offset_int = dst_offset.as_basic_value().into_int_value();
        let mut offsets = offsets;
        offsets.push(src_offset_int);
        let next_dst_offset = self.build_im2col_by_channel_outer(
            (dst_ptr, dst_offset_int),
            src_ptr,
            offsets,
            nest + 1,
            (head, exiting),
            im2col,
        )?;

        self.builder.position_at_end(exiting);
        let next_dst_offset = if nest + 1 == max_nest {
            match im2col.channel {
                operator::Channel::Meld(channel) => self.builder.build_int_add(
                    next_dst_offset,
                    self.context.i64_type().const_int(
                        ((channel - 1) * im2col.one_kernel_shape.size())
                            .try_into()
                            .unwrap(),
                        false,
                    ),
                    "next.dst.offset",
                )?,
                operator::Channel::Split(_) => next_dst_offset,
            }
        } else {
            next_dst_offset
        };
        let padded_img_size: i64 = im2col.padded_len(nest).try_into().unwrap();
        let kernel_size: i64 = im2col.one_kernel_shape[nest].try_into().unwrap();
        let dilation: i64 = im2col.dilations[nest].try_into().unwrap();
        let next_src_offset = self.builder.build_int_add(
            src_offset_int,
            self.context
                .i64_type()
                .const_int(im2col.strides[nest].try_into().unwrap(), false),
            "src.offset.next",
        )?;
        let bound = padded_img_size - (kernel_size - 1) * dilation;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            next_src_offset,
            self.context
                .i64_type()
                .const_int(bound.try_into().unwrap(), false),
            "cond",
        )?;
        self.builder.build_conditional_branch(cond, head, exit)?;
        dst_offset.add_incoming(&[(&next_dst_offset, exiting), (&dst_offset_init, preheader)]);
        src_offset.add_incoming(&[
            (&next_src_offset, exiting),
            (&self.context.i64_type().const_zero(), preheader),
        ]);
        Ok(next_dst_offset)
    }

    fn build_im2col(
        &self,
        dst_info: (PointerValue<'ctx>, &ResolvedTensorDims),
        src_info: (PointerValue<'ctx>, &ResolvedTensorDims),
        elem_type: DataType,
        im2col: &operator::Im2Col,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let (dst_ptr, dst_shape) = dst_info;
        let (src_ptr, src_shape) = src_info;

        let header_nbatch = self
            .context
            .append_basic_block(*self.func, "im2col.header.nbatch");
        let exiting_nbatch = self
            .context
            .append_basic_block(*self.func, "im2col.exit.nbatch");
        let header_channel = self
            .context
            .append_basic_block(*self.func, "im2col.header.channel");
        let exiting_channel = self
            .context
            .append_basic_block(*self.func, "im2col.exit.channel");
        let exit = self.context.append_basic_block(*self.func, "im2col.exit");

        self.builder.build_unconditional_branch(header_nbatch)?;

        self.builder.position_at_end(header_nbatch);
        let ind_nbatch = self
            .builder
            .build_phi(self.context.i64_type(), "ind.nbatch")?;
        let offset_dst_nbatch = self
            .builder
            .build_phi(self.context.i64_type(), "offset.dst.nbatch")?;
        let offset_src_nbatch = self
            .builder
            .build_phi(self.context.i64_type(), "offset.src.nbatch")?;
        let offset_dst_nbatch_int = offset_dst_nbatch.as_basic_value().into_int_value();
        let offset_src_nbatch_int = offset_src_nbatch.as_basic_value().into_int_value();
        self.builder.build_unconditional_branch(header_channel)?;

        self.builder.position_at_end(header_channel);
        let ind_channel = self
            .builder
            .build_phi(self.context.i64_type(), "ind.channel")?;
        let offset_dst_channel = self
            .builder
            .build_phi(self.context.i64_type(), "offset.dst.channel")?;
        let offset_src_channel = self
            .builder
            .build_phi(self.context.i64_type(), "offset.src.channel")?;
        let offset_dst_channel_int = offset_dst_channel.as_basic_value().into_int_value();
        let offset_src_channel_int = offset_src_channel.as_basic_value().into_int_value();
        let offset_dst = self.builder.build_int_add(
            offset_dst_nbatch_int,
            offset_dst_channel_int,
            "offset.dst",
        )?;
        let offset_src = self.builder.build_int_add(
            offset_src_nbatch_int,
            offset_src_channel_int,
            "offset.src",
        )?;

        let src_ptr = TensorPtr {
            ptr: src_ptr,
            ty: ResolvedTensorType::new(elem_type, src_shape.clone()),
            offset: offset_src,
            name: "src".to_string(),
        };

        self.build_im2col_by_channel_outer(
            (dst_ptr, offset_dst),
            src_ptr,
            vec![],
            0,
            (header_channel, exiting_channel),
            im2col,
        )?;

        self.builder.position_at_end(exiting_channel);
        let next_ind_channel = self.builder.build_int_add(
            ind_channel.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.channel.next",
        )?;
        let next_offset_dst_channel = match im2col.channel {
            operator::Channel::Meld(_) => im2col.one_kernel_shape.size(),
            operator::Channel::Split(_) => {
                im2col.one_fm_shape.size() * im2col.one_kernel_shape.size()
            }
        };
        let next_offset_dst_channel = self.builder.build_int_add(
            offset_dst_channel_int,
            self.context
                .i64_type()
                .const_int(next_offset_dst_channel.try_into().unwrap(), false),
            "offset.dst.channel.next",
        )?;
        let next_offset_src_channel = self.builder.build_int_add(
            offset_src_channel_int,
            self.context.i64_type().const_int(
                (src_shape.size() / im2col.nbatch / im2col.channel.inner())
                    .try_into()
                    .unwrap(),
                false,
            ),
            "offset.src.channel.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            next_ind_channel,
            self.context
                .i64_type()
                .const_int(im2col.channel.inner().try_into().unwrap(), false),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, header_channel, exiting_nbatch)?;
        ind_channel.add_incoming(&[
            (&next_ind_channel, exiting_channel),
            (&self.context.i64_type().const_int(0, false), header_nbatch),
        ]);
        offset_dst_channel.add_incoming(&[
            (&next_offset_dst_channel, exiting_channel),
            (&offset_dst_nbatch_int, header_nbatch),
        ]);
        offset_src_channel.add_incoming(&[
            (&next_offset_src_channel, exiting_channel),
            (&self.context.i64_type().const_zero(), header_nbatch),
        ]);

        self.builder.position_at_end(exiting_nbatch);
        let next_ind_nbatch = self.builder.build_int_add(
            ind_nbatch.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.nbatch.next",
        )?;
        let next_offset_dst_nbatch = self.builder.build_int_add(
            offset_dst_nbatch_int,
            self.context.i64_type().const_int(
                (dst_shape.size() / im2col.nbatch).try_into().unwrap(),
                false,
            ),
            "offset.dst.nbatch.next",
        )?;
        let next_offset_src_nbatch = self.builder.build_int_add(
            offset_src_nbatch_int,
            self.context.i64_type().const_int(
                (src_shape.size() / im2col.nbatch).try_into().unwrap(),
                false,
            ),
            "offset.src.nbatch.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            next_ind_nbatch,
            self.context
                .i64_type()
                .const_int(im2col.nbatch.try_into().unwrap(), false),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, header_nbatch, exit)?;
        ind_nbatch.add_incoming(&[
            (&next_ind_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_int(0, false), entry),
        ]);
        offset_dst_nbatch.add_incoming(&[
            (&next_offset_dst_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_int(0, false), entry),
        ]);
        offset_src_nbatch.add_incoming(&[
            (&next_offset_src_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_int(0, false), entry),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    fn build_operation(&self, op: &Operation<'ctx>) -> Result<(), BuilderError> {
        match op {
            Operation::UnaryOp(op, opcode) => {
                match opcode {
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
                        let res = self
                            .build_tail_call(fmax, &[src.into(), zero.into()], "res")?
                            .try_as_basic_value()
                            .left()
                            .unwrap();
                        self.build_store(&op.dst, res)
                    }
                    UnaryOpcode::Transfer => {
                        // TODO
                        let len = match op.src.ty.elem_type {
                            DataType::F32 => 4,
                            DataType::F64 => 8,
                            DataType::I64 => 8,
                            DataType::U64 => 8,
                        };
                        let len_int = self
                            .context
                            .i64_type()
                            .const_int(len.try_into().unwrap(), false);
                        let src_offset =
                            self.builder
                                .build_int_mul(op.src.offset, len_int, "src.offset")?;
                        let dst_offset =
                            self.builder
                                .build_int_mul(op.dst.offset, len_int, "dst.offset")?;
                        for i in 0..len {
                            let src_offset = self.builder.build_int_add(
                                src_offset,
                                self.context
                                    .i64_type()
                                    .const_int(i.try_into().unwrap(), false),
                                "src.offset",
                            )?;
                            let dst_offset = self.builder.build_int_add(
                                dst_offset,
                                self.context
                                    .i64_type()
                                    .const_int(i.try_into().unwrap(), false),
                                "dst.offset",
                            )?;
                            let src_gep = unsafe {
                                self.builder.build_in_bounds_gep(
                                    self.context.i8_type(),
                                    op.src.ptr,
                                    &[src_offset],
                                    "src.gep",
                                )
                            }?;
                            let dst_gep = unsafe {
                                self.builder.build_in_bounds_gep(
                                    self.context.i8_type(),
                                    op.dst.ptr,
                                    &[dst_offset],
                                    "dst.gep",
                                )
                            }?;
                            let load =
                                self.builder
                                    .build_load(self.context.i8_type(), src_gep, "load")?;
                            self.builder.build_store(dst_gep, load)?;
                        }
                        Ok(())
                    }
                }
            }
            Operation::BinaryOp(op, opcode) => match opcode {
                BinaryOpcode::FloatAdd => {
                    let lhs = self.build_load(&op.lhs)?.into_float_value();
                    let rhs = self.build_load(&op.rhs)?.into_float_value();
                    let res = self.builder.build_float_add(lhs, rhs, "res")?;
                    self.build_store(&op.dst, res)
                }
                // BinaryOpcode::IntAdd => todo!(),
                BinaryOpcode::Gemm(ref gemm) => {
                    let prec = gemm.prec;
                    // TODO: Transpose
                    let mut trans_a = gemm.trans_a;
                    let mut trans_b = gemm.trans_b;
                    assert!(op.lhs.ty.dims.ndim() == 2);
                    assert!(op.rhs.ty.dims.ndim() == 2);
                    if op.lhs.ty.stride(0) < op.lhs.ty.stride(1) {
                        trans_a = !trans_a;
                    }
                    if op.rhs.ty.stride(0) < op.rhs.ty.stride(1) {
                        trans_b = !trans_b;
                    }
                    let gemm = GemmArgs {
                        a: (op.lhs.ptr, trans_a),
                        b: (op.rhs.ptr, trans_b),
                        c: (op.dst.ptr, gemm.trans_c),
                        alpha: gemm.alpha,
                        beta: gemm.beta,
                        m: gemm.m as u64,
                        n: gemm.n as u64,
                        k: gemm.k as u64,
                    };
                    self.blas.call_gemm(prec, &gemm, self.builder)?;
                    Ok(())
                }
            },
        }
    }

    fn build_matrix_reduce(
        &self,
        ptrs: &[TensorPtr<'ctx>],
        elem_ty: DataType,
        mn: (u64, u64),
        op: operator::ReduceOp,
        preheader: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let (row, col) = mn;

        let header0 = self.context.append_basic_block(*self.func, "header0");
        let exiting0 = self.context.append_basic_block(*self.func, "exiting0");
        let body = self.context.append_basic_block(*self.func, "body");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.build_unconditional_branch(header0)?;

        self.builder.position_at_end(header0);
        let ind0 = self.builder.build_phi(self.context.i64_type(), "ind0")?;
        let offset0 = self.builder.build_phi(self.context.i64_type(), "offset0")?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ind1 = self.builder.build_phi(self.context.i64_type(), "ind1")?;
        let fp_ty = match elem_ty {
            DataType::F32 => self.context.f32_type(),
            DataType::F64 => self.context.f64_type(),
            _ => todo!(),
        };
        let acc = self.builder.build_phi(fp_ty, "acc")?;
        let offset1 = self.builder.build_int_mul(
            ind1.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(ptrs[1].stride(1).try_into().unwrap(), false),
            "offset1",
        )?;
        let offset1 = self.builder.build_int_add(
            offset0.as_basic_value().into_int_value(),
            offset1,
            "offset1",
        )?;

        macro_rules! load {
            () => {{
                self.build_raw_load(fp_ty, ptrs[1].ptr, offset1)?
                    .into_float_value()
            }};
        }

        let (id_v, res) = match op {
            operator::ReduceOp::Max => {
                let (id_v, fmax) = match elem_ty {
                    DataType::F32 => (
                        self.context.f32_type().const_float(f32::MIN as f64),
                        self.intrinsics.fmax_f32,
                    ),
                    DataType::F64 => (
                        self.context.f64_type().const_float(f64::MIN),
                        self.intrinsics.fmax_f64,
                    ),
                    _ => todo!(),
                };
                let val = load!();
                let res = self
                    .build_tail_call(fmax, &[acc.as_basic_value().into(), val.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap();
                (id_v.as_basic_value_enum(), res)
            }
            operator::ReduceOp::Sum | operator::ReduceOp::Mean | operator::ReduceOp::Variance => {
                let fp_ty = match elem_ty {
                    DataType::F32 => self.context.f32_type(),
                    DataType::F64 => self.context.f64_type(),
                    _ => todo!(),
                };
                let zero = fp_ty.const_zero();
                let val = load!();
                let res = match op {
                    operator::ReduceOp::Sum | operator::ReduceOp::Mean => self
                        .builder
                        .build_float_add(acc.as_basic_value().into_float_value(), val, "res"),
                    operator::ReduceOp::Variance => {
                        // TODO?: stride
                        let mean = self
                            .build_raw_load(
                                fp_ty,
                                ptrs[2].ptr,
                                ind0.as_basic_value().into_int_value(),
                            )?
                            .into_float_value();
                        let diff = self.builder.build_float_sub(val, mean, "diff")?;
                        self.builder.build_float_mul(diff, diff, "diff.squared")
                    }
                    _ => unreachable!(),
                }?;
                (zero.as_basic_value_enum(), res.as_basic_value_enum())
            }
        };
        let ind1_next = self.builder.build_int_add(
            ind1.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind1.next",
        )?;
        let cond1 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind1_next,
            self.context.i64_type().const_int(col, false),
            "cond1",
        )?;
        self.builder
            .build_conditional_branch(cond1, body, exiting0)?;
        ind1.add_incoming(&[
            (&ind1_next, body),
            (&self.context.i64_type().const_zero(), header0),
        ]);
        acc.add_incoming(&[(&res, body), (&id_v, header0)]);

        self.builder.position_at_end(exiting0);
        let res = match op {
            operator::ReduceOp::Mean | operator::ReduceOp::Variance => {
                let div = fp_ty.const_float(col as f64);
                self.builder
                    .build_float_div(res.into_float_value(), div, "res")?
                    .as_basic_value_enum()
            }
            operator::ReduceOp::Max | operator::ReduceOp::Sum => res,
        };
        self.build_raw_store(
            fp_ty,
            ptrs[0].ptr,
            ind0.as_basic_value().into_int_value(),
            res,
        )?;
        let ind0_next = self.builder.build_int_add(
            ind0.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind0.next",
        )?;
        let offset0_next = self.builder.build_int_add(
            offset0.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(ptrs[1].stride(0).try_into().unwrap(), false),
            "offset0.next",
        )?;
        let cond0 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind0_next,
            self.context.i64_type().const_int(row, false),
            "cond0",
        )?;
        self.builder
            .build_conditional_branch(cond0, header0, exit)?;
        ind0.add_incoming(&[
            (&ind0_next, exiting0),
            (&self.context.i64_type().const_zero(), preheader),
        ]);
        offset0.add_incoming(&[
            (&offset0_next, exiting0),
            (&self.context.i64_type().const_zero(), preheader),
        ]);

        self.builder.position_at_end(exit);

        Ok(exit)
    }

    fn init_outlined(&self) -> Result<(OMPContext<'ctx>, LoopBB<'ctx>), BuilderError> {
        let entry = self.context.append_basic_block(*self.func, "entry");
        let body = self.context.append_basic_block(*self.func, "body");
        let exit = self.context.append_basic_block(*self.func, "exit");

        let i32_type = self.context.i32_type();

        self.builder.position_at_end(entry);
        let is_last = self.builder.build_alloca(i32_type, "is.last")?;
        let lb = self.builder.build_alloca(i32_type, "lb")?;
        let ub = self.builder.build_alloca(i32_type, "ub")?;
        let stride = self.builder.build_alloca(i32_type, "stride")?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(exit);
        self.builder.build_return(None)?;

        self.builder.position_at_end(body);

        let omp_ctx = OMPContext {
            global_tid: self.func.get_nth_param(0).unwrap().into_pointer_value(),
            is_last,
            lb,
            ub,
            stride,
        };
        let loop_bb = LoopBB {
            preheader: entry,
            header: body,
            exit,
        };
        Ok((omp_ctx, loop_bb))
    }

    fn build_omp_outlined(
        &self,
        op_ctx: OperationContext<'ctx>,
        nest: usize,
        max_nest: usize,
    ) -> Result<FunctionValue<'ctx>, BuilderError> {
        let fn_type = op_ctx.operation.outlined_type(self.context);
        let fn_name = format!("{}.outlined", self.func.get_name().to_str().unwrap());
        let outlined_fn = self.module.add_function(&fn_name, fn_type, None);

        let new_builder = self.context.create_builder();
        let translator = {
            let mut translator = self.clone();
            translator.builder = &new_builder;
            translator.func = &outlined_fn;
            translator
        };
        let (omp_ctx, loop_bb) = translator.init_outlined()?;
        let op_ctx = op_ctx.to_outlined(&translator, omp_ctx)?;
        translator.build_nested_loop_rec(op_ctx, loop_bb, nest, max_nest, None)?;

        Ok(outlined_fn)
    }

    fn build_omp_for(
        &self,
        op_ctx: OperationContext<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
        max_nest: usize,
    ) -> Result<(), BuilderError> {
        let new_header = self
            .context
            .append_basic_block(*self.func, "omp.for.header");
        let prolog_bb = self
            .context
            .append_basic_block(*self.func, "omp.for.prolog");
        let epilog_bb = self
            .context
            .append_basic_block(*self.func, "omp.for.epilog");

        let i32_type = self.context.i32_type();
        let len: u64 = op_ctx.operation.result_dims()[nest].try_into().unwrap();
        let len = i32_type.const_int(len - 1, false);
        let OMPContext {
            global_tid,
            is_last,
            lb,
            ub,
            stride,
        } = op_ctx.omp_ctx.clone().unwrap();

        let tid = self
            .builder
            .build_load(i32_type, global_tid, "global.tid")?
            .as_basic_value_enum()
            .into_int_value();
        let one = i32_type.const_int(1, false);
        for (ptr, val) in [
            (is_last, i32_type.const_zero()),
            (lb, i32_type.const_zero()),
            (ub, len),
            (stride, one),
        ] {
            // self.builder.build_call(self.intrinsics.lifetime_start, &[i64_type.const_int(4, false).into(), ptr.into()], "")?;
            self.builder.build_store(ptr, val)?;
        }

        let args = StaticInitArgs {
            tid,
            sched: ScheduleType::UnorderedStatic,
            is_last,
            lb,
            ub,
            stride,
            incr: one,
        };
        self.omp.static_init(self.builder, &args)?;
        let lb = self
            .builder
            .build_load(i32_type, lb, "lb")?
            .into_int_value();
        let ub = self
            .builder
            .build_load(i32_type, ub, "ub_omp")?
            .into_int_value();
        let ub = self
            .builder
            .build_call(self.intrinsics.smin_i32, &[ub.into(), len.into()], "ub_min")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let ub = self.builder.build_int_add(ub, one, "ub_open")?;
        let cond = self
            .builder
            .build_int_compare(inkwell::IntPredicate::SLT, lb, ub, "cond")?;
        self.builder
            .build_conditional_branch(cond, prolog_bb, epilog_bb)?;

        self.builder.position_at_end(epilog_bb);
        self.omp
            .static_fini(self.builder, &StaticFiniArgs { tid })?;
        self.builder.build_unconditional_branch(loop_bb.exit)?;

        self.builder.position_at_end(prolog_bb);
        let i64_type = self.context.i64_type();
        let lb = self.builder.build_int_s_extend(lb, i64_type, "lb")?;
        let ub = self.builder.build_int_s_extend(ub, i64_type, "ub")?;

        macro_rules! update_offset {
            ($ptr: expr) => {{
                let add = self.builder.build_int_mul(
                    lb,
                    i64_type.const_int($ptr.stride(nest).try_into().unwrap(), false),
                    "add",
                )?;
                $ptr.offset = self.builder.build_int_add($ptr.offset, add, "offset")?;
            }};
        }
        let mut op_ctx = op_ctx;
        match op_ctx.operation {
            Operation::UnaryOp(ref mut op, _) => {
                update_offset!(op.dst);
                update_offset!(op.src);
            }
            Operation::BinaryOp(ref mut op, _) => {
                update_offset!(op.dst);
                update_offset!(op.lhs);
                update_offset!(op.rhs);
            }
        }
        self.builder.build_unconditional_branch(new_header)?;

        op_ctx.omp_for = None;
        let new_loop_bb = LoopBB {
            preheader: prolog_bb,
            header: new_header,
            exit: epilog_bb,
        };
        let loop_range = Some((lb, ub));
        self.builder.position_at_end(new_header);
        self.build_nested_loop_rec(op_ctx, new_loop_bb, nest, max_nest, loop_range)
    }

    fn build_nested_loop_rec(
        &self,
        op_ctx: OperationContext<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
        max_nest: usize,
        loop_range: Option<(IntValue<'ctx>, IntValue<'ctx>)>,
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
                }
            }};
        }

        if op_ctx.to_paralleize(nest) {
            let tensors = op_ctx.operation.operands_as_vec();
            let mut args = Vec::with_capacity(tensors.len() * 2);
            for (i, tensor) in tensors.iter().enumerate() {
                let ptr_ptr = self
                    .builder
                    .build_alloca(tensor.ptr.get_type(), format!("tensor_{i}_ptr").as_str())?;
                let offset_ptr = self.builder.build_alloca(
                    tensor.offset.get_type(),
                    format!("tensor_{i}_offset").as_str(),
                )?;
                // TODO: Stop using hard-coded size value
                //self.builder.build_call(self.intrinsics.lifetime_start, &[
                //    self.context.i64_type().const_int(8, false).into(),
                //    ptr_ptr.into(),
                //], "")?;
                self.builder.build_store(ptr_ptr, tensor.ptr)?;
                // self.builder.build_call(self.intrinsics.lifetime_start, &[
                //     self.context.i64_type().const_int(8, false).into(),
                //     offset_ptr.into(),
                // ], "")?;
                self.builder.build_store(offset_ptr, tensor.offset)?;
                args.push(ptr_ptr);
                args.push(offset_ptr);
            }

            let outlined = self.build_omp_outlined(op_ctx, nest, max_nest)?;
            let args = ForkCallArgs { outlined, args };
            let call = self.omp.fork_call(self.builder, &args)?;
            //call.set_tail_call(true);
            self.builder.build_unconditional_branch(loop_bb.exit)?;

            return Ok(());
        }

        if op_ctx.to_for(nest) {
            return self.build_omp_for(op_ctx, loop_bb, nest, max_nest);
        }

        if nest == max_nest {
            self.build_operation(&op_ctx.operation)?;
            self.builder.build_unconditional_branch(loop_bb.exit)?;
            return Ok(());
        }
        let ind = self
            .builder
            .build_phi(self.context.i64_type(), format!("ind.{}", nest).as_str())?;
        let exiting_bb = self
            .context
            .append_basic_block(*self.func, format!("exit.{}", nest).as_str());
        let bound = match loop_range {
            Some((_, ub)) => ub,
            None => {
                let bound: u64 = op_ctx.operation.result_dims()[nest].try_into().unwrap();
                self.context.i64_type().const_int(bound, false)
            }
        };
        let mut next_op_ctx = op_ctx;
        next_op_ctx.operation = match next_op_ctx.operation {
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
        let next_bb = self
            .context
            .append_basic_block(*self.func, format!("loop.{}", nest).as_str());
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

        let ind_init = match loop_range {
            Some((lb, _)) => lb,
            None => self.context.i64_type().const_zero(),
        };
        ind.add_incoming(&[(&ind_init, loop_bb.preheader), (&ind_next, exiting_bb)]);
        self.builder.position_at_end(next_bb);
        let next_loop_bb = LoopBB {
            preheader: loop_bb.header,
            header: next_bb,
            exit: exiting_bb,
        };
        self.build_nested_loop_rec(next_op_ctx, next_loop_bb, nest + 1, max_nest, None)
    }

    fn build_nested_loop(
        &self,
        op_ctx: OperationContext<'ctx>,
        preheader: BasicBlock<'ctx>,
        max_nest: usize,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let header = self.context.append_basic_block(*self.func, "header");
        let exit = self.context.append_basic_block(*self.func, "exit");
        self.builder.build_unconditional_branch(header)?;
        self.builder.position_at_end(header);
        let loop_bb = LoopBB {
            preheader,
            header,
            exit,
        };
        self.build_nested_loop_rec(op_ctx, loop_bb, 0, max_nest, None)?;
        self.builder.position_at_end(exit);
        Ok(exit)
    }

    fn build_batchnorm_by_channel(
        &self,
        dst: PointerValue<'ctx>,
        inputs: &[PointerValue<'ctx>],
        elem_ty: DataType,
        mn: (u64, u64),
        preheader: BasicBlock<'ctx>,
        batchnorm: &operator::BatchNormalization,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let src = inputs[operator::args::BATCHNORM_DATA];
        let scale_ptr = inputs[operator::args::BATCHNORM_SCALE];
        let bias_ptr = inputs[operator::args::BATCHNORM_BIAS];
        let mean_ptr = inputs[operator::args::BATCHNORM_MEAN];
        let variance_ptr = inputs[operator::args::BATCHNORM_VAR];
        let (m, n) = mn;
        let (fp_type, sqrt, fma) = match elem_ty {
            DataType::F32 => (
                self.context.f32_type(),
                self.intrinsics.sqrt_f32,
                self.intrinsics.fma_f32,
            ),
            DataType::F64 => (
                self.context.f64_type(),
                self.intrinsics.sqrt_f64,
                self.intrinsics.fma_f64,
            ),
            _ => unreachable!(),
        };
        let epsilon = fp_type.const_float(batchnorm.epsilon as f64);

        let header = self.context.append_basic_block(*self.func, "entry");
        let exit = self.context.append_basic_block(*self.func, "exit");
        let exiting = self.context.append_basic_block(*self.func, "exiting");
        let body = self.context.append_basic_block(*self.func, "body");

        self.builder.build_unconditional_branch(header)?;

        self.builder.position_at_end(header);

        let ind0 = self.builder.build_phi(self.context.i64_type(), "ind0")?;
        let offset0 = self.builder.build_phi(self.context.i64_type(), "offset0")?;
        let ind0_int = ind0.as_basic_value().into_int_value();
        let offset0_int = offset0.as_basic_value().into_int_value();
        let scale = self
            .build_raw_load(fp_type, scale_ptr, ind0_int)?
            .into_float_value();
        let bias = self
            .build_raw_load(fp_type, bias_ptr, ind0_int)?
            .into_float_value();
        let mean = self
            .build_raw_load(fp_type, mean_ptr, ind0_int)?
            .into_float_value();
        let variance = self
            .build_raw_load(fp_type, variance_ptr, ind0_int)?
            .into_float_value();
        let factor = self.builder.build_float_add(variance, epsilon, "factor")?;
        let factor = self
            .build_tail_call(sqrt, &[factor.into()], "factor")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let factor = self.builder.build_float_div(scale, factor, "factor")?;

        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ind1 = self.builder.build_phi(self.context.i64_type(), "ind1")?;
        let offset1 = self.builder.build_int_add(
            offset0_int,
            ind1.as_basic_value().into_int_value(),
            "offset1",
        )?;
        let val = self
            .build_raw_load(fp_type, src, offset1)?
            .into_float_value();
        let val = self.builder.build_float_sub(val, mean, "val.sub.mean")?;
        let val = self
            .build_tail_call(fma, &[val.into(), factor.into(), bias.into()], "val")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.build_raw_store(fp_type, dst, offset1, val)?;
        let ind1_next = self.builder.build_int_add(
            ind1.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind1.next",
        )?;
        let cond1 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind1_next,
            self.context.i64_type().const_int(n, false),
            "cond1",
        )?;
        self.builder
            .build_conditional_branch(cond1, body, exiting)?;
        ind1.add_incoming(&[
            (&ind1_next, body),
            (&self.context.i64_type().const_zero(), header),
        ]);

        self.builder.position_at_end(exiting);
        let ind0_next = self.builder.build_int_add(
            ind0_int,
            self.context.i64_type().const_int(1, false),
            "ind0.next",
        )?;
        let offset0_next = self.builder.build_int_add(
            offset0_int,
            self.context.i64_type().const_int(n, false),
            "offset0.next",
        )?;
        let cond0 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind0_next,
            self.context.i64_type().const_int(m, false),
            "cond0",
        )?;
        self.builder.build_conditional_branch(cond0, header, exit)?;
        ind0.add_incoming(&[
            (&ind0_next, exiting),
            (&self.context.i64_type().const_zero(), preheader),
        ]);
        offset0.add_incoming(&[
            (&offset0_next, exiting),
            (&self.context.i64_type().const_zero(), preheader),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }
}
