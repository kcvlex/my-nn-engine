use crate::codegen::blas::{GemmArgs, Precision, BLAS};
use crate::codegen::plan;
use crate::codegen::plan::{AllocateInfo, AllocateType, ChunkId};
use crate::model::{Graph, Node, NodeId, ValueId};
use crate::operator;
use crate::operator::Operator;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use crate::tensor::tensor::{DataType, ResolvedTensorType, TensorData};

use inkwell::attributes::*;
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::{Linkage, Module};
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine};
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use std::collections::HashMap;

#[derive(Debug)]
pub enum CodeGenError {
    BuilderError(BuilderError),
    LLVMError(inkwell::support::LLVMString),
    TargetMachineError(String),
    IntrinsicNotFound(String),
}

pub enum LLVMPass {
    // Module
    Attributor,

    // CGSCC
    ArgPromotion,
    AttributorCGSCC,
    Inline,

    // Loop
    LoopUnroll,
    LoopVectorize,
    SLPVectorize,

    // Function
    InstCombine,
    Reassociate,
    GlobalValueNumbering,
    SimplifyCFG,
    Mem2Reg,
}

impl LLVMPass {
    pub fn to_llvm_pass(&self) -> &'static str {
        match self {
            LLVMPass::Attributor => "attributor",
            LLVMPass::ArgPromotion => "argpromotion",
            LLVMPass::AttributorCGSCC => "attributor-cgscc",
            LLVMPass::Inline => "inline",
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

    main: FunctionValue<'ctx>,
    main_entry: BasicBlock<'ctx>,

    attrs: Attributes,
    target_machine: TargetMachine,

    intrinsics: Intrinsics<'ctx>,
    blas: BLAS<'ctx>,

    debug_stuff: DebugStuff<'ctx>,

    graph: Graph,
    order: Vec<(NodeId, AllocateInfo)>,
    ptr_values: HashMap<ValueId, PointerValue<'ctx>>,
    mem_size: Vec<u64>,
    chunk2ptr: HashMap<ChunkId, PointerValue<'ctx>>,
}

struct FunctionTranslator<'a, 'ctx> {
    context: &'ctx Context,
    builder: &'a Builder<'ctx>,
    function: &'a FunctionValue<'ctx>,
    intrinsics: &'a Intrinsics<'ctx>,
    blas: &'a BLAS<'a>,

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
    fn add_default_attributes<'ctx>(&self, function: &FunctionValue<'ctx>) {
        for i in 0..function.count_params() {
            function.add_attribute(AttributeLoc::Param(i), self.noalias);
            function.add_attribute(AttributeLoc::Param(i), self.noundef);
        }
        function.add_attribute(AttributeLoc::Function, self.cpu);
        function.add_attribute(AttributeLoc::Function, self.features);
    }
}

#[allow(dead_code)]
struct DebugStuff<'ctx> {
    printf: FunctionValue<'ctx>,
    float_fmt: GlobalValue<'ctx>,
    i64_fmt: GlobalValue<'ctx>,
}

impl<'ctx> DebugStuff<'ctx> {
    fn new(ctx: &'ctx Context, module: &Module<'ctx>, builder: &Builder<'ctx>) -> Self {
        let i32_type = ctx.i32_type();
        let ptr_type = ctx.ptr_type(AddressSpace::default());

        let printf = i32_type.fn_type(&[ptr_type.into()], true);
        let printf =
            module.add_function("printf", printf, Some(inkwell::module::Linkage::External));
        let float_fmt = builder
            .build_global_string_ptr("%f\n", "float_fmt")
            .unwrap();
        let i64_fmt = builder.build_global_string_ptr("%ld\n", "i64_fmt").unwrap();
        Self {
            printf,
            float_fmt,
            i64_fmt,
        }
    }
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

// TODO: Target dependent value
fn memory_usage(graph: &Graph, value: ValueId) -> u64 {
    let result_ty = graph.get_resolved_tensor_type(value).unwrap();
    let data_size = match result_ty.elem_type {
        DataType::I64 => 8,
        DataType::F32 => 4,
        DataType::F64 => 8,
    };
    (result_ty.dims.size() * data_size).try_into().unwrap()
}

fn calc_memsize(graph: &Graph, order: &[(NodeId, AllocateInfo)]) -> Vec<u64> {
    let mut mem_size = vec![
        0;
        order
            .iter()
            .filter_map(|(_, info)| info.ty.chunk_id())
            .max()
            .map(|x| x + 1)
            .unwrap_or(0)
    ];
    for (node_id, info) in order.iter() {
        if let Some(chunk_id) = info.ty.chunk_id() {
            let output_id = &graph.nodes[*node_id].outputs[0];
            mem_size[chunk_id] = mem_size[chunk_id].max(memory_usage(graph, *output_id));
        }
    }
    mem_size
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new(context: &'ctx Context, graph: Graph) -> Result<Self, CodeGenError> {
        let module = context.create_module("main");
        let builder = context.create_builder();

        let ptr_type = context.ptr_type(AddressSpace::default());
        let fn_type = context
            .void_type()
            .fn_type(&[ptr_type.into(), ptr_type.into()], false);
        let main = module.add_function("main", fn_type, None);
        let main_entry = context.append_basic_block(main, "entry");
        builder.position_at_end(main_entry);

        let target_machine = target_machine()?;

        let attrs = Attributes::new(context, &target_machine);
        attrs.add_default_attributes(&main);

        macro_rules! get_intrinsic {
            ($name: expr, $args: expr) => {{
                Intrinsic::find($name)
                    .and_then(|intrinsic| intrinsic.get_declaration(&module, $args))
                    .ok_or_else(|| CodeGenError::IntrinsicNotFound($name.to_string()))
            }};
        }

        let f32_ty = context.f32_type().into();
        let f64_ty = context.f64_type().into();

        let fmax_f32 = get_intrinsic!("llvm.maximum", &[f32_ty, f32_ty])?;
        let fmax_f64 = get_intrinsic!("llvm.maximum", &[f64_ty, f64_ty])?;

        let intrinsics = Intrinsics { fmax_f32, fmax_f64 };

        let blas = BLAS::new(context, &module);

        let debug_stuff = DebugStuff::new(context, &module, &builder);

        let order = plan::plan(&graph);
        let mem_size = calc_memsize(&graph, &order);
        let ptr_values = HashMap::new();
        let chunk2ptr = HashMap::new();

        Ok(CodeGen {
            context,
            module,
            builder,

            main,
            main_entry,

            target_machine,

            attrs,
            intrinsics,
            blas,

            debug_stuff,

            graph,
            order,
            ptr_values,
            mem_size,
            chunk2ptr,
        })
    }

    pub fn module(&self) -> &Module<'ctx> {
        &self.module
    }

    pub fn target_machine(&self) -> &TargetMachine {
        &self.target_machine
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn compile_with_passes(&mut self, passes: &[LLVMPass]) -> Result<(), CodeGenError> {
        self.compile_graph().map_err(CodeGenError::BuilderError)?;
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

    pub fn compile_default(&mut self) -> Result<(), CodeGenError> {
        self.compile_graph().map_err(CodeGenError::BuilderError)?;
        self.module
            .run_passes(
                "default<O3>",
                &self.target_machine,
                inkwell::passes::PassBuilderOptions::create(),
            )
            .map_err(CodeGenError::LLVMError)
    }

    // TODO: Adjust attributes
    fn create_function(&self, name: &str, argc: u32) -> FunctionValue<'ctx> {
        let mut vec = Vec::with_capacity(argc as usize);
        for _ in 0..argc {
            vec.push(self.context.ptr_type(AddressSpace::default()).into());
        }
        let fn_type = self.context.void_type().fn_type(&vec, false);
        let func = self
            .module
            .add_function(name, fn_type, Some(Linkage::Private));
        self.attrs.add_default_attributes(&func);
        func
    }

    fn init_data(&mut self) -> Result<(), BuilderError> {
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
        for (id, value) in self.graph.initializer.iter() {
            let name = format!("gv.{}", self.graph.values[*id].name);
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
            self.ptr_values.insert(*id, gv.as_pointer_value());
        }
        Ok(())
    }

    fn init_main_args(&mut self) -> Result<(), BuilderError> {
        for (i, arr) in [&self.graph.outputs, &self.graph.inputs].iter().enumerate() {
            let ptr = self
                .main
                .get_nth_param(i as u32)
                .unwrap()
                .into_pointer_value();
            for (i, node_id) in arr.iter().enumerate() {
                let value_id = match self.graph.nodes[*node_id].op {
                    Operator::Input(v) | Operator::Output(v) => v,
                    _ => unreachable!(),
                };
                let value = &self.graph.values[value_id];
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
                self.ptr_values.insert(value_id, ptr);
            }
        }
        Ok(())
    }

    pub fn compile_graph(&mut self) -> Result<(), BuilderError> {
        self.init_data()?;
        self.init_main_args()?;
        for (node, alloc) in self
            .order
            .iter()
            .map(|(id, info)| (&self.graph.nodes[*id], info))
        {
            let function = if node.op.is_identity() || node.is_dummy() {
                None
            } else {
                Some(self.compile_node(node)?)
            };

            self.builder.position_at_end(self.main_entry);
            let dst_ptr = match alloc.ty {
                AllocateType::Chunk(chunk) => {
                    if alloc.is_first_use {
                        let ptr = self.builder.build_array_malloc(
                            self.context.i8_type(),
                            self.context
                                .i64_type()
                                .const_int(self.mem_size[chunk], false),
                            format!("chunk.{}", chunk).as_str(),
                        )?;
                        self.chunk2ptr.insert(chunk, ptr);
                        ptr
                    } else {
                        *self.chunk2ptr.get(&chunk).unwrap()
                    }
                }
                AllocateType::Input(v) | AllocateType::Output(v) => {
                    *self.ptr_values.get(&v).unwrap()
                }
            };

            for &id in node.outputs.iter() {
                self.ptr_values.insert(id, dst_ptr);
            }

            if let Some(function) = function {
                let mut args = node.outputs.clone();
                args.extend(node.inputs.clone());
                let args = args
                    .iter()
                    .map(|&id| self.ptr_values.get(&id).unwrap())
                    .map(|ptr| (*ptr).into())
                    .collect::<Vec<_>>();
                let call = self.builder.build_call(function, &args[..], "")?;
                call.set_tail_call(true);
            }
        }

        for ptr in self.chunk2ptr.values() {
            self.builder.build_free(*ptr)?;
        }
        self.builder.position_at_end(self.main_entry);
        self.builder.build_return(None)?;
        Ok(())
    }

    fn compile_node(&self, node: &Node) -> Result<FunctionValue<'ctx>, BuilderError> {
        let mut args = node.outputs.clone();
        args.extend(node.inputs.clone());

        let function = self.create_function(node.name.as_str(), args.len() as u32);
        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);
        let translator = FunctionTranslator {
            context: self.context,
            builder: &self.builder,
            function: &function,
            intrinsics: &self.intrinsics,
            blas: &self.blas,
            debug_stuff: &self.debug_stuff,
        };

        let mut ptrs = args
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let ptr = function
                    .get_nth_param(i as u32)
                    .unwrap()
                    .into_pointer_value();
                let ty = self.graph.get_resolved_tensor_type(id).unwrap().clone();
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
                Operation::BinaryOp(
                    BinaryOps {
                        dst: ptrs[0].clone(),
                        lhs: ptrs[1].clone(),
                        rhs: ptrs[2].clone(),
                    },
                    BinaryOpcode::Gemm(gemm),
                )
            }};
        }

        let exit = match node.op {
            Operator::Add => gen_binaryop!(BinaryOpcode::FloatAdd),
            Operator::ReLU => gen_unaryop!(UnaryOpcode::ReLU),
            Operator::Transpose(ref perm) => {
                ptrs[1].perms = Some(perm.clone());
                gen_unaryop!(UnaryOpcode::Transpose)
            }
            Operator::MatMul => {
                let nest = ptrs[0].ty.dims.ndim() - 2;
                let gemm = gen_gemm!(
                    &operator::Gemm {
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
            Operator::Gemm(ref gemm) => {
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
            Operator::ReduceMatrix(op) => match op {
                operator::ReduceOp::Max => {
                    translator.build_matrix_reduce(&ptrs[0], &ptrs[1], op, entry)
                }
                _ => todo!("{:?}", node.op),
            },
            _ => todo!("{:?}", node.op),
        }?;

        self.builder.position_at_end(exit);
        self.builder.build_return(None)?;
        Ok(function)
    }
}

// TODO: Change `ty` to reference
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

    fn build_im2col_by_channel_inner(
        &self,
        inner_loops: Im2ColsInnerLoop<'_, 'ctx>,
        im2col: &operator::Im2Col,
    ) -> Result<IntValue<'ctx>, BuilderError> {
        if inner_loops.nest as usize == inner_loops.outer_offsets.len() {
            let prolog = self
                .context
                .append_basic_block(*self.function, "inner.prolog");
            let normal = self
                .context
                .append_basic_block(*self.function, "inner.normal");
            let pad = self.context.append_basic_block(*self.function, "inner.pad");
            let epilog = self
                .context
                .append_basic_block(*self.function, "inner.epilog");

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
            let gep = unsafe {
                self.builder.build_in_bounds_gep(
                    inner_loops.elem_ty,
                    inner_loops.dst_ptr,
                    &[inner_loops.dst_offset],
                    "gep",
                )
            }?;
            self.builder.build_store(gep, store_v.as_basic_value())?;
            let next_dst_offset = self.builder.build_int_add(
                inner_loops.dst_offset,
                self.context.i64_type().const_int(1, false),
                "next.dst.offset",
            )?;
            self.builder.build_unconditional_branch(inner_loops.exit)?;
            return Ok(next_dst_offset);
        }

        let head = self
            .context
            .append_basic_block(*self.function, "inner.head");
        let exit = self
            .context
            .append_basic_block(*self.function, "inner.exit");
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
                u64::try_from(inner_loops.src_ptr.ty.dims[nest as usize + 2]).unwrap()
                    + inner_loops.pads[nest as usize],
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
            perms: inner_loops.src_ptr.perms.clone(),
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

        let head = self
            .context
            .append_basic_block(*self.function, "outer.head");
        let exiting = self
            .context
            .append_basic_block(*self.function, "outer.exit");

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
            .append_basic_block(*self.function, "im2col.header.nbatch");
        let exiting_nbatch = self
            .context
            .append_basic_block(*self.function, "im2col.exit.nbatch");
        let header_channel = self
            .context
            .append_basic_block(*self.function, "im2col.header.channel");
        let exiting_channel = self
            .context
            .append_basic_block(*self.function, "im2col.exit.channel");
        let exit = self
            .context
            .append_basic_block(*self.function, "im2col.exit");

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
            perms: None,
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
                    let gemm = GemmArgs {
                        a: (op.lhs.ptr, gemm.trans_a),
                        b: (op.rhs.ptr, gemm.trans_b),
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
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        op: operator::ReduceOp,
        preheader: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let elem_ty = src.ty.elem_type;
        let row: u64 = src.ty.dims[0].try_into().unwrap();
        let col: u64 = src.ty.dims[1].try_into().unwrap();
        let src = src.ptr;
        let dst = dst.ptr;

        let header0 = self.context.append_basic_block(*self.function, "header0");
        let exiting0 = self.context.append_basic_block(*self.function, "exiting0");
        let body = self.context.append_basic_block(*self.function, "body");
        let exit = self.context.append_basic_block(*self.function, "exit");

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
        let offset1 = self.builder.build_int_add(
            offset0.as_basic_value().into_int_value(),
            ind1.as_basic_value().into_int_value(),
            "offset1",
        )?;

        macro_rules! load {
            () => {{
                let gep = unsafe {
                    self.builder
                        .build_in_bounds_gep(fp_ty, src, &[offset1], "gep")
                }?;
                self.builder.build_load(fp_ty, gep, "val")?
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
            operator::ReduceOp::Sum | operator::ReduceOp::Average => {
                let fp_ty = match elem_ty {
                    DataType::F32 => self.context.f32_type(),
                    DataType::F64 => self.context.f64_type(),
                    _ => todo!(),
                };
                let zero = fp_ty.const_zero();
                let val = load!();
                let res = self.builder.build_float_add(
                    acc.as_basic_value().into_float_value(),
                    val.into_float_value(),
                    "res",
                )?;
                (zero.as_basic_value_enum(), res.as_basic_value_enum())
            }
            _ => todo!(),
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
        let gep = unsafe {
            self.builder.build_in_bounds_gep(
                fp_ty,
                dst,
                &[ind0.as_basic_value().into_int_value()],
                "gep",
            )
        }?;
        let res = match op {
            operator::ReduceOp::Average | operator::ReduceOp::Mean => {
                let div = fp_ty.const_float(col as f64);
                self.builder
                    .build_float_div(res.into_float_value(), div, "res")?
                    .as_basic_value_enum()
            }
            operator::ReduceOp::Max | operator::ReduceOp::Sum => res,
        };
        self.builder.build_store(gep, res)?;
        let ind0_next = self.builder.build_int_add(
            ind0.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind0.next",
        )?;
        let offset0_next = self.builder.build_int_add(
            offset0.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(col, false),
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

    fn build_nested_loop_rec(
        &self,
        ops: Operation<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
        max_nest: usize,
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

        if nest == max_nest {
            self.build_operation(&ops)?;
            self.builder.build_unconditional_branch(loop_bb.exit)?;
            return Ok(());
        }
        let ind = self
            .builder
            .build_phi(self.context.i64_type(), format!("ind.{}", nest).as_str())?;
        let exiting_bb = self
            .context
            .append_basic_block(*self.function, format!("exit.{}", nest).as_str());
        let bound: u64 = ops.result_dims()[nest].try_into().unwrap();
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
        self.build_nested_loop_rec(next_ops, next_loop_bb, nest + 1, max_nest)
    }

    fn build_nested_loop(
        &self,
        op: Operation<'ctx>,
        preheader: BasicBlock<'ctx>,
        max_nest: usize,
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
        self.build_nested_loop_rec(op, loop_bb, 0, max_nest)?;
        self.builder.position_at_end(exit);
        Ok(exit)
    }
}
