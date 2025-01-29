mod blas;
mod llvm;
mod omp;
mod op;
mod plan;
pub mod session;
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
use crate::tensor::types::DataType;
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

#[cfg(test)]
mod test {
    use crate::codegen::session::*;
    use crate::tensor::Tensor;

    macro_rules! make_tensor {
        ($ty: ty, $($expr: expr,)*) => {{
            let orig: ndarray::Array<$ty, _> = ndarray::array!($($expr,)*);
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .into_dyn()
                .try_into()
                .map(|t| (t, orig.clone()))
                .map_err(SessionError::TypeError);
            res
        }};
    }

    macro_rules! make_range_tensor {
        ($ty: ty, $($dim: expr),*) => {{
            let len = [$($dim),*].iter().product();
            let orig = ndarray::Array::from_iter((0..len).map(|x| x as $ty))
                .into_shape_with_order(($($dim),*))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?;
            let res: Result<(Tensor, _), _> = orig
                .clone()
                .into_dyn()
                .try_into()
                .map(|t| (t, orig.clone()))
                .map_err(SessionError::TypeError);
            res
        }};
    }

    macro_rules! tensor_assert_eq {
        ($left: expr, $right: expr) => {{
            let right = Tensor::try_from($right).map_err(SessionError::TypeError)?;
            assert_eq!($left, right);
        }};
    }

    macro_rules! assert_eq_epsilon {
        ($left: expr, $right: expr, $epsilon: expr) => {{
            let res = $left.eq_with_epsilon(&$right, $epsilon);
            if !res {
                // For pretty print
                assert_eq!($left, $right);
            }
        }};
    }

    fn with_session<P, F>(path: P, f: F) -> TestResult
    where
        P: AsRef<std::path::Path>,
        F: FnOnce(Session) -> TestResult,
    {
        use std::path::PathBuf;
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/test/operator")
            .join(path);
        let session = Session::new(path, None, 10)?;
        f(session)?;
        Ok(())
    }

    type TestResult = Result<(), SessionError>;

    #[test]
    fn add() -> TestResult {
        with_session("add.onnx", |session| {
            let (input0, orig0) = make_tensor!(f32, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],)?;
            let (input1, orig1) = make_tensor!(f32, [1.0, 2.0, 3.0], [-4.0, -5.0, -6.0],)?;
            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], (orig0 + orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn add_large() -> TestResult {
        with_session("add_large.onnx", |session| {
            let (input0, orig0) = make_tensor!(
                f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
            )?;
            let (input1, orig1) = make_tensor!(
                f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0,
            )?;
            let outputs = session.run(&[input0, input1])?;
            tensor_assert_eq!(outputs[0], (orig0 + orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn add_broadcast() -> TestResult {
        with_session("add_broadcast.onnx", |session| {
            // (1 x 4 x 5)
            let (input0, orig0) = make_tensor!(
                f32,
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0],
                    [4.0, 5.0, 6.0, 7.0, 8.0],
                ],
            )?;

            // (2 x 3 x 1 x 1)
            let (input1, orig1) = make_tensor!(
                f32,
                [[[1.0]], [[2.0]], [[3.0]]],
                [[[1.0]], [[2.0]], [[3.0]]],
            )?;

            // (4 x 5)
            let (input2, orig2) = make_tensor!(
                f32,
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
                [30.0, 31.0, 32.0, 33.0, 34.0],
                [40.0, 41.0, 42.0, 43.0, 44.0],
            )?;

            let output = session.run(&[input0, input1, input2])?;
            tensor_assert_eq!(output[0], (orig0 + orig1 + orig2).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn relu() -> TestResult {
        with_session("relu.onnx", |session| {
            let (input, orig) =
                make_tensor!(f32, [[1.0, -2.0], [42.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],)?;
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], orig.mapv(|x| x.max(0.0)).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn transpose() -> TestResult {
        with_session("transpose.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 1, 7, 5, 1)?;
            let output = session.run(&[input])?;
            let expected = orig
                .view()
                .permuted_axes([2, 3, 1, 0])
                .to_owned()
                .into_dyn();
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn matmul() -> TestResult {
        with_session("matmul.onnx", |session| {
            let (input0, orig0) = make_tensor!(
                f32,
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            )?;
            let (input1, orig1) = make_tensor!(f32, [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],)?;
            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], orig0.dot(&orig1).into_dyn());
            Ok(())
        })
    }

    #[test]
    fn matmul_a_x_tb() -> TestResult {
        with_session("matmul_a_x_tb.onnx", |session| {
            let (input0, orig0) = make_range_tensor!(f32, 5, 7)?;
            let (input1, orig1) = make_range_tensor!(f32, 6, 7)?;

            let output = session.run(&[input0, input1])?;
            tensor_assert_eq!(output[0], orig0.dot(&orig1.t()).into_dyn());
            Ok(())
        })
    }

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-32
    #[test]
    fn conv() -> TestResult {
        with_session("conv.onnx", |session| {
            // (1 x 1 x 5 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 5 x 5)
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [12.0, 21.0, 27.0, 33.0, 24.0],
                    [33.0, 54.0, 63.0, 72.0, 51.0],
                    [63.0, 99.0, 108.0, 117.0, 81.0],
                    [93.0, 144.0, 153.0, 162.0, 111.0],
                    [72.0, 111.0, 117.0, 123.0, 84.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_strides0() -> TestResult {
        with_session("conv_with_strides0.onnx", |session| {
            // (1 x 1 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 4 x 3)
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [12.0, 27.0, 24.0],
                    [63.0, 108.0, 81.0],
                    [123.0, 198.0, 141.0],
                    [112.0, 177.0, 124.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_strides1() -> TestResult {
        with_session("conv_with_strides1.onnx", |session| {
            // (1 x 1 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 3 x 2)
            let (expected, _) =
                make_tensor!(f32, [[[54.0, 72.0], [144.0, 162.0], [234.0, 252.0],]],)?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_strides2() -> TestResult {
        with_session("conv_with_strides2.onnx", |session| {
            // (1 x 1 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 4 x 2)
            let (expected, _) = make_tensor!(
                f32,
                [[[21.0, 33.0], [99.0, 117.0], [189.0, 207.0], [171.0, 183.0],]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_channels() -> TestResult {
        with_session("conv_channels.onnx", |session| {
            // (1 x 2 x 7 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0, 29.0],
                        [30.0, 31.0, 32.0, 33.0, 34.0],
                    ],
                    [
                        [1.0, 2.0, 3.0, 4.0, 5.0],
                        [6.0, 7.0, 8.0, 9.0, 10.0],
                        [11.0, 12.0, 13.0, 14.0, 15.0],
                        [16.0, 17.0, 18.0, 19.0, 20.0],
                        [21.0, 22.0, 23.0, 24.0, 25.0],
                        [26.0, 27.0, 28.0, 29.0, 30.0],
                        [31.0, 32.0, 33.0, 34.0, 35.0],
                    ]
                ],
            )?;

            // (1 x 2 x 3 x 3)
            let (input1, _) = make_tensor!(
                f32,
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
                ],
            )?;
            let output = session.run(&[input0, input1])?;

            // (1 x 1 x 4 x 3)
            let (expected, _) = make_tensor!(
                f32,
                [[
                    [44.0, 93.0, 80.0],
                    [201.0, 342.0, 255.0],
                    [381.0, 612.0, 435.0],
                    [344.0, 543.0, 380.0],
                ]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn conv_with_autopad_same() -> TestResult {
        with_session("conv_with_autopad_same.onnx", |session| {
            // (1 x 1 x 5 x 5)
            let (input0, _) = make_tensor!(
                f32,
                [[
                    [0.0, 1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]],
            )?;

            // (1 x 1 x 3 x 3)
            let (input1, _) =
                make_tensor!(f32, [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0],]],)?;

            let output = session.run(&[input0, input1])?;

            let (expected, _) = make_tensor!(
                f32,
                [[[12.0, 27.0, 24.0], [63.0, 108.0, 81.0], [72.0, 117.0, 84.0],]],
            )?;
            assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn maxpool() -> TestResult {
        with_session("maxpool.onnx", |session| {
            let (input, orig) = make_range_tensor!(f32, 1, 3, 8, 8)?;
            let output = session.run(&[input])?;

            let expected = orig
                .windows((1, 1, 2, 2))
                .into_iter()
                .map(|w| w.iter().cloned().fold(f32::NEG_INFINITY, f32::max))
                .collect::<ndarray::Array<f32, _>>()
                .to_shape((1, 3, 7, 7))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn()
                .to_owned();
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn reducemax() -> TestResult {
        with_session("reducemax.onnx", |session| {
            let (input, orig) = make_tensor!(
                f32,
                [
                    [[5., 1.], [20., 2.]],
                    [[30., 1.], [40., 2.]],
                    [[55., 1.], [60., 2.]],
                ],
            )?;
            let expected = orig
                .fold_axis(ndarray::Axis(2), f32::NEG_INFINITY, |&a, &b| a.max(b))
                .into_shape_with_order((3, 2))
                .map_err(|e| SessionError::OtherError(format!("{:?}", e)))?
                .into_dyn();
            let output = session.run(&[input])?;
            tensor_assert_eq!(output[0], expected);
            Ok(())
        })
    }

    #[test]
    fn batchnorm() -> TestResult {
        with_session("batchnorm.onnx", |session| {
            let (input, _) = make_tensor!(
                f32,
                [
                    [[-0.7736, 1.1965], [0.6127, 1.7081]],
                    [[-1.2942, -0.1194], [0.2656, -0.3478]],
                    [[0.0629, 0.6267], [1.0625, -1.0402]],
                    [[0.9405, 0.8907], [-0.0534, -1.2017]]
                ],
                [
                    [[0.1489, -0.4435], [-0.9640, -1.7148]],
                    [[0.7103, 0.8480], [0.5366, -0.0574]],
                    [[-0.5479, -0.6636], [-0.8631, 1.0075]],
                    [[-0.3005, 0.8960], [0.9123, -0.5381]]
                ],
                [
                    [[0.5928, -1.7610], [1.4378, -1.8061]],
                    [[0.1554, 0.2030], [0.0264, 1.3788]],
                    [[0.0953, 2.1523], [-1.2667, 0.7831]],
                    [[-0.2522, 0.3387], [0.3128, 1.2057]]
                ],
                [
                    [[1.1783, 2.0076], [0.2719, 0.9309]],
                    [[0.2006, 0.3776], [0.7505, 0.2893]],
                    [[-0.3285, 2.2465], [1.1477, 1.3187]],
                    [[-0.4947, -0.3022], [-0.8595, -0.1885]]
                ],
            )?;

            let (expected, _) = make_tensor!(
                f32,
                [
                    [[2.6118e-01, 9.3978e-01], [7.3869e-01, 1.1160e+00]],
                    [[1.8126e-01, 7.8516e-01], [9.8307e-01, 6.6777e-01]],
                    [[6.2633e-01, 1.1128e+00], [1.4888e+00, -3.2538e-01]],
                    [[7.1125e-01, 6.9593e-01], [4.0556e-01, 5.2360e-02]]
                ],
                [
                    [[5.7894e-01, 3.7488e-01], [1.9563e-01, -6.2990e-02]],
                    [[1.2117e+00, 1.2825e+00], [1.1224e+00, 8.1704e-01]],
                    [[9.9338e-02, -5.0442e-04], [-1.7260e-01, 1.4413e+00]],
                    [[3.2954e-01, 6.9757e-01], [7.0256e-01, 2.5648e-01]]
                ],
                [
                    [[7.3184e-01, -7.8910e-02], [1.0229e+00, -9.4441e-02]],
                    [[9.2643e-01, 9.5088e-01], [8.6011e-01, 1.5553e+00]],
                    [[6.5432e-01, 2.4291e+00], [-5.2078e-01, 1.2477e+00]],
                    [[3.4440e-01, 5.2614e-01], [5.1817e-01, 7.9281e-01]]
                ],
                [
                    [[9.3350e-01, 1.2192e+00], [6.2131e-01, 8.4831e-01]],
                    [[9.4964e-01, 1.0406e+00], [1.2323e+00, 9.9528e-01]],
                    [[2.8862e-01, 2.5103e+00], [1.5623e+00, 1.7098e+00]],
                    [[2.6984e-01, 3.2904e-01], [1.5763e-01, 3.6400e-01]]
                ],
            )?;

            let output = session.run(&[input])?;
            assert_eq_epsilon!(output[0], expected, 0.001);
            Ok(())
        })
    }
}
