use crate::codegen::memory;
use crate::model::{Graph, Node, ValueId};
use crate::operator::*;
use crate::tensor::tensor::{DataType, ResolvedTensorType, Tensor};
use cranelift::codegen::{ir, isa::OwnedTargetIsa};
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, FuncId, Linkage, Module, ModuleError};
use std::collections::HashMap;
use std::slice;

#[derive(Debug)]
pub enum CodegenError {
    ModuleError(ModuleError),
    DeallocationError(Value),
    ValueNotFound(ValueId),
    UnresolvedShape,
}

type CodegenResult<T> = Result<T, CodegenError>;

impl ResolvedTensorType {
    fn value_type(&self) -> Type {
        match self.elem_type {
            DataType::I64 => types::I64,
            DataType::F32 => types::F32,
            DataType::F64 => types::F64,
        }
    }
}

pub struct JIT {
    builder_ctx: FunctionBuilderContext,
    isa: OwnedTargetIsa,
    ctx: codegen::Context,
    data_description: DataDescription,
    module: JITModule,
}

impl Default for JIT {
    fn default() -> Self {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();
        let isa_builder = cranelift_native::builder().unwrap_or_else(|msg| {
            panic!("host machine is not supported: {}", msg);
        });
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .unwrap();
        let builder = JITBuilder::with_isa(isa.clone(), cranelift_module::default_libcall_names());

        let module = JITModule::new(builder);
        Self {
            builder_ctx: FunctionBuilderContext::new(),
            isa,
            ctx: module.make_context(),
            data_description: DataDescription::new(),
            module,
        }
    }
}

impl JIT {
    pub fn create_data(&mut self, name: &str, contents: Vec<u8>) -> CodegenResult<DataId> {
        self.data_description.define(contents.into_boxed_slice());
        let id = self
            .module
            .declare_data(name, Linkage::Export, true, false)
            .map_err(CodegenError::ModuleError)?;

        self.module
            .define_data(id, &self.data_description)
            .map_err(CodegenError::ModuleError)?;
        self.data_description.clear();
        self.module
            .finalize_definitions()
            .map_err(CodegenError::ModuleError)?;
        Ok(id)
    }

    pub fn create_function(&mut self, name: &str, signature: &Signature) -> CodegenResult<()> {
        let id = self
            .module
            .declare_function(name, Linkage::Export, signature)
            .map_err(CodegenError::ModuleError)?;
        self.module
            .define_function(id, &mut self.ctx)
            .map_err(CodegenError::ModuleError)?;
        self.module.clear_context(&mut self.ctx);
        self.module
            .finalize_definitions()
            .map_err(CodegenError::ModuleError)?;
        Ok(())
    }

    pub fn get_finalized_data(&self, id: DataId) -> &[u8] {
        let buffer = self.module.get_finalized_data(id);
        unsafe { slice::from_raw_parts(buffer.0, buffer.1) }
    }
}

#[derive(Clone)]
struct TypedValue {
    ty: Type,
    value: Value,
}

// (M, K) * (K, N) = (M, N)
#[derive(Debug, Clone)]
struct MatMulShape {
    m: usize,
    k: usize,
    n: usize,
}

impl MatMulShape {
    fn new(lhs: &ResolvedTensorType, rhs: &ResolvedTensorType) -> Self {
        Self {
            m: lhs.dims[0],
            k: lhs.dims[1],
            n: rhs.dims[1],
        }
    }
}

struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    isa: OwnedTargetIsa,
    ptr_ty: Type,
    malloc: ir::FuncRef,
    allocator: memory::Allocator,
    value2fragment: HashMap<Value, memory::Fragment>,
}

pub struct GraphCompiler<'a> {
    func_id: FuncId,
    translator: FunctionTranslator<'a>,
    graph: &'a Graph,
    id2value: HashMap<ValueId, Value>,
}

impl<'a> GraphCompiler<'a> {
    fn new(jit: &'a mut JIT, graph: &'a Graph) -> CodegenResult<Self> {
        let mut initializer = HashMap::new();
        for (value_id, tensor) in graph.initializer.iter() {
            let data = tensor.data.raw_vec();
            let name = &graph.values[*value_id].name;
            let data_id = jit.create_data(name, data)?;
            initializer.insert(value_id, (name, data_id));
        }

        let ptr_type = jit.module.target_config().pointer_type();
        let sig = {
            let sig = &mut jit.ctx.func.signature;
            sig.params.clear();
            sig.params.push(AbiParam::new(ptr_type));
            sig.params.push(AbiParam::new(ptr_type));
            jit.module.make_signature()
        };

        let func_id = jit
            .module
            .declare_function(&graph.name, Linkage::Export, &sig)
            .map_err(CodegenError::ModuleError)?;
        let mut translator = FunctionTranslator::new(jit)?;

        let mut id2value: HashMap<ValueId, Value> = HashMap::new();
        for (value_id, (name, _)) in initializer.iter() {
            let sym = translator
                .module
                .declare_data(name, Linkage::Export, true, false)
                .map_err(CodegenError::ModuleError)?;
            let local_id = translator
                .module
                .declare_data_in_func(sym, translator.builder.func);
            let value = translator.builder.ins().symbol_value(ptr_type, local_id);
            id2value.insert(**value_id, value);
        }

        let current_block = translator.builder.current_block().unwrap();
        let input_arg = translator.builder.block_params(current_block)[0];
        let output_arg = translator.builder.block_params(current_block)[1];
        for (arg, ids) in [(input_arg, &graph.inputs), (output_arg, &graph.outputs)] {
            let mut ptr = arg;
            for &value_id in ids.iter() {
                id2value.insert(value_id, ptr);
                let ty = graph
                    .get_resolved_tensor_type(value_id)
                    .ok_or(CodegenError::UnresolvedShape)?;
                let size = ty.mem_size();
                ptr = translator.builder.ins().iadd_imm(ptr, size as i64);
            }
        }

        Ok(Self {
            func_id,
            translator,
            graph,
            id2value,
        })
    }

    fn get_tensor_ptr(&self, id: ValueId) -> CodegenResult<TensorPtr> {
        let value = self
            .id2value
            .get(&id)
            .cloned()
            .ok_or(CodegenError::ValueNotFound(id))?;
        let ty = self
            .graph
            .get_resolved_tensor_type(id)
            .ok_or(CodegenError::UnresolvedShape)?
            .clone();
        Ok(TensorPtr { ptr: value, ty })
    }

    fn allocate_or_get_tensor(&mut self, id: ValueId) -> CodegenResult<TensorPtr> {
        if let Some(ptr) = self.id2value.get(&id) {
            // output?
            let ty = self
                .graph
                .get_resolved_tensor_type(id)
                .ok_or(CodegenError::UnresolvedShape)?
                .clone();
            Ok(TensorPtr { ptr: *ptr, ty })
        } else {
            let ty = self
                .graph
                .get_resolved_tensor_type(id)
                .ok_or(CodegenError::UnresolvedShape)?
                .clone();
            let size = ty.mem_size();
            let ptr = self.translator.allocate(size);
            self.id2value.insert(id, ptr);
            Ok(TensorPtr { ptr, ty })
        }
    }

    fn compile_node(&mut self, node: &Node) -> CodegenResult<()> {
        let inputs = node
            .inputs
            .iter()
            .map(|&id| self.get_tensor_ptr(id))
            .collect::<CodegenResult<Vec<_>>>()?;
        match node.op {
            Operator::Add => {
                if false {
                    let mut lhs = inputs[args::ADD_LHS].clone();
                    let mut rhs = inputs[args::ADD_RHS].clone();
                    let res = &self.allocate_or_get_tensor(node.outputs[0])?;
                    lhs.ty = lhs.ty.broadcast(&res.ty.dims);
                    rhs.ty = rhs.ty.broadcast(&res.ty.dims);
                    self.translator
                        .gen_nested_loop_binop(&lhs, &rhs, res, ElementwiseOp::Add);
                } else {
                    let lhs = &inputs[args::ADD_LHS];
                    let rhs = &inputs[args::ADD_RHS];
                    let res = &self.allocate_or_get_tensor(node.outputs[0])?;
                    let (lhs, casted) = if lhs.ty.is_broadcast_required(&res.ty.dims) {
                        let lhs = self.translator.gen_broadcast(res, lhs);
                        (lhs, true)
                    } else {
                        (lhs.clone(), false)
                    };
                    let rhs = if rhs.ty.is_broadcast_required(&res.ty.dims) {
                        let dst = if !casted {
                            res.clone()
                        } else {
                            let ty = res.ty.clone();
                            let ptr = self.translator.allocate(ty.mem_size());
                            TensorPtr { ptr, ty }
                        };
                        self.translator.gen_broadcast(&dst, rhs)
                    } else {
                        rhs.clone()
                    };

                    self.translator
                        .gen_single_loop_binop(&lhs, &rhs, res, ElementwiseOp::Add);
                }
            }
            Operator::ReLU => {
                let input = &inputs[args::RELU_DATA];
                let res = &self.allocate_or_get_tensor(node.outputs[0])?;
                self.translator
                    .gen_nested_loop_unaryop(input, res, ElementwiseOp::ReLU);
            }
            Operator::Reshape => {
                let input = &inputs[args::RESHAPE_DATA].ptr;
                let output_id = node.outputs[0];
                self.id2value.insert(output_id, *input);
            }
            Operator::MatMul => {
                let lhs = &inputs[args::MATMUL_LHS];
                let rhs = &inputs[args::MATMUL_RHS];
                if lhs.ty.dims.ndim() == 2 && rhs.ty.dims.ndim() == 2 {
                    let res = &self.allocate_or_get_tensor(node.outputs[0])?;
                    let shape = MatMulShape::new(&lhs.ty, &rhs.ty);
                    self.translator.gen_matmul_a_b(
                        res.ty.value_type(),
                        res.ptr,
                        lhs.ptr,
                        rhs.ptr,
                        &shape,
                    );
                } else {
                    todo!("MatMul");
                }
            }
            Operator::MatMulRightTransposed => {
                let lhs = &inputs[args::MATMUL_LHS];
                let rhs = &inputs[args::MATMUL_RHS];
                if lhs.ty.dims.ndim() == 2 && rhs.ty.dims.ndim() == 2 {
                    let res = &self.allocate_or_get_tensor(node.outputs[0])?;
                    let shape = MatMulShape::new(&lhs.ty, &rhs.ty.transpose());
                    let ty = TensorOperand::dynamic_vector_op_type(
                        res.ty.value_type(),
                        &self.translator.isa,
                    );
                    self.translator
                        .gen_matmul_a_tb(ty, res.ptr, lhs.ptr, rhs.ptr, &shape);
                } else {
                    todo!("MatMulRightTransposed");
                }
            }
            Operator::Conv(_) | Operator::MaxPool(_) | Operator::Transpose => unimplemented!(),
        }
        Ok(())
    }

    fn finalize(mut self) -> CodegenResult<FuncId> {
        self.translator.builder.ins().return_(&[]);
        self.translator.builder.finalize();
        Ok(self.func_id)
    }

    pub fn compile(jit: &mut JIT, graph: &Graph) -> CodegenResult<*const u8> {
        let mut compiler = GraphCompiler::new(jit, graph)?;
        for (_, node) in graph.nodes.iter() {
            compiler.compile_node(node)?;
        }
        let func_id = compiler.finalize()?;
        println!("{}", jit.ctx.func);
        jit.module
            .define_function(func_id, &mut jit.ctx)
            .map_err(CodegenError::ModuleError)?;
        jit.module.clear_context(&mut jit.ctx);
        jit.module.finalize_definitions().unwrap();
        let code = jit.module.get_finalized_function(func_id);
        Ok(code)
    }
}

#[derive(Debug, Clone)]
struct TensorPtr {
    ptr: Value,
    ty: ResolvedTensorType,
}

#[derive(Debug, Clone)]
struct TensorOperand {
    tensor: TensorPtr,
    op_type: Type, // for SIMD
}

impl TensorOperand {
    fn new(tensor: TensorPtr, isa: &OwnedTargetIsa) -> Self {
        let ty = tensor.ty.value_type();
        Self {
            tensor,
            op_type: Self::dynamic_vector_op_type(ty, isa),
        }
    }

    fn new_scalar(tensor: TensorPtr) -> Self {
        let op_type = tensor.ty.value_type();
        Self { tensor, op_type }
    }

    fn lane_count(&self) -> u32 {
        self.op_type.lane_count()
    }

    fn calc_op_type(ty: Type, max_lane_count: u32) -> Type {
        ty.by(max_lane_count).unwrap_or(ty)
    }

    fn dynamic_vector_op_type(ty: Type, isa: &OwnedTargetIsa) -> Type {
        let max_bytes = isa.dynamic_vector_bytes(ty);
        let max_lane_count = max_bytes / ty.bytes();
        println!("max_bytes={max_bytes}");
        Self::calc_op_type(ty, max_lane_count)
    }

    fn change_lane_count(&mut self, lane_count: u32) {
        self.op_type = Self::calc_op_type(self.tensor.ty.value_type(), lane_count);
    }
}

type UnaryOperand = TensorOperand;
type BinaryOperands = (TensorOperand, TensorOperand);

#[allow(dead_code)]
#[derive(Debug, Clone)]
enum ElementwiseOperands<'a> {
    Unary(&'a UnaryOperand),
    Binary(&'a BinaryOperands),
}

impl<'a> ElementwiseOperands<'a> {
    fn block_params_ty(&self, ptr_ty: Type) -> Vec<Type> {
        let len = 2 + match self {
            Self::Binary(_) => 2,
            Self::Unary(_) => 1,
        };
        let mut res = vec![ptr_ty; len];
        res[PARAMS_INDUCTION] = types::I64;
        res[PARAMS_DST] = ptr_ty;
        match self {
            Self::Binary(_) => {
                res[PARAMS_LHS] = ptr_ty;
                res[PARAMS_RHS] = ptr_ty;
            }
            Self::Unary(_) => {
                res[PARAMS_DATA] = ptr_ty;
            }
        };
        res
    }
}

#[derive(Debug, Clone)]
enum ElementwiseOp {
    Add(BinaryOperands),
    ReLU(UnaryOperand),
    Im2Col(UnaryOperand),
}

impl ElementwiseOp {
    fn operands(&self) -> ElementwiseOperands {
        match self {
            Self::Add(operands) => ElementwiseOperands::Binary(operands),
            Self::ReLU(operand) | Self::Im2Col(operand) => ElementwiseOperands::Unary(operand),
        }
    }

    fn set_operands(&mut self, params: &[Value]) {
        match self {
            Self::Add(operands) => {
                let lhs = params[PARAMS_LHS];
                let rhs = params[PARAMS_RHS];
                operands.0.tensor.ptr = lhs;
                operands.1.tensor.ptr = rhs;
            }
            Self::ReLU(operand) | Self::Im2Col(operand) => {
                let data = params[PARAMS_DATA];
                operand.tensor.ptr = data;
            }
        }
    }

    fn change_lane_count(&mut self, lane_count: u32) {
        match self {
            Self::Add(operands) => {
                operands.0.change_lane_count(lane_count);
                operands.1.change_lane_count(lane_count);
            }
            Self::ReLU(operand) | Self::Im2Col(operand) => {
                operand.change_lane_count(lane_count);
            }
        }
    }

    fn generate(&self, translator: &mut FunctionTranslator<'_>, offset: i32) -> Value {
        match self {
            Self::Add(operands) => {
                let (lhs, rhs) = operands;
                let lhs = translator.builder.ins().load(
                    lhs.op_type,
                    MemFlags::trusted(),
                    lhs.tensor.ptr,
                    offset,
                );
                let rhs = translator.builder.ins().load(
                    rhs.op_type,
                    MemFlags::trusted(),
                    rhs.tensor.ptr,
                    offset,
                );
                // TODO: integer add
                translator.builder.ins().fadd(lhs, rhs)
            }
            Self::ReLU(data) => {
                // TODO: type check
                let data_ty = data.op_type;
                let data = translator.builder.ins().load(
                    data.op_type,
                    MemFlags::trusted(),
                    data.tensor.ptr,
                    offset,
                );
                let zero = match data_ty {
                    types::F32 => translator.builder.ins().f32const(0.0),
                    types::F64 => translator.builder.ins().f64const(0.0),
                    _ => panic!("unsupported type"),
                };
                translator.builder.ins().fmax(data, zero)
            }
            Self::Im2Col(data) => translator.builder.ins().load(
                data.op_type,
                MemFlags::trusted(),
                data.tensor.ptr,
                offset,
            ),
        }
    }

    fn update_params(
        &self,
        params: &mut [Value],
        nest: usize,
        translator: &mut FunctionTranslator<'_>,
    ) {
        match self {
            Self::Add(operands) => {
                let (lhs, rhs) = operands;
                let l_add = lhs.tensor.ty.stride(nest) as i64 * lhs.op_type.bytes() as i64;
                let r_add = rhs.tensor.ty.stride(nest) as i64 * rhs.op_type.bytes() as i64;

                let lhs = params[PARAMS_LHS];
                let rhs = params[PARAMS_RHS];

                let lhs = translator.builder.ins().iadd_imm(lhs, l_add);
                let rhs = translator.builder.ins().iadd_imm(rhs, r_add);

                params[PARAMS_LHS] = lhs;
                params[PARAMS_RHS] = rhs;
            }
            Self::ReLU(data) | Self::Im2Col(data) => {
                let add = data.tensor.ty.stride(nest) as i64 * data.op_type.bytes() as i64;
                let data = params[PARAMS_DATA];
                let data = translator.builder.ins().iadd_imm(data, add);
                params[PARAMS_DATA] = data;
            }
        }
    }
}

const PARAMS_DST: usize = 0;
const PARAMS_INDUCTION: usize = 1;
const PARAMS_LHS: usize = 2;
const PARAMS_RHS: usize = 3;
const PARAMS_DATA: usize = 2;

impl ResolvedTensorType {
    fn value_bytes(&self) -> usize {
        self.value_type().bytes() as usize
    }

    pub fn mem_size(&self) -> usize {
        self.value_bytes() * self.dims.size()
    }
}

impl<'a> FunctionTranslator<'a> {
    pub fn new(jit: &'a mut JIT) -> CodegenResult<Self> {
        let ptr_ty = jit.module.target_config().pointer_type();
        let mut builder = FunctionBuilder::new(&mut jit.ctx.func, &mut jit.builder_ctx);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let malloc = {
            let mut sig = jit.module.make_signature();
            sig.params.push(AbiParam::new(types::I64));
            sig.returns.push(AbiParam::new(ptr_ty));
            let callee = jit
                .module
                .declare_function("malloc", Linkage::Import, &sig)
                .map_err(CodegenError::ModuleError)?;
            jit.module.declare_func_in_func(callee, builder.func)
        };

        let isa = jit.isa.clone();

        // let free = {
        //     let mut sig = jit.module.make_signature();
        //     sig.params.push(AbiParam::new(ptr_ty));
        //     let callee = jit.module
        //         .declare_function("free", Linkage::Import, &sig)
        //         .map_err(CodegenError::ModuleError)?;
        //     jit.module.declare_func_in_func(callee, builder.func)
        // };

        Ok(Self {
            builder,
            module: &mut jit.module,
            isa,
            ptr_ty,
            malloc,
            allocator: memory::Allocator::default(),
            value2fragment: HashMap::new(),
        })
    }

    // TODO: null check
    fn malloc(&mut self, size: Value) -> Value {
        let call = self.builder.ins().call(self.malloc, &[size]);
        self.builder.inst_results(call)[0]
    }

    pub fn allocate(&mut self, size: usize) -> Value {
        let fragment = self.allocator.allocate(size).unwrap_or({
            let allocated = size;
            let size = size.div_ceil(1024) * 1024;
            let ptr = {
                let size = self.builder.ins().iconst(types::I64, size as i64);
                self.malloc(size)
            };
            self.allocator.append_block(ptr, size, allocated)
        });

        let ptr = self
            .builder
            .ins()
            .iadd_imm(fragment.base, fragment.offset as i64);
        self.value2fragment.insert(ptr, fragment);
        ptr
    }

    pub fn free(&mut self, ptr: Value) -> CodegenResult<()> {
        let fragment = self
            .value2fragment
            .remove(&ptr)
            .ok_or(CodegenError::DeallocationError(ptr))?;
        self.allocator.deallocate(fragment);
        Ok(())
    }

    pub fn gen_call_ret(&mut self, name: &str, args: &[TypedValue], rt: Type) -> TypedValue {
        let mut sig = self.module.make_signature();
        for arg in args {
            sig.params.push(AbiParam::new(arg.ty));
        }
        sig.returns.push(AbiParam::new(rt));
        println!("name={:?} sig={:?}", name, sig);

        let callee = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .unwrap();
        let callee = self.module.declare_func_in_func(callee, self.builder.func);
        let args: Vec<Value> = args.iter().map(|arg| arg.value).collect();
        let call = self.builder.ins().call(callee, &args);
        let value = self.builder.inst_results(call)[0];
        TypedValue { ty: rt, value }
    }

    pub fn gen_global_data_addr(&mut self, name: &str) -> TypedValue {
        let sym = self
            .module
            .declare_data(name, Linkage::Export, true, false)
            .expect("problem declaring data object");
        let local_id = self.module.declare_data_in_func(sym, self.builder.func);

        let ty = self.module.target_config().pointer_type();
        let value = self.builder.ins().symbol_value(ty, local_id);
        TypedValue { ty, value }
    }

    // TODO: Check type
    pub fn gen_add(&mut self, lhs: TypedValue, rhs: TypedValue) -> TypedValue {
        let value = self.builder.ins().iadd(lhs.value, rhs.value);
        TypedValue { ty: lhs.ty, value }
    }

    pub fn gen_const(&mut self, imm: i64) -> TypedValue {
        let value = self.builder.ins().iconst(types::I64, imm);
        TypedValue {
            ty: types::I64,
            value,
        }
    }

    pub fn call_memcpy(&mut self, dst: Value, src: Value, len: Value) {
        self.builder
            .call_memcpy(self.module.target_config(), dst, src, len)
    }

    // Example: 3D tensor addition
    //
    // ```
    // lhs0 = pointer of lhs
    // rhs0 = pointer of rhs
    // res = pointer of res
    // i0 = 0;
    //
    // loop {
    //   i1 = 0;
    //   lhs1 = lhs0;
    //   rhs1 = rhs0;
    //   if (i0 == d0) break;
    //
    //   loop {
    //     i2 = 0;
    //     lhs2 = lhs1;
    //     rhs2 = rhs1;
    //     if (i1 == d1) break;
    //
    //     loop {
    //       if (i2 == d2) break;
    //
    //       *res = *lhs2 + *rhs2;
    //       i2++;
    //       lhs2 += l_stride2;
    //       rhs2 += r_stride2;
    //       res++;
    //     }  // end of loop i2
    //
    //     i1++;
    //     lhs1 += l_stride1;
    //     rhs1 += r_stride1;
    //   }  // end of loop i1
    //
    //   i0++;
    //   lhs0 += l_stride0;
    //   rhs0 += r_stride0;
    // }
    // ```
    //
    // Each variable is initialized in the `head`.
    // Each variable is updated in the `exit`.
    fn gen_nested_loop_rec(
        &mut self,
        op: &mut ElementwiseOp,
        res: &TensorPtr,
        nest: usize,
        header: ir::Block,
        next: ir::Block,
    ) {
        let exit = self.builder.create_block();
        for ty in op.operands().block_params_ty(self.ptr_ty) {
            self.builder.append_block_param(header, ty);
        }
        self.builder.append_block_param(exit, self.ptr_ty);
        let params = self.builder.block_params(header).to_vec();
        let ind = params[PARAMS_INDUCTION];
        let dst = params[PARAMS_DST];
        println!("nest={nest}");

        let mut params = if nest + 1 == res.ty.dims.ndim() {
            self.builder.switch_to_block(header);
            self.builder.ins().brif(ind, exit, &[dst], next, &[dst]);
            self.builder.switch_to_block(exit);

            op.set_operands(params.as_slice());
            let dst = self.builder.block_params(exit)[PARAMS_DST];
            let sum = op.generate(self, 0);
            self.builder.ins().store(MemFlags::trusted(), sum, dst, 0);
            let dst = self
                .builder
                .ins()
                .iadd_imm(dst, res.ty.value_bytes() as i64);
            let mut params = params;
            params[PARAMS_DST] = dst;
            params
        } else {
            let inner_loop = self.builder.create_block();
            self.builder.switch_to_block(header);
            let inner_ind = self
                .builder
                .ins()
                .iconst(types::I64, res.ty.dims[nest + 1] as i64);
            {
                let mut params = params.clone();
                params[PARAMS_INDUCTION] = inner_ind;
                self.builder
                    .ins()
                    .brif(ind, inner_loop, params.as_slice(), next, &[dst]);
            }

            self.gen_nested_loop_rec(op, res, nest + 1, inner_loop, exit);
            self.builder.switch_to_block(exit);
            let mut params = params;
            let dst = self.builder.block_params(exit)[0];
            params[PARAMS_DST] = dst;
            params
        };

        op.update_params(&mut params, nest, self);
        let ind = params[PARAMS_INDUCTION];
        let ind = self.builder.ins().iadd_imm(ind, -1);
        params[PARAMS_INDUCTION] = ind;
        self.builder.ins().jump(header, params.as_slice());
        self.builder.seal_block(header);
        self.builder.seal_block(exit);
    }

    fn gen_nested_loop(&mut self, op: &mut ElementwiseOp, res: &TensorPtr, params: &[Value]) {
        let header = self.builder.create_block();
        let exit = self.builder.create_block();
        self.builder.append_block_param(exit, self.ptr_ty); // dummy
        self.builder.ins().jump(header, params);
        self.gen_nested_loop_rec(op, res, 0, header, exit);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
    }

    pub fn gen_broadcast(&mut self, dst: &TensorPtr, src: &TensorPtr) -> TensorPtr {
        let mut src = src.clone();
        src.ty = src.ty.broadcast(&dst.ty.dims);
        self.gen_nested_loop_unaryop(&src, dst, ElementwiseOp::Im2Col);
        dst.clone()
    }

    pub fn gen_nested_loop_unaryop<T>(&mut self, input: &TensorPtr, res: &TensorPtr, op: T)
    where
        T: Fn(UnaryOperand) -> ElementwiseOp,
    {
        println!("dims={:?}", res.ty.dims);
        let ind = self.builder.ins().iconst(types::I64, res.ty.dims[0] as i64);
        let params = {
            let mut params = vec![ind; 3];
            params[PARAMS_DST] = res.ptr;
            params[PARAMS_DATA] = input.ptr;
            params[PARAMS_INDUCTION] = ind;
            params
        };
        let mut op = op(TensorOperand::new_scalar(input.clone()));
        self.gen_nested_loop(&mut op, res, params.as_slice());
    }

    pub fn gen_nested_loop_binop<'b, T>(
        &mut self,
        lhs: &'b TensorPtr,
        rhs: &'b TensorPtr,
        res: &TensorPtr,
        op: T,
    ) where
        T: Fn(BinaryOperands) -> ElementwiseOp,
    {
        let ind = self.builder.ins().iconst(types::I64, res.ty.dims[0] as i64);
        let params = {
            let mut params = vec![ind; 4];
            params[PARAMS_DST] = res.ptr;
            params[PARAMS_LHS] = lhs.ptr;
            params[PARAMS_RHS] = rhs.ptr;
            params[PARAMS_INDUCTION] = ind;
            params
        };
        let lhs = TensorOperand::new_scalar(lhs.clone());
        let rhs = TensorOperand::new_scalar(rhs.clone());
        let mut op = op((lhs, rhs));
        self.gen_nested_loop(&mut op, res, params.as_slice());
    }

    pub fn gen_single_loop_binop<'b, T>(
        &mut self,
        lhs: &'b TensorPtr,
        rhs: &'b TensorPtr,
        res: &TensorPtr,
        op: T,
    ) where
        T: Fn(BinaryOperands) -> ElementwiseOp,
    {
        let lhs = TensorOperand::new(lhs.clone(), &self.isa);
        let rhs = TensorOperand::new(rhs.clone(), &self.isa);
        let lane_count = lhs.lane_count() as i64;
        let trip_count = res.ty.dims.size() as i64;
        let main_trip_count = trip_count / lane_count;
        let rem_trip_count = trip_count - main_trip_count * lane_count;

        let ind = self.builder.ins().iconst(types::I64, main_trip_count);
        let header = self.builder.create_block();
        let body = self.builder.create_block();
        let exit = self.builder.create_block();
        let lhs_bytes = lhs.op_type.bytes() as i64;
        let rhs_bytes = rhs.op_type.bytes() as i64;
        let dst_bytes = res.ty.value_bytes() as i64 * lane_count;
        let lhs_ptr = lhs.tensor.ptr;
        let rhs_ptr = rhs.tensor.ptr;
        let mut op = op((lhs, rhs));

        for ty in op.operands().block_params_ty(self.ptr_ty) {
            self.builder.append_block_param(header, ty);
            self.builder.append_block_param(exit, ty);
        }

        let params = {
            let mut params = vec![ind; 4];
            params[PARAMS_DST] = res.ptr;
            params[PARAMS_LHS] = lhs_ptr;
            params[PARAMS_RHS] = rhs_ptr;
            params[PARAMS_INDUCTION] = ind;
            params
        };
        self.builder.ins().jump(header, params.as_slice());
        self.builder.switch_to_block(header);
        let params = self.builder.block_params(header).to_owned();
        let ind = params[PARAMS_INDUCTION];
        let lhs = params[PARAMS_LHS];
        let rhs = params[PARAMS_RHS];
        let dst = params[PARAMS_DST];
        self.builder.ins().brif(ind, body, &[], exit, &params);
        self.builder.switch_to_block(body);
        self.builder.seal_block(body);
        op.set_operands(&params);
        let sum = op.generate(self, 0);
        self.builder.ins().store(MemFlags::trusted(), sum, dst, 0);
        let ind = self.builder.ins().iadd_imm(ind, -1);
        let lhs = self.builder.ins().iadd_imm(lhs, lhs_bytes);
        let rhs = self.builder.ins().iadd_imm(rhs, rhs_bytes);
        let dst = self.builder.ins().iadd_imm(dst, dst_bytes);
        let params = {
            let mut params = vec![ind; 4];
            params[PARAMS_DST] = dst;
            params[PARAMS_LHS] = lhs;
            params[PARAMS_RHS] = rhs;
            params[PARAMS_INDUCTION] = ind;
            params
        };
        self.builder.ins().jump(header, params.as_slice());
        self.builder.seal_block(header);

        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
        op.change_lane_count(1);
        op.set_operands(self.builder.block_params(exit));
        let dst = self.builder.block_params(exit)[PARAMS_DST];
        for i in 0..rem_trip_count {
            let offset = i as i32 * res.ty.value_bytes() as i32;
            let sum = op.generate(self, offset);
            self.builder
                .ins()
                .store(MemFlags::trusted(), sum, dst, offset);
        }
    }

    fn gen_matmul_a_b(
        &mut self,
        ty: Type,
        dst: Value,
        lhs: Value,
        rhs: Value,
        shape: &MatMulShape,
    ) {
        let block_i0 = self.builder.create_block();
        let block_i1 = self.builder.create_block();
        let block_j0 = self.builder.create_block();
        let block_j1 = self.builder.create_block();
        let block_k0 = self.builder.create_block();
        let block_k1 = self.builder.create_block();
        let block_exit = self.builder.create_block();

        const INDUCTION: usize = 0;
        const DST: usize = 1;
        const LHS: usize = 2;
        const RHS: usize = 3;
        const ACC: usize = 4;

        let &MatMulShape {
            m: for_i,
            n: for_j,
            k: for_k,
        } = shape;

        for block in [block_i0, block_j0, block_k0] {
            self.builder.append_block_param(block, types::I64);
            self.builder.append_block_param(block, self.ptr_ty);
            self.builder.append_block_param(block, self.ptr_ty);
            self.builder.append_block_param(block, self.ptr_ty);
        }
        self.builder.append_block_param(block_k0, ty);
        self.builder.append_block_param(block_j1, ty);

        let ind_i = self.builder.ins().iconst(types::I64, for_i as i64);
        let params = {
            let mut params = vec![ind_i; 4];
            params[DST] = dst;
            params[LHS] = lhs;
            params[RHS] = rhs;
            params
        };
        self.builder.ins().jump(block_i0, params.as_slice());

        {
            self.builder.switch_to_block(block_i0);
            let ind_i = self.builder.block_params(block_i0)[INDUCTION];
            let ind_j = self.builder.ins().iconst(types::I64, for_j as i64);
            let mut params_j = self.builder.block_params(block_i0).to_vec();
            params_j[INDUCTION] = ind_j;
            self.builder
                .ins()
                .brif(ind_i, block_j0, params_j.as_slice(), block_exit, &[]);
        }

        {
            self.builder.switch_to_block(block_j0);
            let ind_j = self.builder.block_params(block_j0)[INDUCTION];
            let ind_k = self.builder.ins().iconst(types::I64, for_k as i64);
            let acc = match ty {
                types::F32 => self.builder.ins().f32const(0.0),
                types::F64 => self.builder.ins().f64const(0.0),
                _ => panic!("unsupported type"),
            };
            let mut params_k = self.builder.block_params(block_j0).to_vec();
            params_k[INDUCTION] = ind_k;
            params_k.push(acc);
            self.builder
                .ins()
                .brif(ind_j, block_k0, params_k.as_slice(), block_i1, &[]);
        }

        {
            self.builder.switch_to_block(block_k0);
            let ind_k = self.builder.block_params(block_k0)[INDUCTION];
            let acc = self.builder.block_params(block_k0)[ACC];
            self.builder
                .ins()
                .brif(ind_k, block_k1, &[], block_j1, &[acc]);
        }

        {
            self.builder.switch_to_block(block_k1);
            let lhs = self.builder.block_params(block_k0)[LHS];
            let rhs = self.builder.block_params(block_k0)[RHS];
            let acc = self.builder.block_params(block_k0)[ACC];
            let ind_k = self.builder.block_params(block_k0)[INDUCTION];

            let acc = {
                let lhs = self.builder.ins().load(ty, MemFlags::trusted(), lhs, 0);
                let rhs = self.builder.ins().load(ty, MemFlags::trusted(), rhs, 0);
                self.builder.ins().fma(lhs, rhs, acc)
            };
            let ind_k = self.builder.ins().iadd_imm(ind_k, -1);
            let lhs = self.builder.ins().iadd_imm(lhs, ty.bytes() as i64);
            let rhs = self
                .builder
                .ins()
                .iadd_imm(rhs, ty.bytes() as i64 * for_j as i64);
            let params = {
                let mut params = self.builder.block_params(block_k0).to_vec();
                params[INDUCTION] = ind_k;
                params[LHS] = lhs;
                params[RHS] = rhs;
                params[ACC] = acc;
                params
            };
            self.builder.ins().jump(block_k0, params.as_slice());
        }

        {
            self.builder.switch_to_block(block_j1);
            let acc = self.builder.block_params(block_j1)[0];
            let rhs = self.builder.block_params(block_j0)[RHS];
            let dst = self.builder.block_params(block_j0)[DST];
            let ind_j = self.builder.block_params(block_j0)[INDUCTION];
            self.builder.ins().store(MemFlags::trusted(), acc, dst, 0);
            let ind_j = self.builder.ins().iadd_imm(ind_j, -1);
            let rhs = self.builder.ins().iadd_imm(rhs, ty.bytes() as i64);
            let dst = self.builder.ins().iadd_imm(dst, ty.bytes() as i64);
            let params = {
                let mut params = self.builder.block_params(block_j0).to_vec();
                params[INDUCTION] = ind_j;
                params[RHS] = rhs;
                params[DST] = dst;
                params
            };
            self.builder.ins().jump(block_j0, params.as_slice());
        }

        {
            self.builder.switch_to_block(block_i1);
            let ind_i = self.builder.block_params(block_i0)[INDUCTION];
            let lhs = self.builder.block_params(block_i0)[LHS];
            let dst = self.builder.block_params(block_i0)[DST];
            let ind_i = self.builder.ins().iadd_imm(ind_i, -1);
            let lhs = self
                .builder
                .ins()
                .iadd_imm(lhs, ty.bytes() as i64 * for_k as i64);
            let dst = self
                .builder
                .ins()
                .iadd_imm(dst, ty.bytes() as i64 * for_j as i64);
            let params = {
                let mut params = self.builder.block_params(block_i0).to_vec();
                params[INDUCTION] = ind_i;
                params[LHS] = lhs;
                params[DST] = dst;
                params
            };
            self.builder.ins().jump(block_i0, params.as_slice());
        }

        self.builder.switch_to_block(block_exit);
        for block in [
            block_i0, block_i1, block_j0, block_j1, block_k0, block_k1, block_exit,
        ] {
            self.builder.seal_block(block);
        }
    }

    fn gen_matmul_a_tb(
        &mut self,
        ty: Type,
        dst: Value,
        lhs: Value,
        rhs: Value,
        shape: &MatMulShape,
    ) {
        let block_i0 = self.builder.create_block();
        let block_i1 = self.builder.create_block();
        let block_j0 = self.builder.create_block();
        let block_j1 = self.builder.create_block();
        let block_k0 = self.builder.create_block();
        let block_k1 = self.builder.create_block();
        let block_acc = self.builder.create_block();
        let block_exit = self.builder.create_block();

        let &MatMulShape {
            m: for_i,
            n: for_j,
            k: for_k,
        } = shape;
        const INDUCTION: usize = 0;
        const DST: usize = 1;
        const LHS: usize = 2;
        const RHS: usize = 3;
        const ACC: usize = 4;

        let lane_count = ty.lane_count() as usize;
        let main_trip_count = for_k / lane_count;
        let rem_trip_count = for_k - main_trip_count * lane_count;
        let for_k = main_trip_count;
        let zero = match ty.lane_type() {
            types::F32 => self.builder.ins().f32const(0.0),
            types::F64 => self.builder.ins().f64const(0.0),
            _ => panic!("unsupported type"),
        };

        for block in [block_i0, block_j0, block_k0] {
            self.builder.append_block_param(block, types::I64);
            self.builder.append_block_param(block, self.ptr_ty);
            self.builder.append_block_param(block, self.ptr_ty);
            self.builder.append_block_param(block, self.ptr_ty);
        }
        self.builder.append_block_param(block_k0, ty);
        self.builder.append_block_param(block_acc, ty);
        self.builder.append_block_param(block_j1, ty.lane_type());

        let ind_i = self.builder.ins().iconst(types::I64, for_i as i64);
        let params = {
            let mut params = vec![ind_i; 4];
            params[DST] = dst;
            params[LHS] = lhs;
            params[RHS] = rhs;
            params
        };
        self.builder.ins().jump(block_i0, params.as_slice());

        {
            self.builder.switch_to_block(block_i0);
            let ind_i = self.builder.block_params(block_i0)[INDUCTION];
            let ind_j = self.builder.ins().iconst(types::I64, for_j as i64);
            let mut params_j = self.builder.block_params(block_i0).to_vec();
            params_j[INDUCTION] = ind_j;
            self.builder
                .ins()
                .brif(ind_i, block_j0, params_j.as_slice(), block_exit, &[]);
        }

        {
            self.builder.switch_to_block(block_j0);
            let ind_j = self.builder.block_params(block_j0)[INDUCTION];
            let ind_k = self.builder.ins().iconst(types::I64, for_k as i64);
            let acc = self.builder.ins().splat(ty, zero);
            let mut params_k = self.builder.block_params(block_j0).to_vec();
            params_k[INDUCTION] = ind_k;
            params_k.push(acc);
            self.builder
                .ins()
                .brif(ind_j, block_k0, params_k.as_slice(), block_i1, &[]);
        }

        {
            self.builder.switch_to_block(block_k0);
            let ind_k = self.builder.block_params(block_k0)[INDUCTION];
            let acc = self.builder.block_params(block_k0)[ACC];
            self.builder
                .ins()
                .brif(ind_k, block_k1, &[], block_acc, &[acc]);
        }

        {
            self.builder.switch_to_block(block_k1);
            let lhs = self.builder.block_params(block_k0)[LHS];
            let rhs = self.builder.block_params(block_k0)[RHS];
            let acc = self.builder.block_params(block_k0)[ACC];
            let ind_k = self.builder.block_params(block_k0)[INDUCTION];

            let acc = {
                let lhs = self.builder.ins().load(ty, MemFlags::trusted(), lhs, 0);
                let rhs = self.builder.ins().load(ty, MemFlags::trusted(), rhs, 0);
                self.builder.ins().fma(lhs, rhs, acc)
            };
            let ind_k = self.builder.ins().iadd_imm(ind_k, -1);
            let lhs = self.builder.ins().iadd_imm(lhs, ty.bytes() as i64);
            let rhs = self.builder.ins().iadd_imm(rhs, ty.bytes() as i64);
            let params = {
                let mut params = self.builder.block_params(block_k0).to_vec();
                params[INDUCTION] = ind_k;
                params[LHS] = lhs;
                params[RHS] = rhs;
                params[ACC] = acc;
                params
            };
            self.builder.ins().jump(block_k0, params.as_slice());
        }

        // NOTE: The addition order isn't preserved
        {
            self.builder.switch_to_block(block_acc);
            let acc_v = self.builder.block_params(block_acc)[0];
            let mut acc = self.builder.ins().extractlane(acc_v, 0);
            for i in 1..lane_count {
                let tmp = self.builder.ins().extractlane(acc_v, i as u8);
                acc = self.builder.ins().fadd(acc, tmp);
            }
            for i in 0..rem_trip_count {
                let offset = i as i32 * ty.lane_type().bytes() as i32;
                let lhs = self.builder.block_params(block_k0)[LHS];
                let rhs = self.builder.block_params(block_k0)[RHS];
                let lhs = self
                    .builder
                    .ins()
                    .load(ty.lane_type(), MemFlags::trusted(), lhs, offset);
                let rhs = self
                    .builder
                    .ins()
                    .load(ty.lane_type(), MemFlags::trusted(), rhs, offset);
                acc = self.builder.ins().fma(lhs, rhs, acc);
            }
            self.builder.ins().jump(block_j1, &[acc]);
        }

        {
            self.builder.switch_to_block(block_j1);
            let acc = self.builder.block_params(block_j1)[0];
            let rhs = self.builder.block_params(block_j0)[RHS];
            let dst = self.builder.block_params(block_j0)[DST];
            let ind_j = self.builder.block_params(block_j0)[INDUCTION];
            self.builder.ins().store(MemFlags::trusted(), acc, dst, 0);
            let ind_j = self.builder.ins().iadd_imm(ind_j, -1);
            let rhs = self
                .builder
                .ins()
                .iadd_imm(rhs, ty.lane_type().bytes() as i64 * shape.k as i64);
            let dst = self
                .builder
                .ins()
                .iadd_imm(dst, ty.lane_type().bytes() as i64);
            let params = {
                let mut params = self.builder.block_params(block_j0).to_vec();
                params[INDUCTION] = ind_j;
                params[RHS] = rhs;
                params[DST] = dst;
                params
            };
            self.builder.ins().jump(block_j0, params.as_slice());
        }

        {
            self.builder.switch_to_block(block_i1);
            let ind_i = self.builder.block_params(block_i0)[INDUCTION];
            let lhs = self.builder.block_params(block_i0)[LHS];
            let dst = self.builder.block_params(block_i0)[DST];
            let ind_i = self.builder.ins().iadd_imm(ind_i, -1);
            let lhs = self
                .builder
                .ins()
                .iadd_imm(lhs, ty.lane_type().bytes() as i64 * shape.k as i64);
            let dst = self
                .builder
                .ins()
                .iadd_imm(dst, ty.lane_type().bytes() as i64 * for_j as i64);
            let params = {
                let mut params = self.builder.block_params(block_i0).to_vec();
                params[INDUCTION] = ind_i;
                params[LHS] = lhs;
                params[DST] = dst;
                params
            };
            self.builder.ins().jump(block_i0, params.as_slice());
        }

        self.builder.switch_to_block(block_exit);
        for block in [
            block_i0, block_i1, block_j0, block_j1, block_k0, block_k1, block_acc, block_exit,
        ] {
            self.builder.seal_block(block);
        }
    }
}

pub fn sample(jit: &mut JIT) -> std::io::Result<*const u8> {
    let literal = "literal";
    let contents = "junjiruW\0";
    jit.create_data(literal, contents.as_bytes().to_vec())
        .unwrap();
    let ptr_type = jit.module.target_config().pointer_type();
    jit.ctx.func.signature.returns.push(AbiParam::new(ptr_type));
    let id = jit
        .module
        .declare_function("hello", Linkage::Export, &jit.ctx.func.signature)
        .unwrap();

    let mut translator = FunctionTranslator::new(jit)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

    let size = 42;
    let data = translator.gen_global_data_addr(literal);
    let memory = translator.allocate(size);
    let seek = memory;
    let memlen = translator.gen_const(contents.len() as i64);
    let memlensub = translator.gen_const(contents.len() as i64 - 1);
    translator.call_memcpy(seek, data.value, memlen.value);
    let seek = translator.gen_add(
        TypedValue {
            value: seek,
            ty: ptr_type,
        },
        memlensub.clone(),
    );
    translator.call_memcpy(seek.value, data.value, memlen.value);
    let memory = TypedValue {
        ty: ptr_type,
        value: memory,
    };
    let res = translator.gen_call_ret("puts", &[memory.clone()], types::I64);
    translator
        .free(memory.value)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    translator.builder.ins().return_(&[res.value]);
    translator.builder.finalize();

    println!("{}", jit.ctx.func);

    jit.module.define_function(id, &mut jit.ctx).unwrap();
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions().unwrap();
    let code = jit.module.get_finalized_function(id);
    Ok(code)
}

pub fn sample2(jit: &mut JIT) -> std::io::Result<(*const u8, DataId)> {
    let input0: Tensor = ndarray::array!([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],])
        .try_into()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let input1: Tensor = ndarray::array!([[[2.0, 3.0], [4.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]],])
        .try_into()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let output = Tensor::zeros(input1.ty.elem_type.clone(), input1.ty.dims.clone());
    jit.create_data("input0", input0.data.raw_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    jit.create_data("input1", input1.data.raw_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let output_id = jit
        .create_data("output", output.data.raw_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

    jit.ctx.func.signature.params.clear();
    jit.ctx
        .func
        .signature
        .params
        .push(AbiParam::new(jit.module.target_config().pointer_type()));
    let sig = jit.module.make_signature();
    let id = jit
        .module
        .declare_function("hello2", Linkage::Export, &sig)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let mut translator = FunctionTranslator::new(jit)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

    let output_ty = input0.ty.clone();
    let input0_ptr = translator.gen_global_data_addr("input0").value;
    let input1_ptr = translator.gen_global_data_addr("input1").value;
    let output_ptr = translator.gen_global_data_addr("output").value;

    let input0 = TensorPtr {
        ptr: input0_ptr,
        ty: input0.ty,
    };
    let input1 = TensorPtr {
        ptr: input1_ptr,
        ty: input1.ty,
    };
    let output = TensorPtr {
        ptr: output_ptr,
        ty: output_ty,
    };
    translator.gen_nested_loop_binop(&input0, &input1, &output, ElementwiseOp::Add);

    translator.builder.ins().return_(&[]);
    translator.builder.finalize();

    println!("{}", jit.ctx.func);

    jit.module.define_function(id, &mut jit.ctx).unwrap();
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions().unwrap();
    let code = jit.module.get_finalized_function(id);
    println!("{:?}", id);
    Ok((code, output_id))
}

pub fn sample3(jit: &mut JIT) -> std::io::Result<(*const u8, DataId)> {
    let input: Tensor = ndarray::array!([[[1.0, -2.0], [3.0, 4.0]], [[-5.0, 6.0], [-7.0, -8.0]],])
        .try_into()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let output = Tensor::zeros(input.ty.elem_type.clone(), input.ty.dims.clone());
    jit.create_data("input", input.data.raw_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let output_id = jit
        .create_data("output", output.data.raw_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

    jit.ctx.func.signature.params.clear();
    jit.ctx
        .func
        .signature
        .params
        .push(AbiParam::new(jit.module.target_config().pointer_type()));
    let sig = jit.module.make_signature();
    let id = jit
        .module
        .declare_function("hello2", Linkage::Export, &sig)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let mut translator = FunctionTranslator::new(jit)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;

    let output_ty = input.ty.clone();
    let input_ptr = translator.gen_global_data_addr("input").value;
    let output_ptr = translator.gen_global_data_addr("output").value;

    let input = TensorPtr {
        ptr: input_ptr,
        ty: input.ty,
    };
    let output = TensorPtr {
        ptr: output_ptr,
        ty: output_ty,
    };
    translator.gen_nested_loop_unaryop(&input, &output, ElementwiseOp::ReLU);

    translator.builder.ins().return_(&[]);
    translator.builder.finalize();

    println!("{}", jit.ctx.func);

    jit.module.define_function(id, &mut jit.ctx).unwrap();
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions().unwrap();
    let code = jit.module.get_finalized_function(id);
    println!("{:?}", id);
    Ok((code, output_id))
}
