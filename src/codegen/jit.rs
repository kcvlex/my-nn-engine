use crate::codegen::memory;
use crate::tensor::{
    resolved_dimensions::ResolvedTensorDims,
    tensor::{DataType, ResolvedTensorType, Tensor, TensorData},
};
use cranelift::codegen::ir;
use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, DataId, Linkage, Module, ModuleError};
use std::collections::HashMap;
use std::slice;

#[derive(Debug)]
pub enum CodegenError {
    ModuleError(ModuleError),
    DeallocationError(Value),
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
        let builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let module = JITModule::new(builder);
        Self {
            builder_ctx: FunctionBuilderContext::new(),
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

struct FunctionTranslator<'a> {
    builder: FunctionBuilder<'a>,
    module: &'a mut JITModule,
    ptr_ty: Type,
    malloc: ir::FuncRef,
    allocator: memory::Allocator,
    value2fragment: HashMap<Value, memory::Fragment>,
}

/*
impl Tensor {
    fn elem_type(&self) -> Type {
        match self.ty.elem_type {
            tensor::DataType::F32 => types::F32,
            tensor::DataType::F64 => types::F64,
        }
    }
}
*/

#[derive(Debug, Clone)]
struct TensorPtr<'a> {
    ptr: Value,
    ty: &'a ResolvedTensorType,
}

#[derive(Debug, Clone)]
struct UnaryOperand<'a>(&'a ResolvedTensorType);
#[derive(Debug, Clone)]
struct BinaryOperands<'a>(&'a ResolvedTensorType, &'a ResolvedTensorType);

#[derive(Debug, Clone)]
enum ElementwiseOperands<'a> {
    Unary(UnaryOperand<'a>),
    Binary(BinaryOperands<'a>),
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
            },
            Self::Unary(_) => {
                res[PARAMS_DATA] = ptr_ty;
            },
        };
        res
    }
}

#[derive(Debug, Clone)]
enum ElementwiseOp<'a> {
    Add(BinaryOperands<'a>),
    Add2(BinaryOperands<'a>),  // for debug
    ReLU(UnaryOperand<'a>),
}

impl<'a> ElementwiseOp<'a> {
    fn operands(&self) -> ElementwiseOperands<'a> {
        match self {
            ElementwiseOp::Add(operands) | ElementwiseOp::Add2(operands) => ElementwiseOperands::Binary(operands.clone()),
            ElementwiseOp::ReLU(operand) => ElementwiseOperands::Unary(operand.clone()),
        }
    }

    fn generate(&self, params: &[Value], translator: & mut FunctionTranslator<'_>) -> Value {
        match self {
            Self::Add(operands) | Self::Add2(operands) => {
                let BinaryOperands(lhs, rhs) = operands;
                let lhs_ty = lhs.value_type();
                let rhs_ty = rhs.value_type();
                let lhs = params[PARAMS_LHS];
                let rhs = params[PARAMS_RHS];
                let lhs = translator
                    .builder
                    .ins()
                    .load(lhs_ty, MemFlags::trusted(), lhs, 0);
                let rhs = translator
                    .builder
                    .ins()
                    .load(rhs_ty, MemFlags::trusted(), rhs, 0);
                translator.builder.ins().fadd(lhs, rhs)
            },
            _ => todo!(),
        }
    }

    fn update_params(&self, params: &mut [Value], nest: usize, translator: &mut FunctionTranslator<'_>) {
        match self {
            Self::Add(operands) | Self::Add2(operands) => {
                let BinaryOperands(lhs, rhs) = operands;
                let l_add = lhs.stride(nest) as i64 * lhs.value_bytes() as i64;
                let r_add = rhs.stride(nest) as i64 * rhs.value_bytes() as i64;

                let lhs = params[PARAMS_LHS];
                let rhs = params[PARAMS_RHS];

                let lhs = translator.builder.ins().iadd_imm(lhs, l_add);
                let rhs = translator.builder.ins().iadd_imm(rhs, r_add);

                params[PARAMS_LHS] = lhs;
                params[PARAMS_RHS] = rhs;
            },
            _ => todo!(),
        }
    }
}

const PARAMS_DST: usize = 0;
const PARAMS_INDUCTION: usize = 1;
const PARAMS_LHS: usize = 2;
const PARAMS_RHS: usize = 3;
const PARAMS_DATA: usize = 2;

struct LoopGenerator<'a> {
    translator: &'a mut FunctionTranslator<'a>,
}

impl ResolvedTensorType {
    fn value_bytes(&self) -> usize {
        self.value_type().bytes() as usize
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
    fn gen_simple_add_rec(
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
        // let lhs = params[LoopVar::LHS as usize];
        // let rhs = params[LoopVar::RHS as usize];
        let ind = params[PARAMS_INDUCTION];
        let dst = params[PARAMS_DST];
        println!("nest={nest}");

        let mut params = if nest + 1 == res.ty.dims.ndim() {
            self.builder.switch_to_block(header);
            self.builder.ins().brif(ind, exit, &[dst], next, &[dst]);
            self.builder.switch_to_block(exit);

            let sum = op.generate(params.as_slice(), self);
            self.builder.ins().store(MemFlags::trusted(), sum, dst, 0);
            let dst = self.builder.block_params(exit)[0];
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

            self.gen_simple_add_rec(op, res, nest + 1, inner_loop, exit);
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

    pub fn gen_simple_add(&mut self, lhs: &TensorPtr, rhs: &TensorPtr, res: &TensorPtr) {
        let ind = self.builder.ins().iconst(types::I64, res.ty.dims[0] as i64);
        let params = vec![lhs.ptr, rhs.ptr, res.ptr, ind];
        let mut op = ElementwiseOp::Add(BinaryOperands(lhs.ty, rhs.ty));
        let header = self.builder.create_block();
        let exit = self.builder.create_block();
        self.builder.append_block_param(exit, self.ptr_ty); // dummy
        self.builder.ins().jump(header, params.as_slice());
        self.gen_simple_add_rec(&mut op, res, 0, header, exit);
        self.builder.switch_to_block(exit);
        self.builder.seal_block(exit);
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
    println!("{:?}", id);
    Ok(code)
}

pub fn sample2(jit: &mut JIT) -> std::io::Result<(*const u8, DataId)> {
    let dim = ResolvedTensorDims::new(vec![2, 2, 2]);
    let input0 = Tensor::new(
        dim.clone(),
        TensorData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let input1 = Tensor::new(
        dim.clone(),
        TensorData::F32(vec![2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
    )
    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    jit.create_data("input0", input0.raw_data().to_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    jit.create_data("input1", input1.raw_data().to_vec())
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("{:?}", e)))?;
    let output_id = jit
        .create_data("output", vec![0; dim.size() * types::F32.bytes() as usize])
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
        ty: &input0.ty,
    };
    let input1 = TensorPtr {
        ptr: input1_ptr,
        ty: &input1.ty,
    };
    let output = TensorPtr {
        ptr: output_ptr,
        ty: &output_ty,
    };
    translator.gen_simple_add(&input0, &input1, &output);

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
