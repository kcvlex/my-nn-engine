use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{DataDescription, Linkage, Module};
use std::slice;

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
    pub fn create_data(&mut self, name: &str, contents: Vec<u8>) -> Result<&[u8], String> {
        self.data_description.define(contents.into_boxed_slice());
        let id = self
            .module
            .declare_data(name, Linkage::Export, true, false)
            .map_err(|e| e.to_string())?;

        self.module
            .define_data(id, &self.data_description)
            .map_err(|e| e.to_string())?;
        self.data_description.clear();
        self.module.finalize_definitions().unwrap();
        let buffer = self.module.get_finalized_data(id);
        // TODO: Can we move the unsafe into cranelift?
        Ok(unsafe { slice::from_raw_parts(buffer.0, buffer.1) })
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
}

impl<'a> FunctionTranslator<'a> {
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

    pub fn gen_call_void(&mut self, name: &str, args: &[TypedValue]) {
        let mut sig = self.module.make_signature();
        for arg in args {
            sig.params.push(AbiParam::new(arg.ty));
        }

        let callee = self
            .module
            .declare_function(name, Linkage::Import, &sig)
            .unwrap();
        println!("{:?}", callee);
        let callee = self.module.declare_func_in_func(callee, self.builder.func);
        println!("{:?}", callee);
        let args: Vec<Value> = args.iter().map(|arg| arg.value).collect();
        println!("{:?}", args);
        self.builder.ins().call(callee, &args);
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
}

pub fn sample(jit: &mut JIT) -> std::io::Result<*const u8> {
    let literal = "literal";
    let contents = "junjiruW\0";
    jit.create_data(literal, contents.as_bytes().to_vec())
        .unwrap();
    let ptr_type = jit.module.target_config().pointer_type();
    jit.ctx.func.signature.returns.push(AbiParam::new(ptr_type));

    let mut builder = FunctionBuilder::new(&mut jit.ctx.func, &mut jit.builder_ctx);
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);
    builder.seal_block(entry_block);
    let mut translator = FunctionTranslator {
        builder,
        module: &mut jit.module,
    };

    let size = translator.gen_const(42);
    let data = translator.gen_global_data_addr(literal);
    let memory = translator.gen_call_ret("malloc", &[size.clone()], ptr_type);
    let seek = memory.clone();
    let memlen = translator.gen_const(contents.len() as i64);
    let memlensub = translator.gen_const(contents.len() as i64 - 1);
    translator.gen_call_ret(
        "memcpy",
        &[seek.clone(), data.clone(), memlensub.clone()],
        ptr_type,
    );
    let seek = translator.gen_add(seek, memlensub.clone());
    translator.gen_call_ret(
        "memcpy",
        &[seek.clone(), data.clone(), memlen.clone()],
        ptr_type,
    );
    let res = translator.gen_call_ret("puts", &[memory.clone()], types::I64);
    translator.gen_call_void("free", &[memory.clone()]);
    translator.builder.ins().return_(&[res.value]);
    translator.builder.finalize();

    println!("{}", jit.ctx.func);

    let id = jit
        .module
        .declare_function("hello", Linkage::Export, &jit.ctx.func.signature)
        .unwrap();
    jit.module.define_function(id, &mut jit.ctx).unwrap();
    jit.module.clear_context(&mut jit.ctx);
    jit.module.finalize_definitions().unwrap();
    let code = jit.module.get_finalized_function(id);
    println!("{:?}", id);
    Ok(code)
}
