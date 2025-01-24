use my_onnx::codegen::session::Session;
use my_onnx::tensor::tensor::{Tensor, TensorData};
use std::env;
use std::io::{Error, Result};

fn main0() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    // 7
    let input: ndarray::Array<f32, _> = ndarray::array!([[
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3294, 0.7255, 0.6235, 0.5922, 0.2353,
            0.1412, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8706, 0.9961, 0.9961, 0.9961, 0.9961,
            0.9451, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.6667, 0.2039,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2627, 0.4471, 0.2824, 0.4471, 0.6392,
            0.8902, 0.9961, 0.8824, 0.9961, 0.9961, 0.9961, 0.9804, 0.8980, 0.9961, 0.9961, 0.5490,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0667, 0.2588, 0.0549, 0.2627, 0.2627, 0.2627, 0.2314, 0.0824, 0.9255, 0.9961, 0.4157,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3255, 0.9922, 0.8196, 0.0706,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0863, 0.9137, 1.0000, 0.3255, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5059, 0.9961, 0.9333, 0.1725, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2314, 0.9765, 0.9961, 0.2431, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5216, 0.9961, 0.7333, 0.0196, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0353, 0.8039, 0.9725, 0.2275, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.4941, 0.9961, 0.7137, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.2941, 0.9843, 0.9412, 0.2235, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0745, 0.8667, 0.9961, 0.6510, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0118, 0.7961, 0.9961, 0.8588, 0.1373, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.1490, 0.9961, 0.9961, 0.3020, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.1216, 0.8784, 0.9961, 0.4510, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.5216, 0.9961, 0.9961, 0.2039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2392,
            0.9490, 0.9961, 0.9961, 0.2039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4745,
            0.9961, 0.9961, 0.8588, 0.1569, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4745,
            0.9961, 0.8118, 0.0706, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
        [
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
            0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
        ],
    ],],);
    let input: Tensor = input
        .into_dyn()
        .try_into()
        .map_err(|e| Error::other(format!("{:?}", e)))?;
    use my_onnx::tensor::resolved_dimensions::ResolvedTensorDims;
    let session = Session::new(
        &args[1],
        Some(&[&ResolvedTensorDims::new(vec![1, 28, 28])]),
        100,
    )
    .map_err(|e| Error::other(format!("{:?}", e)))?;
    session.write_model("model.dot");
    let output = session
        .run(&[input])
        .map_err(|e| Error::other(format!("{:?}", e)))?;
    println!("{:?}", output);
    //println!("{:?}", output.unwrap()[0].data.raw_vec());
    Ok(())
}

fn main1() -> Result<()> {
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    use inkwell::module::Module;
    use inkwell::values::*;
    use inkwell::OptimizationLevel;

    type SumFunc = unsafe extern "C" fn(*const u8, *const u8, u64);

    struct CodeGen<'ctx> {
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,
        execution_engine: ExecutionEngine<'ctx>,
    }

    impl<'ctx> CodeGen<'ctx> {
        fn build_outlined(&self) -> Option<FunctionValue<'ctx>> {
            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
            let function_outlined = self.module.add_function(
                "pow2.outlined",
                self.context.void_type().fn_type(
                    &[
                        ptr_type.into(),
                        ptr_type.into(),
                        ptr_type.into(),
                        ptr_type.into(),
                        ptr_type.into(),
                        ptr_type.into(),
                    ],
                    false,
                ),
                None,
            );

            let i32_type = self.context.i32_type();
            let f32_type = self.context.f32_type();
            let entry = self.context.append_basic_block(function_outlined, "entry");
            let body = self.context.append_basic_block(function_outlined, "body");
            let exit = self.context.append_basic_block(function_outlined, "exit");

            self.builder.position_at_end(entry);

            let len = function_outlined.get_nth_param(2)?.into_pointer_value();
            let len = self
                .builder
                .build_load(i32_type, len, "len")
                .unwrap()
                .into_int_value();
            let dst = function_outlined.get_nth_param(3)?.into_pointer_value();
            let dst = self
                .builder
                .build_load(ptr_type, dst, "dst")
                .unwrap()
                .into_pointer_value();
            let src = function_outlined.get_nth_param(4)?.into_pointer_value();
            let src = self
                .builder
                .build_load(ptr_type, src, "src")
                .unwrap()
                .into_pointer_value();
            let cond = self
                .builder
                .build_int_compare(
                    inkwell::IntPredicate::SLT,
                    len,
                    i32_type.const_int(0, false),
                    "cond",
                )
                .unwrap();
            let _ = self
                .builder
                .build_conditional_branch(cond, exit, body)
                .unwrap();

            self.builder.position_at_end(body);
            let ind = self.builder.build_phi(i32_type, "ind").unwrap();
            let ind_int = ind.as_basic_value().into_int_value();
            let gep = unsafe {
                self.builder
                    .build_in_bounds_gep(f32_type, src, &[ind_int], "gep.src")
                    .unwrap()
            };
            let val = self
                .builder
                .build_load(f32_type, gep, "val")
                .unwrap()
                .into_float_value();
            let val = self.builder.build_float_mul(val, val, "val2").unwrap();
            let gep = unsafe {
                self.builder
                    .build_in_bounds_gep(f32_type, dst, &[ind_int], "gep.dst")
                    .unwrap()
            };
            self.builder.build_store(gep, val).unwrap();
            let ind_next = self
                .builder
                .build_int_add(ind_int, i32_type.const_int(1, false), "ind.next")
                .unwrap();
            let cond = self
                .builder
                .build_int_compare(inkwell::IntPredicate::SLT, ind_next, len, "cond")
                .unwrap();
            let _ = self
                .builder
                .build_conditional_branch(cond, body, exit)
                .unwrap();
            ind.add_incoming(&[(&i32_type.const_int(0, false), entry), (&ind_next, body)]);

            self.builder.position_at_end(exit);
            let printf = i32_type.fn_type(&[ptr_type.into()], true);
            let printf = self.module.add_function(
                "printf",
                printf,
                Some(inkwell::module::Linkage::External),
            );
            let omp_get_thread_num = i32_type.fn_type(&[], false);
            let omp_get_thread_num = self.module.add_function(
                "omp_get_thread_num",
                omp_get_thread_num,
                Some(inkwell::module::Linkage::External),
            );
            let printf_fmt = unsafe {
                self.builder
                    .build_global_string("TID=%d\n", "printf_fmt")
                    .unwrap()
            };
            let tid = self
                .builder
                .build_call(omp_get_thread_num, &[], "tid")
                .unwrap()
                .try_as_basic_value()
                .left()
                .unwrap()
                .into_int_value();
            let _ = self
                .builder
                .build_call(
                    printf,
                    &[printf_fmt.as_pointer_value().into(), tid.into()],
                    "",
                )
                .unwrap();
            self.builder.build_return(None).unwrap();

            Some(function_outlined)
        }

        fn jit_compile_sum(&self) -> Option<()> {
            let i32_type = self.context.i32_type();
            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
            let void_type = self.context.void_type();
            let fn_type =
                void_type.fn_type(&[ptr_type.into(), ptr_type.into(), i32_type.into()], false);
            let function = self.module.add_function("pow2", fn_type, None);

            let function_outlined = self.build_outlined()?;
            let entry = self.context.append_basic_block(function, "entry");

            self.builder.position_at_end(entry);
            let len_alloca = self.builder.build_alloca(i32_type, "len").unwrap();
            let dst_alloca = self.builder.build_alloca(ptr_type, "dst").unwrap();
            let src_alloca = self.builder.build_alloca(ptr_type, "src").unwrap();
            self.builder
                .build_store(len_alloca, function.get_nth_param(2)?.into_int_value())
                .unwrap();
            self.builder
                .build_store(dst_alloca, function.get_nth_param(0)?.into_pointer_value())
                .unwrap();
            self.builder
                .build_store(src_alloca, function.get_nth_param(1)?.into_pointer_value())
                .unwrap();
            let kmpc_fork_call =
                void_type.fn_type(&[ptr_type.into(), i32_type.into(), ptr_type.into()], true);
            let kmpc_fork_call = self.module.add_function(
                "__kmpc_fork_call",
                kmpc_fork_call,
                Some(inkwell::module::Linkage::External),
            );
            let dummy = unsafe {
                self.builder
                    .build_global_string("wn;unknown;0;0;;", "dummy")
                    .unwrap()
            };
            let ident = {
                let ident = self.context.const_struct(
                    &[
                        i32_type.const_int(0, false).into(),
                        i32_type.const_int(0, false).into(),
                        i32_type.const_int(0, false).into(),
                        i32_type.const_int(0, false).into(),
                        dummy.as_pointer_value().into(),
                    ],
                    false,
                );
                let ident_ty = self.context.struct_type(
                    &[
                        i32_type.into(),
                        i32_type.into(),
                        i32_type.into(),
                        i32_type.into(),
                        ptr_type.into(),
                    ],
                    false,
                );
                let gv = self.module.add_global(ident_ty, None, "ident");
                gv.set_initializer(&ident);
                gv
            };
            self.builder
                .build_call(
                    kmpc_fork_call,
                    &[
                        ident.as_pointer_value().into(),
                        i32_type.const_int(3, false).into(),
                        function_outlined
                            .as_global_value()
                            .as_pointer_value()
                            .into(),
                        len_alloca.into(),
                        dst_alloca.into(),
                        src_alloca.into(),
                    ],
                    "",
                )
                .unwrap();
            self.builder.build_return(None).unwrap();

            // let arr = {
            //     let gv = self.module.add_global(i64_type.array_type(3), None, "arr");
            //     let v0 = i64_type.const_int(3, false);
            //     let v1 = i64_type.const_int(1, false);
            //     let v2 = i64_type.const_int(4, false);
            //     let arr = i64_type.const_array(&[v0, v1, v2]);
            //     gv.set_initializer(&arr);
            //     gv
            // };
            // let _ = unsafe { self.builder.build_global_string("Hello, World!", "hello_world").unwrap() };

            Some(())
        }
    }

    use inkwell::targets::*;
    Target::initialize_native(&InitializationConfig::default()).unwrap();
    let target_triple = TargetMachine::get_default_triple();
    let cpu = TargetMachine::get_host_cpu_name().to_string();
    let features = TargetMachine::get_host_cpu_features().to_string();
    let target_machine = Target::from_triple(&target_triple)
        .unwrap()
        .create_target_machine(
            &target_triple,
            &cpu,
            &features,
            OptimizationLevel::Aggressive,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .unwrap();

    let context = Context::create();
    let module = context.create_module("pow2");
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::Default)
        .map_err(|e| Error::other(format!("{:?}", e)))?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };
    codegen.jit_compile_sum();
    let passes: &[&str] = &[
        "instcombine",
        "reassociate",
        "gvn",
        "simplifycfg",
        // "basic-aa",
        "mem2reg",
        "loop-vectorize",
        "slp-vectorizer",
    ];

    codegen
        .module
        .run_passes(
            passes.join(",").as_str(),
            &target_machine,
            inkwell::passes::PassBuilderOptions::create(),
        )
        .unwrap();

    codegen.module.print_to_stderr();

    target_machine
        .write_to_file(
            &codegen.module,
            inkwell::targets::FileType::Object,
            "out.o".as_ref(),
        )
        .unwrap();

    let pow2: JitFunction<SumFunc> = unsafe { codegen.execution_engine.get_function("pow2").ok() }
        .ok_or(Error::other("Unable to JIT compile `sum` function"))?;

    codegen.execution_engine.get_function_value("pow2").unwrap();

    let mut src = Vec::new();
    for i in 0..100 {
        src.push((i + 2) as f32);
    }
    let mut dst = vec![0f32; src.len()];

    unsafe {
        pow2.call(
            dst.as_mut_ptr() as *const u8,
            src.as_ptr() as *const u8,
            src.len() as u64,
        );
        println!("src={:?}", src);
        println!("dst={:?}", dst);
    }

    Ok(())
}

use image::{ImageReader, ImageResult};
use std::path::PathBuf;

fn main_resnet_input() -> ImageResult<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let img = dir.join("download/imagenet-sample-images/n01440764_tench.JPEG");
    let img = ImageReader::open(img)?.decode()?;
    let img = img.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);
    img.save("download/sample.jpg")
}

fn main_resnet_sample() -> ImageResult<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let img = dir.join("download/sample.jpg");
    let img = ImageReader::open(img)?.decode()?;
    let img = img.to_rgb8();
    for h in 0..10 {
        for w in 0..10 {
            println!("{:?}", img.get_pixel(w, h));
        }
    }
    Ok(())
}

enum Select {
    MainResnetInput,
    MainResnetSample,
    MainRunResnet,
    Main0,
    Main1,
}
use ndarray::Array;

fn main_run_resnet() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let img = dir.join("download/sample.jpg");
    let img = ImageReader::open(img)
        .map_err(|e| Error::other(format!("{:?}", e)))?
        .decode()
        .map_err(|e| Error::other(format!("{:?}", e)))?
        .to_rgb8();
    for i in 0..10 {
        println!(
            "img[0][{i}].R={:?}",
            (img.get_pixel(0, i)[0] as f32) / 255.0
        );
        println!(
            "img[0][{i}].G={:?}",
            (img.get_pixel(0, i)[1] as f32) / 255.0
        );
        println!(
            "img[0][{i}].B={:?}",
            (img.get_pixel(0, i)[2] as f32) / 255.0
        );
    }
    let input: Tensor = Array::from_shape_fn((1, 3, 224, 224), |(_, c, h, w)| {
        img.get_pixel(h as u32, w as u32)[c] as f32 / 255.0
    })
    .into_dyn()
    .try_into()
    .map_err(|e| Error::other(format!("{:?}", e)))?;
    // println!("{:?}", input);
    let session = Session::new(
        //dir.join("models/resnet18-v2-7.onnx"),
        &args[1],
        Some(&[&input.ty.dims]),
        100,
    )
    .map_err(|e| Error::other(format!("{:?}", e)))?;
    session.write_model("model.dot");
    let output = session
        .run(&[input])
        .map_err(|e| Error::other(format!("{:?}", e)))?;
    let output = match output[0].data {
        TensorData::F32(ref vec) => vec.clone(),
        _ => return Err(Error::other("Invalid output data type")),
    };
    // println!("{:?}", vec);
    let output = Array::from_vec(output)
        .into_shape_with_order(1000)
        // .into_shape_with_order((1, 3, 224, 224))
        // .into_shape_with_order((1, 64, 56, 56))
        .unwrap()
        // .index_axis(Axis(0), 0)
        // .index_axis(Axis(0), 0)
        // .index_axis(Axis(0), 0)
        .to_vec();
    println!("{:?}", output);
    Ok(())
}

fn main_(select: Select) -> Result<()> {
    match select {
        Select::MainResnetInput => {
            main_resnet_input().map_err(|e| Error::other(format!("{:?}", e)))
        }
        Select::MainResnetSample => {
            main_resnet_sample().map_err(|e| Error::other(format!("{:?}", e)))
        }
        Select::MainRunResnet => main_run_resnet(),
        Select::Main0 => main0(),
        Select::Main1 => main1(),
    }
}

fn main() -> Result<()> {
    if true {
        main_(Select::Main0)?;
        // main_(Select::MainRunResnet)?;
    } else {
        // main_(Select::MainResnetInput)?;
        // main_(Select::MainResnetSample)?;
        // main_(Select::Main0)?;
    }
    Ok(())
}
