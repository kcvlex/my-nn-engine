use my_onnx::codegen::session::Session;
use my_onnx::model::Model;
use my_onnx::optimize::{matmul_a_tb::MatMulAxTB, optimizer::Optimizer};
use my_onnx::tensor::tensor::Tensor;
use std::env;
use std::fs::File;
use std::io::{Error, Result, Write};

fn main_cranelift() -> Result<()> {
    let args: Vec<_> = env::args().collect();
    let mut optimizer = Optimizer::new(String::from("test pass"));
    optimizer.passes.push(Box::new(MatMulAxTB::default()));
    if false {
        let mut model =
            Model::load_from_path(&args[1]).map_err(|e| Error::other(format!("{:?}", e)))?;
        model
            .graph
            .infer()
            .map_err(|e| Error::other(format!("{:?}", e)))?;
        optimizer.run(&mut model.graph);
        let file = File::options()
            .truncate(true)
            .create(true)
            .write(true)
            .open("graph.dot")?;
        let mut writer = std::io::BufWriter::new(file);
        writer.write_all(model.graph.to_dot().as_bytes())?;
    } else {
        let session =
            Session::new(&args[1], optimizer).map_err(|e| Error::other(format!("{:?}", e)))?;

        {
            // 7
            let input: ndarray::Array<f32, _> = ndarray::array!([[
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3294, 0.7255, 0.6235, 0.5922,
                    0.2353, 0.1412, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.8706, 0.9961, 0.9961, 0.9961,
                    0.9961, 0.9451, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765, 0.7765,
                    0.6667, 0.2039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2627, 0.4471, 0.2824, 0.4471,
                    0.6392, 0.8902, 0.9961, 0.8824, 0.9961, 0.9961, 0.9961, 0.9804, 0.8980, 0.9961,
                    0.9961, 0.5490, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0667, 0.2588, 0.0549, 0.2627, 0.2627, 0.2627, 0.2314, 0.0824, 0.9255,
                    0.9961, 0.4157, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.3255, 0.9922,
                    0.8196, 0.0706, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0863, 0.9137, 1.0000,
                    0.3255, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5059, 0.9961, 0.9333,
                    0.1725, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2314, 0.9765, 0.9961, 0.2431,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5216, 0.9961, 0.7333, 0.0196,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0353, 0.8039, 0.9725, 0.2275, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4941, 0.9961, 0.7137, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.2941, 0.9843, 0.9412, 0.2235, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0745, 0.8667, 0.9961, 0.6510, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0118, 0.7961, 0.9961, 0.8588, 0.1373, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.1490, 0.9961, 0.9961, 0.3020, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.1216, 0.8784, 0.9961, 0.4510, 0.0039, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.5216, 0.9961, 0.9961, 0.2039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.2392, 0.9490, 0.9961, 0.9961, 0.2039, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.4745, 0.9961, 0.9961, 0.8588, 0.1569, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.4745, 0.9961, 0.8118, 0.0706, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
                [
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000
                ],
            ],],);
            let input: Tensor = input
                .try_into()
                .map_err(|e| Error::other(format!("{:?}", e)))?;
            let output = session.run(&[input]);
            println!("{:?}", output);
            //println!("{:?}", output.unwrap()[0].data.raw_vec());
        }
    }
    Ok(())
}

fn main_inkwell() -> Result<()> {
    use inkwell::attributes::*;
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    use inkwell::module::Module;
    use inkwell::types::*;
    use inkwell::values::*;
    use inkwell::OptimizationLevel;

    /// Convenience type alias for the `sum` function.
    ///
    /// Calling this is innately `unsafe` because there's no guarantee it doesn't
    /// do `unsafe` operations internally.
    type SumFunc = unsafe extern "C" fn(*const u8, *const u8, u64);

    struct CodeGen<'ctx> {
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,
        execution_engine: ExecutionEngine<'ctx>,
    }

    impl<'ctx> CodeGen<'ctx> {
        fn jit_compile_sum(&self) -> Option<()> {
            let i64_type = self.context.i64_type();
            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
            let f32_type = self.context.f32_type();
            let fn_type =
                i64_type.fn_type(&[ptr_type.into(), ptr_type.into(), i64_type.into()], false);
            let function = self.module.add_function("pow2", fn_type, None);

            let arr = {
                let gv = self.module.add_global(i64_type.array_type(3), None, "arr");
                let v0 = i64_type.const_int(3, false);
                let v1 = i64_type.const_int(1, false);
                let v2 = i64_type.const_int(4, false);
                let arr = i64_type.const_array(&[v0, v1, v2]);
                gv.set_initializer(&arr);
                gv
            };
            // let _ = unsafe { self.builder.build_global_string("Hello, World!", "hello_world").unwrap() };

            let entry = self.context.append_basic_block(function, "entry");
            let body = self.context.append_basic_block(function, "body");
            let exit = self.context.append_basic_block(function, "exit");

            // let vec_ty = f32_type;
            // let vscale_i64 = inkwell::intrinsics::Intrinsic::find("llvm.vscale.i64").unwrap();
            // let vscale_i64 = vscale_i64.get_declaration(&self.module, &[]).unwrap();

            self.builder.position_at_end(entry);
            // let vscale = {
            //     let call = self.builder.build_call(vscale_i64, &[], "vscale").unwrap();
            //     call.set_tail_call(true);
            //     call.try_as_basic_value().left().unwrap().into_int_value()
            // };
            let len = function.get_nth_param(2)?.into_int_value();
            let _ = self.builder.build_unconditional_branch(body).unwrap();

            self.builder.position_at_end(body);
            let ind = self.builder.build_phi(i64_type, "ind").unwrap();
            let ind_int = ind.as_basic_value().into_int_value();

            let dst = function.get_nth_param(0)?.into_pointer_value();
            let src = function.get_nth_param(1)?.into_pointer_value();

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
            self.builder
                .build_store(gep, val)
                .unwrap()
                .set_alignment(4)
                .unwrap();
            let ind_next = self
                .builder
                .build_int_add(ind_int, i64_type.const_int(1, false), "ind.next")
                .unwrap();
            let cond = self
                .builder
                .build_int_compare(inkwell::IntPredicate::SLT, ind_next, len, "cond")
                .unwrap();
            let _ = self
                .builder
                .build_conditional_branch(cond, body, exit)
                .unwrap();
            ind.add_incoming(&[(&i64_type.const_int(0, false), entry), (&ind_next, body)]);

            self.builder.position_at_end(exit);
            self.builder.build_return(None).unwrap();

            for name in ["noalias", "nocapture", "noundef"].iter() {
                let attr = {
                    let kind_id = Attribute::get_named_enum_kind_id(name);
                    self.context.create_enum_attribute(kind_id, 0)
                };
                function.add_attribute(AttributeLoc::Param(0), attr);
                function.add_attribute(AttributeLoc::Param(1), attr);
            }

            function.print_to_stderr();

            Some(())
        }
    }

    use inkwell::targets::*;
    Target::initialize_all(&InitializationConfig::default());
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
    {
        let function = codegen
            .module
            .get_function("pow2")
            .ok_or(Error::other("Unable to find `sum` function"))?;
        function.print_to_stderr();
    }

    let pow2: JitFunction<SumFunc> = unsafe { codegen.execution_engine.get_function("pow2").ok() }
        .ok_or(Error::other("Unable to JIT compile `sum` function"))?;

    codegen
        .execution_engine
        .get_function_value("pow2")
        .unwrap()
        .print_to_stderr();

    let src = [3f32, 1f32, 4f32, 1f32, 5f32, 9f32];
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

fn main() -> Result<()> {
    if false {
        main_cranelift()
    } else {
        main_inkwell()
    }
}
