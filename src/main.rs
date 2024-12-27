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
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    use inkwell::module::Module;
    use inkwell::OptimizationLevel;
    use inkwell::values::*;

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
        fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
            let i64_type = self.context.i64_type();
            let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
            let f32_type = self.context.f32_type();
            let fn_type = i64_type.fn_type(&[ptr_type.into(), ptr_type.into(), i64_type.into()], false);
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
            let remainder_entry = self.context.append_basic_block(function, "remainder.entry");
            let remainder = self.context.append_basic_block(function, "remainder");
            let exit = self.context.append_basic_block(function, "exit");

            let vec_ty = f32_type.vec_type(4);
            // let vscale_i64 = inkwell::intrinsics::Intrinsic::find("llvm.vscale.i64").unwrap();
            // let vscale_i64 = vscale_i64.get_declaration(&self.module, &[]).unwrap();

            self.builder.position_at_end(entry);
            // let vscale = {
            //     let call = self.builder.build_call(vscale_i64, &[], "vscale").unwrap();
            //     call.set_tail_call(true);
            //     call.try_as_basic_value().left().unwrap().into_int_value()
            // };
            let vscale = i64_type.const_int(4, false);
            let len = function.get_nth_param(2)?.into_int_value();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::ULE, vscale, len, "cond").unwrap();
            let _ = self.builder.build_conditional_branch(cond, body, remainder_entry).unwrap();

            self.builder.position_at_end(body);
            let ind_vec = self.builder.build_phi(i64_type, "ind.vec").unwrap();
            let ind_vec_int = ind_vec.as_basic_value().into_int_value();

            let dst = function.get_nth_param(0)?.into_pointer_value();
            let src = function.get_nth_param(1)?.into_pointer_value();

            let gep = unsafe { self.builder.build_in_bounds_gep(f32_type, src, &[ind_vec_int], "gep.src").unwrap() };
            let val = {
                let tmp = self.builder.build_load(vec_ty, gep, "val").unwrap();
                tmp.as_instruction_value().unwrap().set_alignment(4).unwrap();
                tmp.into_vector_value()
            };
            let val = self.builder.build_float_mul(val, val, "val2").unwrap();
            let gep = unsafe { self.builder.build_in_bounds_gep(f32_type, dst, &[ind_vec_int], "gep.dst").unwrap() };
            self.builder.build_store(gep, val).unwrap().set_alignment(4).unwrap();
            let ind_vec_next = self.builder.build_int_add(ind_vec_int, vscale, "ind.vec.next").unwrap();
            let next_end = self.builder.build_int_add(ind_vec_next, vscale, "next.end").unwrap();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::ULE, next_end, len, "cond").unwrap();
            let _ = self.builder.build_conditional_branch(cond, body, remainder_entry).unwrap();
            ind_vec.add_incoming(&[(&i64_type.const_int(0, false), entry), (&ind_vec_next, body)]);

            self.builder.position_at_end(remainder_entry);
            let ind_init = self.builder.build_phi(i64_type, "ind.init").unwrap();
            let ind_init_int = ind_init.as_basic_value().into_int_value();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::ULT, ind_init_int, len, "cond").unwrap();
            let _ = self.builder.build_conditional_branch(cond, remainder, exit).unwrap();
            ind_init.add_incoming(&[(&i64_type.const_int(0, false), entry), (&ind_vec_next, body)]);

            self.builder.position_at_end(remainder);
            let ind = self.builder.build_phi(i64_type, "ind").unwrap();
            let ind_int = ind.as_basic_value().into_int_value();
            let gep = unsafe { self.builder.build_in_bounds_gep(f32_type, src, &[ind_int], "gep.src").unwrap() };
            let val = self.builder.build_load(f32_type, gep, "val").unwrap().into_float_value();
            let val = self.builder.build_float_mul(val, val, "val2").unwrap();
            let gep = unsafe { self.builder.build_in_bounds_gep(f32_type, dst, &[ind_int], "gep.dst").unwrap() };
            let _ = self.builder.build_store(gep, val).unwrap();
            let ind_next = self.builder.build_int_add(ind_int, i64_type.const_int(1, false), "ind.next").unwrap();
            let cond = self.builder.build_int_compare(inkwell::IntPredicate::ULT, ind_next, len, "cond").unwrap();
            let _ = self.builder.build_conditional_branch(cond, remainder, exit).unwrap();
            ind.add_incoming(&[(&ind_init_int, remainder_entry), (&ind_next, remainder)]);

            self.builder.position_at_end(exit);
            self.builder.build_return(None).unwrap();

            unsafe { self.execution_engine.get_function("pow2").ok() }
        }
    }

    let context = Context::create();
    let module = context.create_module("pow2");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None).map_err(|e| Error::other(format!("{:?}", e)))?;
    let codegen = CodeGen {
        context: &context,
        module,
        builder: context.create_builder(),
        execution_engine,
    };

    let pow2 = codegen.jit_compile_sum()
        .ok_or(Error::other("Unable to JIT compile `sum` function"))?;

    println!("{}", codegen.module.to_string());
    
    let src = [3f32, 1f32, 4f32, 1f32, 5f32, 9f32];
    let mut dst = vec![0f32; src.len()];

    unsafe {
        pow2.call(dst.as_mut_ptr() as *const u8, src.as_ptr() as *const u8, src.len() as u64);
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
