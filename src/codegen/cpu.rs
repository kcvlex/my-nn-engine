pub(crate) mod blas;
mod llvm;
mod omp;
mod op;
mod translator;

use std::collections::HashMap;
use std::path::Path;

use inkwell::basic_block::BasicBlock;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::intrinsics::Intrinsic;
use inkwell::module::Module;
use inkwell::targets::CodeModel;
use inkwell::targets::FileType;
use inkwell::targets::InitializationConfig;
use inkwell::targets::RelocMode;
use inkwell::targets::Target;
use inkwell::targets::TargetMachine;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;
use inkwell::OptimizationLevel;

use crate::codegen::cpu::blas::*;
use crate::codegen::cpu::llvm::*;
use crate::codegen::cpu::omp::*;
use crate::codegen::cpu::op::*;
use crate::codegen::cpu::translator::*;
use crate::codegen::*;
use crate::onnx::model::ValueId;
use crate::onnx::operator::args;
use crate::onnx::operator::Contiguous;
use crate::onnx::operator::Layout;
use crate::onnx::operator::Operator;
use crate::options::Options;
use crate::schedule::*;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

struct UnitInfo<'ll> {
    ty: UnitType,
    module: Module<'ll>,
    func: FunctionValue<'ll>,
    entry: BasicBlock<'ll>,
}

enum UnitType {
    Main,
    Kernel(KernelId),
}

pub struct CodeGenContext {
    pub schedule: Schedule,
    blas_backend: blas::Backend,
    mem_size: Vec<u64>,
    value2alloc: HashMap<ValueId, AllocateInfo>,
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
    profile: bool,
}

// TODO: correct?
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
    pub fn new(schedule: Schedule, blas_backend: blas::Backend) -> Result<Self, CodeGenError> {
        let mem_size = calc_memsize(&schedule);
        let mem_alloc_result = schedule.analysis.get::<mem_alloc::MemAllocResult>();
        let value2alloc = mem_alloc_result
            .0
            .values()
            .flatten()
            .map(|info| (info.value_id, *info))
            .collect::<HashMap<_, _>>();

        Ok(CodeGenContext {
            schedule,
            blas_backend,
            mem_size,
            value2alloc,
        })
    }

    fn need_to_generate(&self, kernel_id: KernelId) -> bool {
        let kernel = &self.schedule.kernels[kernel_id];
        // if node.is_dummy() {
        //     return false;
        // }
        if matches_opaque!(kernel, Operator::Identity | Operator::Reinterpret(_)) {
            let chunk_in = self.value2alloc.get(&kernel.inputs[0]).map(|info| &info.ty);
            let chunk_out = self
                .value2alloc
                .get(&kernel.outputs[0])
                .map(|info| &info.ty);
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

    pub fn all_necessary_kernels(&self) -> Vec<KernelId> {
        self.schedule
            .kernels
            .iter()
            .map(|(id, _)| id)
            .filter(|&id| self.need_to_generate(id))
            .collect()
    }
}

// TODO: Target dependent value
fn memory_usage(sched: &Schedule, value: ValueId) -> u64 {
    let result_ty = sched.get_resolved_tensor_type(value).unwrap();
    let data_size = match result_ty.elem_type {
        DataType::SInt(SIntType::I32) => 4,
        DataType::SInt(SIntType::I64) => 8,
        DataType::UInt(UIntType::U64) => 8,
        DataType::Float(FloatType::F32) => 4,
        DataType::Float(FloatType::F64) => 8,
    };
    (result_ty.dims.size() * data_size).try_into().unwrap()
}

fn calc_memsize(sched: &Schedule) -> Vec<u64> {
    let max_chunk_id = sched.max_chunk_id().map(|x| x + 1).unwrap_or(0);
    let mut mem_size = vec![0; max_chunk_id];
    let mem_alloc_result = sched.analysis.get::<mem_alloc::MemAllocResult>();
    for info in mem_alloc_result.0.values().flatten() {
        if let Some(chunk_id) = info.ty.chunk_id() {
            mem_size[chunk_id] = mem_size[chunk_id].max(memory_usage(sched, info.value_id));
        }
    }
    mem_size
}

impl CodeGenContext {
    pub fn new_codegen_for_kernel<'ll>(
        &self,
        kernel_id: KernelId,
        ll_ctx: &'ll Context,
    ) -> Result<CodeGen<'ll, '_>, CodeGenError> {
        let kernel = &self.schedule.kernels[kernel_id];
        let target_machine = target_machine()?;
        let attrs = Attributes::new(ll_ctx, &target_machine);
        let module = ll_ctx.create_module(get_kernel_name_or(kernel, kernel_id).as_str());
        let func = self.declare_node_func(kernel_id, ll_ctx, &module, &attrs);
        let entry = ll_ctx.append_basic_block(func, "entry");
        let ty = UnitType::Kernel(kernel_id);
        let unit = UnitInfo {
            ty,
            module,
            func,
            entry,
        };

        self.new_codegen_with_func(ll_ctx, unit, attrs, target_machine, false)
    }

    pub fn new_codegen_for_main<'ll>(
        &self,
        ll_ctx: &'ll Context,
        options: &Options,
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
        self.new_codegen_with_func(ll_ctx, unit, attrs, target_machine, options.profile)
    }

    fn new_codegen_with_func<'ll>(
        &self,
        ll_ctx: &'ll Context,
        unit: UnitInfo<'ll>,
        attrs: Attributes,
        target_machine: TargetMachine,
        profile: bool,
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
        let pow = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.pow", &[f32_ty, f32_ty])?,
            f_f64: get_intrinsic!("llvm.pow", &[f64_ty, f64_ty])?,
        };
        let sqrt = FloatIntrinsics {
            f_f32: get_intrinsic!("llvm.sqrt", &[f32_ty])?,
            f_f64: get_intrinsic!("llvm.sqrt", &[f64_ty])?,
        };
        let smin_i32 = get_intrinsic!("llvm.smin", &[i32_ty, i32_ty])?;
        let smin_i64 = get_intrinsic!("llvm.smin", &[i64_ty, i64_ty])?;
        let smax_i32 = get_intrinsic!("llvm.smax", &[i32_ty, i32_ty])?;
        let smax_i64 = get_intrinsic!("llvm.smax", &[i64_ty, i64_ty])?;
        let tanh = FloatIntrinsics {
            f_f32: unit
                .module
                .add_function("tanhf", f32_ty.fn_type(&[f32_ty.into()], false), None),
            f_f64: unit
                .module
                .add_function("tanh", f64_ty.fn_type(&[f64_ty.into()], false), None),
        };
        // let lifetime_start = get_intrinsic!("llvm.lifetime.start", &[i64_ty, ptr_ty])?;
        // let lifetime_end = get_intrinsic!("llvm.lifetime.end", &[i64_ty, ptr_ty])?;

        let intrinsics = Intrinsics {
            ceil,
            exp,
            floor,
            fma,
            fmax,
            log,
            pow,
            sqrt,
            smin_i32,
            smin_i64,
            smax_i32,
            smax_i64,
            tanh,
            // lifetime_start,
            // lifetime_end,
        };

        let blas = BLAS::new(ll_ctx, &unit.module, self.blas_backend);
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
            profile,
        })
    }

    fn declare_node_func<'ctx>(
        &self,
        kernel_id: KernelId,
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        attrs: &Attributes,
    ) -> FunctionValue<'ctx> {
        let kernel = &self.schedule.kernels[kernel_id];
        let allocs = kernel
            .outputs
            .iter()
            .chain(kernel.inputs.iter())
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
        let func = module.add_function(
            get_kernel_name_or(kernel, kernel_id).as_str(),
            fn_type,
            None,
        );
        attrs.add_default_attributes(&func, |i| is_noalias[i]);
        func
    }
}

// TODO
fn get_kernel_name_or(kernel: &Kernel, kernel_id: KernelId) -> String {
    if kernel.name.is_empty() {
        format!("kernel.{}", kernel_id.index())
    } else {
        kernel.name.clone()
    }
}

impl<'ll> CodeGen<'ll, '_> {
    pub fn compile(&self) -> Result<(), CodeGenError> {
        (match self.unit.ty {
            UnitType::Main => self.compile_main(),
            UnitType::Kernel(kernel_id) => self.compile_kernel(kernel_id),
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

    pub fn into_module(self) -> Module<'ll> {
        self.unit.module
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
                let value = &self.gen_ctx.schedule.get_value($value_id);
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

        for (i, arr) in [
            &self.gen_ctx.schedule.outputs,
            &self.gen_ctx.schedule.inputs,
        ]
        .iter()
        .copied()
        .enumerate()
        {
            let ptr = self
                .unit
                .func
                .get_nth_param(i as u32)
                .unwrap()
                .into_pointer_value();
            for (i, value_id) in arr.iter().enumerate() {
                // let value_id = match self.gen_ctx.graph.nodes[*node_id].op {
                //     Operator::Input(v) | Operator::Output(v) => v,
                //     _ => unreachable!(),
                // };

                // // TODO: necessary?
                // if self.gen_ctx.graph.initializer.contains_key(&value_id) {
                //     continue;
                // }

                init_ptr!(*value_id, ptr, i);
            }
        }

        {
            let ptr = self
                .unit
                .func
                .get_nth_param(2)
                .unwrap()
                .into_pointer_value();
            for (i, value_id) in self.gen_ctx.schedule.initializers.iter().enumerate() {
                init_ptr!(*value_id, ptr, i);
            }
        }

        Ok(ptr_values)
    }

    fn compile_main(&self) -> Result<(), BuilderError> {
        let mut ptr_values = self.init_main_args()?;
        let mut chunk2ptr = HashMap::new();
        let builder = self.ll_ctx.create_builder();

        let rdtsc = self.profile.then(|| {
            Intrinsic::find("llvm.readcyclecounter")
                .and_then(|i| i.get_declaration(&self.unit.module, &[]))
                .unwrap()
        });
        let fprintf = if self.profile {
            let i32_ty = self.ll_ctx.i32_type();
            let ptr_ty = self.ll_ctx.ptr_type(AddressSpace::default());
            let fn_ty = i32_ty.fn_type(&[ptr_ty.into(), ptr_ty.into()], true);
            Some(self.unit.module.add_function(
                "fprintf",
                fn_ty,
                Some(inkwell::module::Linkage::External),
            ))
        } else {
            None
        };
        let stderr_ptr = if self.profile {
            let ptr_ty = self.ll_ctx.ptr_type(AddressSpace::default());
            let global = self.unit.module.add_global(ptr_ty, None, "stderr");
            global.set_linkage(inkwell::module::Linkage::External);
            Some(global)
        } else {
            None
        };

        // Arena: allocate one big buffer, assign chunks at offsets
        builder.position_at_end(self.unit.entry);
        let i8_ty = self.ll_ctx.i8_type();
        let i64_ty = self.ll_ctx.i64_type();
        let alignment = 256u64;
        let align_up = |size: u64| (size + alignment - 1) & !(alignment - 1);
        {
            let mut offset = 0u64;
            for (chunk_id, &size) in self.gen_ctx.mem_size.iter().enumerate() {
                let ptr = if offset == 0 && chunk_id == 0 {
                    // first chunk: malloc the full arena
                    let arena_size: u64 = self.gen_ctx.mem_size.iter().map(|&s| align_up(s)).sum();
                    let arena = builder.build_array_malloc(
                        i8_ty,
                        i64_ty.const_int(arena_size, false),
                        "arena",
                    )?;
                    chunk2ptr.insert(usize::MAX, arena); // store arena base
                    arena
                } else {
                    let arena = *chunk2ptr.get(&usize::MAX).unwrap();
                    unsafe {
                        builder.build_in_bounds_gep(
                            i8_ty,
                            arena,
                            &[i64_ty.const_int(offset, false)],
                            &format!("chunk.{chunk_id}"),
                        )?
                    }
                };
                chunk2ptr.insert(chunk_id, ptr);
                offset += align_up(size);
            }
        }

        for (kernel_id, kernel) in self.gen_ctx.schedule.kernels.iter() {
            let function = if !self.gen_ctx.need_to_generate(kernel_id) {
                None
            } else {
                Some(self.gen_ctx.declare_node_func(
                    kernel_id,
                    self.ll_ctx,
                    &self.unit.module,
                    &self.attrs,
                ))
            };

            builder.position_at_end(self.unit.entry);

            let mem_alloc = self
                .gen_ctx
                .schedule
                .analysis
                .get::<mem_alloc::MemAllocResult>();
            for alloc in mem_alloc.0[&kernel_id].iter() {
                let dst_ptr = match alloc.ty {
                    AllocateType::Chunk(chunk) => *chunk2ptr.get(&chunk).unwrap(),
                    AllocateType::Input(v) | AllocateType::Output(v) => {
                        *ptr_values.get(&v).unwrap()
                    }
                };
                ptr_values.insert(alloc.value_id, dst_ptr);
            }

            if let Some(function) = function {
                let start = if let Some(rdtsc) = rdtsc {
                    Some(
                        builder
                            .build_call(rdtsc, &[], "tsc.start")?
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                            .into_int_value(),
                    )
                } else {
                    None
                };

                let args = kernel
                    .outputs
                    .iter()
                    .chain(kernel.inputs.iter())
                    .map(|&id| ptr_values.get(&id).unwrap())
                    .map(|ptr| (*ptr).into())
                    .collect::<Vec<_>>();
                let call = builder.build_call(function, &args[..], "")?;
                call.set_tail_call(true);

                if let (Some(start), Some(rdtsc), Some(fprintf), Some(stderr_ptr)) =
                    (start, rdtsc, fprintf, stderr_ptr)
                {
                    let end = builder
                        .build_call(rdtsc, &[], "tsc.end")?
                        .try_as_basic_value()
                        .left()
                        .unwrap()
                        .into_int_value();
                    let diff = builder.build_int_sub(end, start, "tsc.diff")?;
                    let kernel_name = get_kernel_name_or(kernel, kernel_id);
                    let op_type = match &kernel.body {
                        KernelBody::Opaque(Opaque { op }) => op.name().to_string(),
                        KernelBody::ElementWises(_) => "ElementWises".to_string(),
                    };
                    let fmt = builder.build_global_string_ptr(
                        &format!("[profile] [{op_type}] {kernel_name}: %lu cycles\n"),
                        &format!("fmt.{}", kernel_id.index()),
                    )?;
                    let stderr_val = builder.build_load(
                        self.ll_ctx.ptr_type(AddressSpace::default()),
                        stderr_ptr.as_pointer_value(),
                        "stderr",
                    )?;
                    builder.build_call(
                        fprintf,
                        &[
                            stderr_val.into(),
                            fmt.as_pointer_value().into(),
                            diff.into(),
                        ],
                        "",
                    )?;
                }
            }
        }
        builder.position_at_end(self.unit.entry);

        if let Some(arena) = chunk2ptr.get(&usize::MAX) {
            builder.build_free(*arena)?;
        }

        builder.build_return(None)?;
        Ok(())
    }

    fn compile_kernel(&self, kernel_id: KernelId) -> Result<(), BuilderError> {
        let kernel = &self.gen_ctx.schedule.kernels[kernel_id];
        // dbg!(&node);
        let args = kernel
            .outputs
            .iter()
            .chain(kernel.inputs.iter())
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
                    .schedule
                    .get_resolved_tensor_type(*id)
                    .unwrap()
                    .clone();
                let offset = self.ll_ctx.i64_type().const_int(0, false);
                TensorPtr::new_with_index(ptr, ty, offset, "ptr", i)
            })
            .collect::<Vec<_>>();

        // TODO
        if matches_opaque!(kernel, Operator::Identity | Operator::Reinterpret(_)) {
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

        let omp_result = self
            .gen_ctx
            .schedule
            .analysis
            .get::<crate::schedule::omp::OmpResult>();
        let use_omp = omp_result.0.contains(&kernel_id);

        let mut ptrs = ptrs;

        // TODO: When same Input is used in multiple nodes
        let adjust_ptrs_and_convert_op = |op: &Operator,
                                          ptrs: &mut [TensorPtr<'_>],
                                          operands: &[(DataType, Option<usize>)],
                                          target_dim: &ResolvedTensorDims|
         -> SingleOpcode {
            match op {
                Operator::Add | Operator::Div | Operator::Mul | Operator::Pow | Operator::Sub => {
                    assert!(operands.len() == 2);
                    for (_, i) in operands.iter().filter_map(|(dt, idx)| idx.map(|i| (dt, i))) {
                        ptrs[i].ty = ptrs[i].ty.broadcast(target_dim);
                    }
                }

                Operator::Contiguous(_) |
                Operator::BatchNormalization(_) |
                Operator::Cast(_) |
                Operator::Exp |
                Operator::GeLU(_) |
                Operator::LeakyReLU(_) |
                Operator::Log |
                Operator::Reciprocal |
                Operator::ReLU |
                Operator::Sigmoid |
                Operator::Sqrt |
                Operator::Tanh => (),

                _ => unreachable!(),
            };

            match op {
                Operator::Add => SingleOpcode::Add,
                Operator::BatchNormalization(bn) => SingleOpcode::BatchNorm(*bn),
                Operator::Cast(cast) => {
                    assert!(operands.len() == 1);
                    let src = operands[0].0;
                    SingleOpcode::Cast(src, cast.to)
                }
                Operator::Div => SingleOpcode::Div,
                Operator::Exp => SingleOpcode::Exp,
                Operator::GeLU(v) => SingleOpcode::GeLU(*v),
                Operator::LeakyReLU(v) => SingleOpcode::LeakyReLU(*v),
                Operator::Log => SingleOpcode::Log,
                Operator::Mul => SingleOpcode::Mul,
                Operator::Pow => {
                    assert!(operands.len() == 2);
                    let lhs = operands[0].0;
                    let rhs = operands[1].0;
                    SingleOpcode::Pow(lhs, rhs)
                }
                Operator::Reciprocal => SingleOpcode::Reciprocal,
                Operator::ReLU => SingleOpcode::ReLU,
                Operator::Sigmoid => SingleOpcode::Sigmoid,
                Operator::Sqrt => SingleOpcode::Sqrt,
                Operator::Sub => SingleOpcode::Sub,
                Operator::Tanh => SingleOpcode::Tanh,
                _ => unreachable!(),
            }
        };

        let exit = match &kernel.body {
            KernelBody::Opaque(Opaque { op }) => match op {
                Operator::Add |
                Operator::BatchNormalization(_) |
                Operator::Exp |
                Operator::LeakyReLU(_) |
                Operator::Log |
                Operator::Mul |
                Operator::Pow |
                Operator::Reciprocal |
                Operator::ReLU |
                Operator::Sigmoid |
                Operator::Sqrt |
                Operator::Sub |
                Operator::Tanh => unreachable!(),

                Operator::Contiguous(Contiguous { ref ops }) => {
                    translator.build_contiguous(&ptrs[0], ptrs[1].clone(), entry, ops)
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
                Operator::MatMul => {
                    translator.build_matmul(&ptrs[0], &ptrs[1], &ptrs[2], ptrs.get(3), entry)
                }
                Operator::Gather(ref gather) => translator.build_gather(
                    ptrs[0].clone(),
                    ptrs[1].clone(),
                    ptrs[2].clone(),
                    entry,
                    gather,
                ),
                Operator::Gemm(ref gemm) => {
                    translator.build_gemm(&ptrs[0], &ptrs[1], &ptrs[2], ptrs.get(3), entry, gemm)
                }
                Operator::MaxPool(ref pooling) => match pooling.layout {
                    Layout::NCHW => {
                        translator.build_maxpool_nchw(&ptrs[0], &ptrs[1], pooling, entry)
                    }
                    Layout::NHWC => {
                        translator.build_maxpool_nhwc(&ptrs[0], &ptrs[1], pooling, entry)
                    }
                },
                Operator::Im2Col(ref im2col) => {
                    translator.build_im2col(&ptrs[0], &ptrs[1], im2col, entry)
                }
                Operator::OneHot(ref one_hot) => {
                    translator.build_one_hot(ptrs[0].clone(), ptrs[1].clone(), entry, one_hot)
                }
                Operator::ReduceMatrix(op) => {
                    let m = ptrs[1].ty.dims[0] as u64;
                    let n = ptrs[1].ty.dims[1] as u64;
                    let elem_type = ptrs[0].ty.elem_type;
                    translator.build_matrix_reduce(&ptrs, elem_type, (m, n), *op, entry)
                }
                Operator::Resize(ref resize) => {
                    translator.build_resize(ptrs[0].clone(), ptrs[1].clone(), entry, resize)
                }
                Operator::Softmax(ref softmax) => translator.build_softmax(
                    ptrs[0].clone(),
                    ptrs[1].clone(),
                    entry,
                    softmax,
                    use_omp,
                ),
                Operator::LayerNormalization(ref ln) => translator.build_layer_norm(
                    ptrs[0].clone(),
                    ptrs[1 + args::LAYER_NORM_DATA].clone(),
                    ptrs[1 + args::LAYER_NORM_SCALE].clone(),
                    ptrs[1 + args::LAYER_NORM_BIAS].clone(),
                    entry,
                    ln,
                ),
                Operator::Split(ref split) => translator.build_split(
                    &ptrs[..ptrs.len() - 1],
                    ptrs.last().unwrap().clone(),
                    entry,
                    split,
                ),
                Operator::Attention(_) => panic!(),
                Operator::BatchedGemm(ref gemm) => {
                    translator.build_batched_gemm(&ptrs[0], &ptrs[1], &ptrs[2], entry, gemm)
                }
                _ => todo!("{:?}", op),
            },
            KernelBody::ElementWises(ElementWises { ops }) => {
                let target_dim = ptrs[0].ty.dims.clone();
                // Fused kernels may contain unary ops whose inputs have fewer
                // dimensions than the kernel output (e.g., Reciprocal(scalar)
                // fused with a Mul that produces a tensor).
                for ptr in ptrs[1..].iter_mut() {
                    ptr.ty = ptr.ty.broadcast(&target_dim);
                }
                let ops: Vec<_> = ops
                    .iter()
                    .map(|(op, args)| {
                        let operands = args
                            .iter()
                            .map(|arg| match arg {
                                ElementwiseOpArg::Input(i) => {
                                    let dtype = ptrs[1 + *i].ty.elem_type;
                                    (dtype, Some(*i))
                                }
                                ElementwiseOpArg::NthResult(_) => {
                                    // Use output type for NthResult as a stop-gap.
                                    // TODO: correct?
                                    (ptrs[0].ty.elem_type, None)
                                }
                            })
                            .collect::<Vec<_>>();
                        let operator =
                            adjust_ptrs_and_convert_op(op, &mut ptrs[1..], &operands, &target_dim);
                        (operator, args.clone())
                    })
                    .collect();
                let op = Operation {
                    opcode: Opcode::Fused(ops),
                    operands: ptrs.clone().into(),
                };
                translator.build_flat_loop(op, entry, use_omp)
            }
        }?;

        builder.position_at_end(exit);
        builder.build_return(None)?;

        Ok(())
    }
}
