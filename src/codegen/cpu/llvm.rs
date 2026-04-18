use inkwell::attributes::*;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::targets::TargetMachine;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;

use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

pub struct FloatIntrinsics<'ll> {
    pub f_f32: FunctionValue<'ll>,
    pub f_f64: FunctionValue<'ll>,
}

impl<'ctx> FloatIntrinsics<'ctx> {
    pub fn get(&self, ty: FloatType) -> FunctionValue<'ctx> {
        match ty {
            FloatType::F32 => self.f_f32,
            FloatType::F64 => self.f_f64,
        }
    }
}

#[allow(dead_code)]
pub struct Intrinsics<'ll> {
    pub ceil: FloatIntrinsics<'ll>,
    pub cos: FloatIntrinsics<'ll>,
    pub exp: FloatIntrinsics<'ll>,
    pub fma: FloatIntrinsics<'ll>,
    pub fmax: FloatIntrinsics<'ll>,
    pub fmin: FloatIntrinsics<'ll>,
    pub floor: FloatIntrinsics<'ll>,
    pub log: FloatIntrinsics<'ll>,
    pub pow: FloatIntrinsics<'ll>,
    pub sin: FloatIntrinsics<'ll>,
    pub sqrt: FloatIntrinsics<'ll>,
    pub tanh: FloatIntrinsics<'ll>,

    pub smin_i32: FunctionValue<'ll>,
    pub smin_i64: FunctionValue<'ll>,
    pub smax_i32: FunctionValue<'ll>,
    pub smax_i64: FunctionValue<'ll>,
    // lifetime_start: FunctionValue<'ctx>,
    // lifetime_end: FunctionValue<'ctx>,
}

pub struct Attributes {
    pub noalias: Attribute,
    pub noundef: Attribute,
    pub cpu: Attribute,
    pub features: Attribute,
}

impl Attributes {
    pub fn new(context: &Context, target_machine: &TargetMachine) -> Self {
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

    pub fn add_default_attributes<P>(&self, func: &FunctionValue<'_>, is_noalias: P)
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
pub struct DebugStuff<'ll> {
    pub printf: FunctionValue<'ll>,
    pub fflush: FunctionValue<'ll>,
    pub float_fmt: GlobalValue<'ll>,
    pub i64_fmt: GlobalValue<'ll>,
    pub i64_i64_fmt: GlobalValue<'ll>,
    pub stdout: GlobalValue<'ll>,
}

impl<'ll> DebugStuff<'ll> {
    pub fn new(ctx: &'ll Context, module: &Module<'ll>, builder: &Builder<'ll>) -> Self {
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

    #[allow(dead_code)]
    pub fn print_float(
        &self,
        ctx: &'ll Context,
        builder: &Builder<'ll>,
        value: FloatValue<'ll>,
    ) -> Result<(), inkwell::builder::BuilderError> {
        let v = builder.build_float_ext(value, ctx.f64_type(), "v")?;
        builder.build_call(
            self.printf,
            &[self.float_fmt.as_pointer_value().into(), v.into()],
            "printf",
        )?;
        Ok(())
    }
}

impl SIntType {
    pub fn llvm_type<'ctx>(&self, ctx: &'ctx Context) -> inkwell::types::IntType<'ctx> {
        match self {
            SIntType::I32 => ctx.i32_type(),
            SIntType::I64 => ctx.i64_type(),
        }
    }
}

impl UIntType {
    pub fn llvm_type<'ctx>(&self, ctx: &'ctx Context) -> inkwell::types::IntType<'ctx> {
        match self {
            UIntType::U64 => ctx.i64_type(),
        }
    }
}

impl FloatType {
    pub fn llvm_type<'ctx>(&self, ctx: &'ctx Context) -> inkwell::types::FloatType<'ctx> {
        match self {
            FloatType::F32 => ctx.f32_type(),
            FloatType::F64 => ctx.f64_type(),
        }
    }
}

impl DataType {
    pub fn llvm_type<'ctx>(&self, ctx: &'ctx Context) -> BasicTypeEnum<'ctx> {
        match self {
            DataType::SInt(t) => t.llvm_type(ctx).as_basic_type_enum(),
            DataType::UInt(t) => t.llvm_type(ctx).as_basic_type_enum(),
            DataType::Float(t) => t.llvm_type(ctx).as_basic_type_enum(),
        }
    }
}
