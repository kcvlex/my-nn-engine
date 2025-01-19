use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone)]
pub struct OMP<'ctx> {
    i32_type: IntType<'ctx>,

    fork_call: FunctionValue<'ctx>,
    static_init: FunctionValue<'ctx>,
    static_fini: FunctionValue<'ctx>,

    dummy_ident: GlobalValue<'ctx>,
}

// https://github.com/llvm/llvm-project/blob/0cb2fe5183c9b25bb96140c27d12b1ad4a80aa92/llvm/include/llvm/Frontend/OpenMP/OMPConstants.h#LL81
#[derive(Clone, Copy)]
pub enum ScheduleType {
    UnorderedStatic = 34,
}

pub struct ForkCallArgs<'ctx> {
    pub outlined: FunctionValue<'ctx>,
    pub args: Vec<PointerValue<'ctx>>,
}

pub struct StaticInitArgs<'ctx> {
    pub tid: IntValue<'ctx>,
    pub sched: ScheduleType,
    pub is_last: PointerValue<'ctx>,
    pub lb: PointerValue<'ctx>,
    pub ub: PointerValue<'ctx>,
    pub stride: PointerValue<'ctx>,
    pub incr: IntValue<'ctx>,
}

pub struct StaticFiniArgs<'ctx> {
    pub tid: IntValue<'ctx>,
}

impl<'ctx> OMP<'ctx> {
    pub fn new(
        ctx: &'ctx Context,
        module: &Module<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<Self, BuilderError> {
        let ptr_type = ctx.ptr_type(AddressSpace::default());
        let i32_type = ctx.i32_type();
        let void_type = ctx.void_type();

        let fork_call =
            void_type.fn_type(&[ptr_type.into(), i32_type.into(), ptr_type.into()], true);
        let fork_call = module.add_function("__kmpc_fork_call", fork_call, None);

        let static_init = void_type.fn_type(
            &[
                ptr_type.into(),
                i32_type.into(),
                i32_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                ptr_type.into(),
                i32_type.into(),
                i32_type.into(),
            ],
            false,
        );
        let static_init = module.add_function("__kmpc_for_static_init_4", static_init, None);

        let static_fini = void_type.fn_type(&[ptr_type.into(), i32_type.into()], false);
        let static_fini = module.add_function("__kmpc_for_static_fini", static_fini, None);

        let ident_ty = ctx.struct_type(
            &[
                i32_type.into(),
                i32_type.into(),
                i32_type.into(),
                i32_type.into(),
                ptr_type.into(),
            ],
            false,
        );
        let dummy_ident_value =
            unsafe { builder.build_global_string("wn;unknown;0;0;;", "dummy") }?;
        let dummy_ident_value = ctx.const_struct(
            &[
                i32_type.const_zero().into(),
                i32_type.const_zero().into(),
                i32_type.const_zero().into(),
                i32_type.const_zero().into(),
                dummy_ident_value.as_pointer_value().into(),
            ],
            false,
        );
        let dummy_ident = module.add_global(ident_ty, None, "dummy_ident");
        dummy_ident.set_initializer(&dummy_ident_value);

        Ok(Self {
            i32_type,

            fork_call,
            static_init,
            static_fini,

            dummy_ident,
        })
    }

    pub fn fork_call(
        &self,
        builder: &Builder<'ctx>,
        args: &ForkCallArgs<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let mut vec = Vec::with_capacity(args.args.len() + 2);
        vec.push(args.outlined.as_global_value().as_pointer_value().into());
        vec.push(
            self.i32_type
                .const_int(args.args.len() as u64, false)
                .into(),
        );
        for arg in args.args.iter() {
            vec.push((*arg).into());
        }
        builder.build_call(self.fork_call, &vec, "")
    }

    pub fn static_init(
        &self,
        builder: &Builder<'ctx>,
        args: &StaticInitArgs<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        builder.build_call(
            self.static_init,
            &[
                self.dummy_ident.as_pointer_value().into(),
                args.tid.into(),
                self.i32_type.const_int(args.sched as u64, false).into(),
                args.is_last.into(),
                args.lb.into(),
                args.ub.into(),
                args.stride.into(),
                args.incr.into(),
                self.i32_type.const_int(1, false).into(),
            ],
            "",
        )
    }

    pub fn static_fini(
        &self,
        builder: &Builder<'ctx>,
        args: &StaticFiniArgs<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        builder.build_call(
            self.static_fini,
            &[self.dummy_ident.as_pointer_value().into(), args.tid.into()],
            "",
        )
    }
}
