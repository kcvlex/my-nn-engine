use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::{Linkage, Module};
use inkwell::types::*;
use inkwell::values::*;
use inkwell::AddressSpace;

#[allow(non_camel_case_types)]
pub enum CBLAS_ORDER {
    RowMajor,
    // ColMajor,
}

#[allow(non_camel_case_types)]
pub enum CBLAS_TRANSPOSE {
    NoTrans,
    Trans,
}

impl From<bool> for CBLAS_TRANSPOSE {
    fn from(value: bool) -> Self {
        if value {
            Self::Trans
        } else {
            Self::NoTrans
        }
    }
}

// TODO: Use FFI
impl CBLAS_ORDER {
    fn to_raw(&self) -> u32 {
        match self {
            Self::RowMajor => 101,
            // Self::ColMajor => 102,
        }
    }
}

impl CBLAS_TRANSPOSE {
    fn to_raw(&self) -> u32 {
        match self {
            Self::NoTrans => 111,
            Self::Trans => 112,
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub enum Precision {
    Single,
    Double,
}

impl Precision {
    fn char(&self) -> char {
        match self {
            Self::Single => 's',
            Self::Double => 'd',
        }
    }
}

struct Routines<'ctx> {
    gemm: FunctionValue<'ctx>,
    dot: FunctionValue<'ctx>,
    i32_ty: IntType<'ctx>,
    fp_ty: FloatType<'ctx>,
}

#[derive(Debug, Clone)]
pub struct GemmArgs<'ctx> {
    pub a: (PointerValue<'ctx>, bool),
    pub b: (PointerValue<'ctx>, bool),
    pub c: (PointerValue<'ctx>, bool),
    pub m: u64,
    pub n: u64,
    pub k: u64,

    #[allow(dead_code)]
    pub alpha: f64,
    #[allow(dead_code)]
    pub beta: f64,
}

pub struct DotArgs<'ctx> {
    pub x: (PointerValue<'ctx>, u64),
    pub y: (PointerValue<'ctx>, u64),
    pub n: u64,
}

impl<'ctx> Routines<'ctx> {
    fn new(ctx: &'ctx Context, module: &Module<'ctx>, precision: Precision) -> Self {
        let fp_ty = match precision {
            Precision::Single => ctx.f32_type(),
            Precision::Double => ctx.f64_type(),
        };

        let void_ty = ctx.void_type();
        let ptr_ty = ctx.ptr_type(AddressSpace::default());
        let i32_ty = ctx.i32_type();

        let gemm = void_ty.fn_type(
            &[
                i32_ty.into(), // order
                i32_ty.into(), // trans_a
                i32_ty.into(), // trans_b
                i32_ty.into(), // M
                i32_ty.into(), // N
                i32_ty.into(), // K
                fp_ty.into(),  // alpha
                ptr_ty.into(), // A
                i32_ty.into(), // lda
                ptr_ty.into(), // B
                i32_ty.into(), // ldb
                fp_ty.into(),  // beta
                ptr_ty.into(), // B
                i32_ty.into(), // ldc
            ],
            false,
        );
        let gemm = module.add_function(
            format!("cblas_{}gemm", precision.char()).as_str(),
            gemm,
            Some(Linkage::External),
        );

        let dot = fp_ty.fn_type(
            &[
                i32_ty.into(), // N
                ptr_ty.into(), // X
                i32_ty.into(), // incX
                ptr_ty.into(), // Y
                i32_ty.into(), // incY
            ],
            false,
        );
        let dot = module.add_function(
            format!("cblas_{}dot", precision.char()).as_str(),
            dot,
            Some(Linkage::External),
        );

        Self {
            gemm,
            dot,
            fp_ty,
            i32_ty,
        }
    }

    fn call_gemm(
        &self,
        gemm: &GemmArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let (a_ptr, a_trans) = gemm.a;
        let (b_ptr, b_trans) = gemm.b;
        let (c_ptr, c_trans) = gemm.c;

        if c_trans {
            unimplemented!()
        }

        let m = self.i32_ty.const_int(gemm.m, false);
        let n = self.i32_ty.const_int(gemm.n, false);
        let k = self.i32_ty.const_int(gemm.k, false);
        let inc_a = if !a_trans { k } else { m };
        let inc_b = if !b_trans { n } else { k };

        // TODO: alpha & beta
        builder.build_call(
            self.gemm,
            &[
                self.i32_ty
                    .const_int(CBLAS_ORDER::RowMajor.to_raw().into(), false)
                    .into(),
                self.i32_ty
                    .const_int(CBLAS_TRANSPOSE::from(a_trans).to_raw().into(), false)
                    .into(),
                self.i32_ty
                    .const_int(CBLAS_TRANSPOSE::from(b_trans).to_raw().into(), false)
                    .into(),
                m.into(),
                n.into(),
                k.into(),
                self.fp_ty.const_float(1.0).into(),
                a_ptr.into(),
                inc_a.into(),
                b_ptr.into(),
                inc_b.into(),
                self.fp_ty.const_float(0.0).into(),
                c_ptr.into(),
                n.into(),
            ],
            "",
        )
    }

    fn call_dot(
        &self,
        dot: &DotArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let (ptr_x, inc_x) = dot.x;
        let (ptr_y, inc_y) = dot.y;
        builder.build_call(
            self.dot,
            &[
                self.i32_ty.const_int(dot.n, false).into(),
                ptr_x.into(),
                self.i32_ty.const_int(inc_x, false).into(),
                ptr_y.into(),
                self.i32_ty.const_int(inc_y, false).into(),
            ],
            "",
        )
    }
}

#[allow(non_camel_case_types)]
pub struct BLAS<'ctx> {
    s_routines: Routines<'ctx>,
    d_routines: Routines<'ctx>,
}

impl<'ctx> BLAS<'ctx> {
    pub fn new(ctx: &'ctx Context, module: &'_ Module<'ctx>) -> Self {
        Self {
            s_routines: Routines::new(ctx, module, Precision::Single),
            d_routines: Routines::new(ctx, module, Precision::Double),
        }
    }

    pub fn call_gemm(
        &self,
        precision: Precision,
        gemm: &GemmArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        match precision {
            Precision::Single => self.s_routines.call_gemm(gemm, builder),
            Precision::Double => self.d_routines.call_gemm(gemm, builder),
        }
    }

    #[allow(dead_code)]
    pub fn call_dot(
        &self,
        precision: Precision,
        dot: &DotArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        match precision {
            Precision::Single => self.s_routines.call_dot(dot, builder),
            Precision::Double => self.d_routines.call_dot(dot, builder),
        }
    }
}
