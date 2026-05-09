use std::path::Path;

use inkwell::builder::Builder;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Linkage;
use inkwell::module::Module;
use inkwell::support::load_library_permanently;
use inkwell::values::*;
use inkwell::AddressSpace;

use crate::tensor::types::FloatType;

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

struct Routines<'ctx> {
    gemm: FunctionValue<'ctx>,
    gemm_batch_strided: Option<FunctionValue<'ctx>>,
    dot: FunctionValue<'ctx>,
    i32_ty: inkwell::types::IntType<'ctx>,
    fp_ty: inkwell::types::FloatType<'ctx>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    OpenBLAS,
    MKL,
}

impl Backend {
    pub fn shared_library_name(&self) -> &'static str {
        match self {
            Self::OpenBLAS => "libopenblas.so",
            Self::MKL => "libmkl_rt.so",
        }
    }
}

/// Try to load a BLAS shared lib (MKL preferred) and the matching OpenMP
/// runtime into the LLVM JIT global symbol space so kernel modules can resolve
/// `cblas_*` and `__kmpc_*` symbols. Returns the picked backend.
pub fn load_jit_runtime() -> Result<Backend, String> {
    let backend = [Backend::MKL, Backend::OpenBLAS]
        .into_iter()
        .find(|b| load_library_permanently(Path::new(b.shared_library_name())).is_ok())
        .ok_or_else(|| "no BLAS shared library found".to_string())?;

    // MKL uses libiomp5 internally; share the same runtime to avoid contention
    // between two OpenMP runtimes. Otherwise fall back to LLVM's libomp.
    let omp_lib = if backend == Backend::MKL {
        "libiomp5.so"
    } else {
        "libomp.so"
    };
    load_library_permanently(Path::new(omp_lib))
        .map_err(|e| format!("Failed to load {omp_lib}: {e:?}"))?;
    Ok(backend)
}

#[derive(Debug, Clone)]
pub struct GemmArgs<'ctx> {
    pub a: (PointerValue<'ctx>, bool),
    pub b: (PointerValue<'ctx>, bool),
    pub c: PointerValue<'ctx>,
    pub m: u64,
    pub n: u64,
    pub k: u64,

    #[allow(dead_code)]
    pub alpha: f64,
    #[allow(dead_code)]
    pub beta: f64,
}

pub struct BatchedGemmArgs<'ctx> {
    pub a: (PointerValue<'ctx>, bool),
    pub b: (PointerValue<'ctx>, bool),
    pub c: PointerValue<'ctx>,
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub alpha: f64,
    pub beta: f64,
    pub stride_a: u64,
    pub stride_b: u64,
    pub stride_c: u64,
    pub batch_count: u64,
}

pub struct DotArgs<'ctx> {
    pub x: (PointerValue<'ctx>, u64),
    pub y: (PointerValue<'ctx>, u64),
    pub n: u64,
}

impl<'ctx> Routines<'ctx> {
    fn new(ctx: &'ctx Context, module: &Module<'ctx>, fp_ty: FloatType, backend: Backend) -> Self {
        let prefix = match fp_ty {
            FloatType::F32 => 's',
            FloatType::F64 => 'd',
            FloatType::BF16 => unimplemented!("BF16 not supported on CPU backend"),
        };
        let fp_ty = fp_ty.llvm_type(ctx);

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
            format!("cblas_{}gemm", prefix).as_str(),
            gemm,
            Some(Linkage::External),
        );

        let gemm_batch_strided = if backend == Backend::MKL {
            let ty = void_ty.fn_type(
                &[
                    i32_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    fp_ty.into(),
                    ptr_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    ptr_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    fp_ty.into(),
                    ptr_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                    i32_ty.into(),
                ],
                false,
            );
            Some(module.add_function(
                format!("cblas_{}gemm_batch_strided", prefix).as_str(),
                ty,
                Some(Linkage::External),
            ))
        } else {
            None
        };

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
            format!("cblas_{}dot", prefix).as_str(),
            dot,
            Some(Linkage::External),
        );

        Self {
            gemm,
            gemm_batch_strided,
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
        let c_ptr = gemm.c;

        let m = self.i32_ty.const_int(gemm.m, false);
        let n = self.i32_ty.const_int(gemm.n, false);
        let k = self.i32_ty.const_int(gemm.k, false);
        let inc_a = if !a_trans { k } else { m };
        let inc_b = if !b_trans { n } else { k };

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
                self.fp_ty.const_float(gemm.alpha).into(),
                a_ptr.into(),
                inc_a.into(),
                b_ptr.into(),
                inc_b.into(),
                self.fp_ty.const_float(gemm.beta).into(),
                c_ptr.into(),
                n.into(),
            ],
            "",
        )
    }

    fn call_gemm_batch_strided(
        &self,
        gemm: &BatchedGemmArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let func = self
            .gemm_batch_strided
            .expect("batch_strided not available");
        let (a_ptr, a_trans) = gemm.a;
        let (b_ptr, b_trans) = gemm.b;
        let m = self.i32_ty.const_int(gemm.m, false);
        let n = self.i32_ty.const_int(gemm.n, false);
        let k = self.i32_ty.const_int(gemm.k, false);
        let lda = if !a_trans { k } else { m };
        let ldb = if !b_trans { n } else { k };
        builder.build_call(
            func,
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
                self.fp_ty.const_float(gemm.alpha).into(),
                a_ptr.into(),
                lda.into(),
                self.i32_ty.const_int(gemm.stride_a, false).into(),
                b_ptr.into(),
                ldb.into(),
                self.i32_ty.const_int(gemm.stride_b, false).into(),
                self.fp_ty.const_float(gemm.beta).into(),
                gemm.c.into(),
                n.into(),
                self.i32_ty.const_int(gemm.stride_c, false).into(),
                self.i32_ty.const_int(gemm.batch_count, false).into(),
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
    backend: Backend,
    s_routines: Routines<'ctx>,
    d_routines: Routines<'ctx>,
}

impl<'ctx> BLAS<'ctx> {
    pub fn new(ctx: &'ctx Context, module: &'_ Module<'ctx>, backend: Backend) -> Self {
        Self {
            backend,
            s_routines: Routines::new(ctx, module, crate::tensor::types::FloatType::F32, backend),
            d_routines: Routines::new(ctx, module, crate::tensor::types::FloatType::F64, backend),
        }
    }

    pub fn call_gemm(
        &self,
        ty: FloatType,
        gemm: &GemmArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        match ty {
            FloatType::F32 => self.s_routines.call_gemm(gemm, builder),
            FloatType::F64 => self.d_routines.call_gemm(gemm, builder),
            FloatType::BF16 => unimplemented!("BF16 not supported on CPU backend"),
        }
    }

    pub fn call_gemm_batch_strided(
        &self,
        ty: FloatType,
        gemm: &BatchedGemmArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        match ty {
            FloatType::F32 => self.s_routines.call_gemm_batch_strided(gemm, builder),
            FloatType::F64 => self.d_routines.call_gemm_batch_strided(gemm, builder),
            FloatType::BF16 => unimplemented!("BF16 not supported on CPU backend"),
        }
    }

    pub fn has_batch_strided(&self) -> bool {
        // NOTE: OpenBLAS also supports cblas_sgemm_batch_strided and cblas_dgemm_batch_strided,
        // but they didn't work for some reason (segmentation fault locally).
        self.backend == Backend::MKL
    }

    #[allow(dead_code)]
    pub fn call_dot(
        &self,
        ty: FloatType,
        dot: &DotArgs<'ctx>,
        builder: &Builder<'ctx>,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        match ty {
            FloatType::F32 => self.s_routines.call_dot(dot, builder),
            FloatType::F64 => self.d_routines.call_dot(dot, builder),
            FloatType::BF16 => unimplemented!("BF16 not supported on CPU backend"),
        }
    }
}
