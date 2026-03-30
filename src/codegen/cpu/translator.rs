use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::*;
use inkwell::values::*;
use itertools::izip;
use itertools::Itertools;
use smallvec::smallvec;

use crate::codegen::cpu::blas::*;
use crate::codegen::cpu::llvm::*;
use crate::codegen::cpu::omp::*;
use crate::codegen::cpu::op::*;
use crate::onnx::operator;
use crate::onnx::operator::ReinterpretType;
use crate::schedule::ElementwiseOpArg;
use crate::tensor::data::ScalarData;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

#[derive(Clone)]
pub struct FunctionTranslator<'a, 'ctx> {
    pub context: &'ctx Context,
    pub module: &'a Module<'ctx>,
    pub builder: &'a Builder<'ctx>,
    pub func: &'a FunctionValue<'ctx>,
    pub intrinsics: &'a Intrinsics<'ctx>,
    pub blas: &'a BLAS<'a>,
    pub omp: &'a OMP<'ctx>,

    #[allow(dead_code)]
    pub debug_stuff: &'a DebugStuff<'ctx>,
}

impl<'ctx> FunctionTranslator<'_, 'ctx> {
    fn build_gep(&self, ptr: &TensorPtr<'ctx>) -> Result<PointerValue<'ctx>, BuilderError> {
        unsafe {
            self.builder.build_in_bounds_gep(
                ptr.ty.elem_type.llvm_type(self.context),
                ptr.ptr,
                &[ptr.offset],
                format!("gep.{}", ptr.name).as_str(),
            )
        }
    }

    fn build_load(&self, ptr: &TensorPtr<'ctx>) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let gep = self.build_gep(ptr)?;
        self.builder.build_load(
            ptr.ty.elem_type.llvm_type(self.context),
            gep,
            format!("load.{}", ptr.name).as_str(),
        )
    }

    fn build_store<V: BasicValue<'ctx>>(
        &self,
        ptr: &TensorPtr<'ctx>,
        val: V,
    ) -> Result<(), BuilderError> {
        let gep = self.build_gep(ptr)?;
        self.builder.build_store(gep, val).map(|_| ())
    }

    fn build_raw_load<T: BasicType<'ctx> + Copy>(
        &self,
        ty: T,
        ptr: PointerValue<'ctx>,
        offset: IntValue<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let gep = unsafe { self.builder.build_in_bounds_gep(ty, ptr, &[offset], "gep") }?;
        self.builder.build_load(ty, gep, "load")
    }

    fn build_raw_store<T: BasicType<'ctx>, V: BasicValue<'ctx>>(
        &self,
        ty: T,
        ptr: PointerValue<'ctx>,
        offset: IntValue<'ctx>,
        val: V,
    ) -> Result<(), BuilderError> {
        let gep = unsafe { self.builder.build_in_bounds_gep(ty, ptr, &[offset], "gep") }?;
        self.builder.build_store(gep, val).map(|_| ())
    }

    fn build_tail_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let call = self.builder.build_call(function, args, name)?;
        //call.set_tail_call(true);
        Ok(call)
    }

    fn build_single_op(
        &self,
        opcode: SingleOpcode,
        ty: DataType,
        operands: &[BasicValueEnum<'ctx>],
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        macro_rules! unary_op {
            ($ops: expr) => {{
                assert!(operands.len() == 1);
                operands[0]
            }};
        }

        macro_rules! binary_op {
            ($ops: expr) => {{
                assert!(operands.len() == 2);
                (operands[0], operands[1])
            }};
        }

        let res = match opcode {
            opcode @ (SingleOpcode::Add |
            SingleOpcode::Div |
            SingleOpcode::Mul |
            SingleOpcode::Sub) => {
                macro_rules! body {
                    ($into: ident, $arith: ident) => {{
                        let (lhs, rhs) = binary_op!(operands);
                        let lhs = lhs.$into();
                        let rhs = rhs.$into();
                        self.builder.$arith(lhs, rhs, "res")?.as_basic_value_enum()
                    }};
                }

                let is_float = matches!(ty, DataType::Float(_));

                match (opcode, is_float) {
                    (SingleOpcode::Add, true) => {
                        body!(into_float_value, build_float_add)
                    }
                    (SingleOpcode::Div, true) => {
                        body!(into_float_value, build_float_div)
                    }
                    (SingleOpcode::Mul, true) => {
                        body!(into_float_value, build_float_mul)
                    }
                    (SingleOpcode::Sub, true) => {
                        body!(into_float_value, build_float_sub)
                    }
                    (SingleOpcode::Add, false) => {
                        body!(into_int_value, build_int_add)
                    }
                    (SingleOpcode::Div, false) => {
                        // TODO: signed or unsigned?
                        body!(into_int_value, build_int_signed_div)
                    }
                    (SingleOpcode::Mul, false) => {
                        body!(into_int_value, build_int_mul)
                    }
                    (SingleOpcode::Sub, false) => {
                        body!(into_int_value, build_int_sub)
                    }
                    _ => unreachable!(),
                }
            }

            SingleOpcode::BatchNorm(operator::BatchNormalization { epsilon, .. }) => {
                let src = operands[operator::args::BATCHNORM_DATA].into_float_value();
                let scale = operands[operator::args::BATCHNORM_SCALE].into_float_value();
                let bias = operands[operator::args::BATCHNORM_BIAS].into_float_value();
                let mean = operands[operator::args::BATCHNORM_MEAN].into_float_value();
                let variance = operands[operator::args::BATCHNORM_VAR].into_float_value();
                let fp_type = ty.float_type().unwrap();
                let sqrt = self.intrinsics.sqrt.get(fp_type);
                let fma = self.intrinsics.fma.get(fp_type);
                let fp_type = fp_type.llvm_type(self.context);
                let epsilon = fp_type.const_float(epsilon as f64);

                let factor = self.builder.build_float_add(variance, epsilon, "factor")?;
                let factor = self
                    .build_tail_call(sqrt, &[factor.into()], "factor")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                let factor = self.builder.build_float_div(scale, factor, "factor")?;
                let val = self.builder.build_float_sub(src, mean, "val.sub.mean")?;
                self.build_tail_call(fma, &[val.into(), factor.into(), bias.into()], "val")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }

            SingleOpcode::Cast(from, to) => {
                let src = unary_op!(operands);

                // Cast from source type to target type
                match (from, to) {
                    // Same type, no conversion needed
                    (DataType::SInt(sfrom), DataType::SInt(sto)) if sfrom == sto => src,
                    (DataType::UInt(ufrom), DataType::UInt(uto)) if ufrom == uto => src,
                    (DataType::Float(ffrom), DataType::Float(fto)) if ffrom == fto => src,

                    // Integer to integer casts
                    (DataType::SInt(_), DataType::SInt(sto)) => self
                        .builder
                        .build_int_cast(src.into_int_value(), sto.llvm_type(self.context), "cast")?
                        .as_basic_value_enum(),
                    (DataType::UInt(_), DataType::UInt(uto)) => self
                        .builder
                        .build_int_cast(src.into_int_value(), uto.llvm_type(self.context), "cast")?
                        .as_basic_value_enum(),
                    (DataType::SInt(_), DataType::UInt(uto)) => self
                        .builder
                        .build_int_cast(src.into_int_value(), uto.llvm_type(self.context), "cast")?
                        .as_basic_value_enum(),
                    (DataType::UInt(_), DataType::SInt(sto)) => self
                        .builder
                        .build_int_cast(src.into_int_value(), sto.llvm_type(self.context), "cast")?
                        .as_basic_value_enum(),

                    // Float to float casts
                    (DataType::Float(ffrom), DataType::Float(fto)) => {
                        if ffrom.bit_width() < fto.bit_width() {
                            self.builder
                                .build_float_cast(
                                    src.into_float_value(),
                                    fto.llvm_type(self.context),
                                    "cast",
                                )?
                                .as_basic_value_enum()
                        } else {
                            self.builder
                                .build_float_trunc(
                                    src.into_float_value(),
                                    fto.llvm_type(self.context),
                                    "cast",
                                )?
                                .as_basic_value_enum()
                        }
                    }

                    // Integer to float casts
                    (DataType::SInt(_), DataType::Float(fto)) => self
                        .builder
                        .build_signed_int_to_float(
                            src.into_int_value(),
                            fto.llvm_type(self.context),
                            "cast",
                        )?
                        .as_basic_value_enum(),
                    (DataType::UInt(_), DataType::Float(fto)) => self
                        .builder
                        .build_unsigned_int_to_float(
                            src.into_int_value(),
                            fto.llvm_type(self.context),
                            "cast",
                        )?
                        .as_basic_value_enum(),

                    // Float to integer casts
                    (DataType::Float(_), DataType::SInt(sto)) => self
                        .builder
                        .build_float_to_signed_int(
                            src.into_float_value(),
                            sto.llvm_type(self.context),
                            "cast",
                        )?
                        .as_basic_value_enum(),
                    (DataType::Float(_), DataType::UInt(uto)) => self
                        .builder
                        .build_float_to_unsigned_int(
                            src.into_float_value(),
                            uto.llvm_type(self.context),
                            "cast",
                        )?
                        .as_basic_value_enum(),
                }
            }

            SingleOpcode::GeLU(operator::GeLU { approximate }) => {
                if !approximate {
                    unimplemented!()
                }
                let ty = ty.float_type().unwrap();
                let tanh = self.intrinsics.tanh.get(ty);
                let fma = self.intrinsics.fma.get(ty);
                let ty = ty.llvm_type(self.context);

                // x * (0.5 + 0.5 * tanh(x * (sqrt(2/pi) + 0.044715*sqrt(2/pi)*x^2)))
                let a = ty.const_float(0.5);
                let b = ty.const_float(0.7978845608028654); // sqrt(2/pi)
                let c = ty.const_float(0.044715 * 0.7978845608028654); // 0.044715*sqrt(2/pi)

                let x = unary_op!(operands).into_float_value();
                let val = self.builder.build_float_mul(x, x, "x.sq")?;
                let val = self
                    .build_tail_call(fma, &[c.into(), val.into(), b.into()], "val")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                let val = self.builder.build_float_mul(x, val, "x.mul")?;
                let val = self
                    .build_tail_call(tanh, &[val.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                let val = self
                    .build_tail_call(fma, &[a.into(), val.into(), a.into()], "val")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                self.builder
                    .build_float_mul(x, val, "res")?
                    .as_basic_value_enum()
            }

            opcode @ (SingleOpcode::Exp | SingleOpcode::Log | SingleOpcode::Sqrt) => {
                let src = unary_op!(operands);
                let ty = ty.float_type().unwrap();
                let f = match opcode {
                    SingleOpcode::Exp => self.intrinsics.exp.get(ty),
                    SingleOpcode::Log => self.intrinsics.log.get(ty),
                    SingleOpcode::Sqrt => self.intrinsics.sqrt.get(ty),
                    _ => unreachable!(),
                };
                let src = src.into_float_value();
                self.build_tail_call(f, &[src.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }

            SingleOpcode::LeakyReLU(operator::LeakyReLU { alpha }) => {
                let ty = ty.float_type().unwrap().llvm_type(self.context);
                let zero = ty.const_zero();
                let src = unary_op!(operands).into_float_value();
                let lt = self.builder.build_float_compare(
                    inkwell::FloatPredicate::OLT,
                    src,
                    zero,
                    "lt",
                )?;
                let lhs = self
                    .builder
                    .build_float_mul(src, ty.const_float(alpha), "lhs")?;
                let rhs = src;
                self.builder.build_select(lt, lhs, rhs, "res")?
            }

            // TODO: Improve implmentation.
            SingleOpcode::Pow(base_ty, exp_ty) => {
                let (lhs, rhs) = binary_op!(operands);
                let cast_ty = {
                    let width = base_ty.bit_width().max(exp_ty.bit_width());
                    if width <= 32 {
                        FloatType::F32
                    } else {
                        assert!(width <= 64);
                        FloatType::F64
                    }
                };
                let llvm_cast_ty = cast_ty.llvm_type(self.context);
                let convert_op = |v: BasicValueEnum<'ctx>, ty: DataType| match ty {
                    DataType::SInt(_) => {
                        self.builder
                            .build_signed_int_to_float(v.into_int_value(), llvm_cast_ty, "")
                    }
                    DataType::UInt(_) => self.builder.build_unsigned_int_to_float(
                        v.into_int_value(),
                        llvm_cast_ty,
                        "",
                    ),
                    DataType::Float(fty) => {
                        if fty.bit_width() == cast_ty.bit_width() {
                            Ok(v.into_float_value())
                        } else {
                            assert!(fty.bit_width() < cast_ty.bit_width());
                            self.builder
                                .build_float_cast(v.into_float_value(), llvm_cast_ty, "")
                        }
                    }
                };

                let lhs = convert_op(lhs, base_ty)?;
                let rhs = convert_op(rhs, exp_ty)?;
                let pow = self.intrinsics.pow.get(cast_ty);
                let res = self
                    .build_tail_call(pow, &[lhs.into(), rhs.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap();

                match base_ty {
                    DataType::SInt(sty) => self
                        .builder
                        .build_float_to_signed_int(
                            res.into_float_value(),
                            sty.llvm_type(self.context),
                            "res",
                        )?
                        .as_basic_value_enum(),
                    DataType::UInt(uty) => self
                        .builder
                        .build_float_to_unsigned_int(
                            res.into_float_value(),
                            uty.llvm_type(self.context),
                            "res",
                        )?
                        .as_basic_value_enum(),
                    DataType::Float(fty) => {
                        if fty == cast_ty {
                            res.as_basic_value_enum()
                        } else {
                            self.builder
                                .build_float_trunc(
                                    res.into_float_value(),
                                    fty.llvm_type(self.context),
                                    "res",
                                )?
                                .as_basic_value_enum()
                        }
                    }
                }
            }

            SingleOpcode::Reciprocal => {
                let ty = ty.float_type().unwrap();
                let src = unary_op!(operands).into_float_value();
                let one = ty.llvm_type(self.context).const_float(1.0);
                // TODO: Add `arcp` flag
                self.builder.build_float_div(one, src, "res")?.into()
            }

            SingleOpcode::ReLU => {
                let ty = ty.float_type().unwrap();
                let fmax = self.intrinsics.fmax.get(ty);
                let ty = ty.llvm_type(self.context);
                let zero = ty.const_zero();
                let src = unary_op!(operands).into_float_value();
                self.build_tail_call(fmax, &[src.into(), zero.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }

            SingleOpcode::Sigmoid => {
                let ty = ty.float_type().unwrap();
                let exp = self.intrinsics.exp.get(ty);
                let ty = ty.llvm_type(self.context);
                let src = unary_op!(operands).into_float_value();
                let src = self.builder.build_float_neg(src, "neg")?;
                let exp = self
                    .build_tail_call(exp, &[src.into()], "exp")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_float_value();
                let one = ty.const_float(1.0);
                let den = self.builder.build_float_add(one, exp, "den")?;
                self.builder.build_float_div(one, den, "res")?.into()
            }

            SingleOpcode::Tanh => {
                let src = unary_op!(operands).into_float_value();
                let tanh = self.intrinsics.tanh.get(ty.float_type().unwrap());
                self.build_tail_call(tanh, &[src.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }

            SingleOpcode::Transfer => unary_op!(operands),
        };

        Ok(res)
    }

    fn build_operation(&self, op: &Operation<'ctx>) -> Result<(), BuilderError> {
        let ty = op.result_type();
        let res = match op.opcode {
            Opcode::Single(opcode) => {
                let operands = op
                    .src_operands()
                    .iter()
                    .map(|op| self.build_load(op))
                    .collect::<Result<Vec<_>, _>>()?;
                self.build_single_op(opcode, ty, &operands)?
            }
            Opcode::Fused(ref ops) => {
                let mut intermediates = Vec::with_capacity(ops.len());
                let srcs = op.src_operands();
                let ty = op.result_type();
                for (opcode, args) in ops.iter() {
                    let opcode = *opcode;
                    let operands = args
                        .iter()
                        .map(|arg| match arg {
                            ElementwiseOpArg::Input(i) => self.build_load(&srcs[*i]),
                            ElementwiseOpArg::NthResult(i) => Ok(intermediates[*i]),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let res = self.build_single_op(opcode, ty, &operands)?;
                    intermediates.push(res);
                }
                *intermediates.last().unwrap()
            }
        };
        self.build_store(op.dst_operand(), res)
    }

    pub fn build_im2col(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        im2col: &operator::Im2Col,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        // NCHW: src[N, C_in, H_in, W_in], NHWC: src[N, H_in, W_in, C_in]
        // Output: dst[N * H_out * W_out, C_in * kH * kW]
        //
        // for n in 0..N:
        //   for oh in 0..H_out:
        //     for ow in 0..W_out:
        //       for kh in 0..kH:
        //         for kw in 0..kW:
        //           for c in 0..C_in:
        //             ih = oh * stride_h + kh * dilation_h - pad_h
        //             iw = ow * stride_w + kw * dilation_w - pad_w
        //             row = n * H_out * W_out + oh * W_out + ow
        //             col = c * kH * kW + kh * kW + kw
        //             if oob(ih, iw):
        //               dst[row, col] = pad_val
        //             else:
        //               dst[row, col] = src[n, c, ih, iw]  (NCHW)
        //                            or src[n, ih, iw, c]  (NHWC)
        assert_eq!(im2col.one_fm_shape.ndim(), 2);

        let layout = im2col.layout;
        let i64_ty = self.context.i64_type();
        let elem_ty = src.ty.elem_type.llvm_type(self.context);

        let nbatch = im2col.nbatch as u64;
        let h_out = im2col.one_fm_shape[0] as u64;
        let w_out = im2col.one_fm_shape[1] as u64;
        let kh = im2col.one_kernel_shape[0] as u64;
        let kw = im2col.one_kernel_shape[1] as u64;
        let c_in = im2col.channel.inner() as u64;
        let (h_in, w_in) = match layout {
            operator::Layout::NCHW => (src.ty.dims[2] as u64, src.ty.dims[3] as u64),
            operator::Layout::NHWC => (src.ty.dims[1] as u64, src.ty.dims[2] as u64),
        };
        let stride_h = im2col.strides[0] as u64;
        let stride_w = im2col.strides[1] as u64;
        let dilation_h = im2col.dilations[0] as u64;
        let dilation_w = im2col.dilations[1] as u64;
        let calc_pad = |dim: usize| -> u64 {
            match &im2col.pad {
                operator::ConvPad::NotSet(pad) => pad[dim].0 as u64,
                operator::ConvPad::Valid => 0,
                operator::ConvPad::SameLower | operator::ConvPad::SameUpper => {
                    let padded_len = im2col.padded_len(dim) as u64;
                    let src_dim = match layout {
                        operator::Layout::NCHW => src.ty.dims[dim + 2] as u64,
                        operator::Layout::NHWC => src.ty.dims[dim + 1] as u64,
                    };
                    let pad_len = padded_len - src_dim;
                    let pad_left = pad_len / 2;
                    let add_left = pad_len % 2 == 1 && matches!(im2col.pad, operator::ConvPad::SameLower);
                    pad_left + add_left as u64
                }
            }
        };
        let pad_h = calc_pad(0);
        let pad_w = calc_pad(1);

        let pad_val = match (src.ty.elem_type, im2col.pad_val) {
            (_, operator::PadVal::Zero) => elem_ty.const_zero(),
            (DataType::Float(ft), operator::PadVal::NInf) => {
                let v = match ft {
                    FloatType::F32 => f32::NEG_INFINITY as f64,
                    FloatType::F64 => f64::NEG_INFINITY,
                };
                ft.llvm_type(self.context)
                    .const_float(v)
                    .as_basic_value_enum()
            }
            _ => unimplemented!(),
        };

        let dst_cols = c_in * kh * kw;

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);
        let hdr_n = bb("nhwc.n.hdr");
        let hdr_oh = bb("nhwc.oh.hdr");
        let hdr_ow = bb("nhwc.ow.hdr");
        let hdr_kh = bb("nhwc.kh.hdr");
        let hdr_kw = bb("nhwc.kw.hdr");
        let hdr_c = bb("nhwc.c.hdr");
        let body = bb("nhwc.body");
        let latch_c = bb("nhwc.c.latch");
        let latch_kw = bb("nhwc.kw.latch");
        let latch_kh = bb("nhwc.kh.latch");
        let latch_ow = bb("nhwc.ow.latch");
        let latch_oh = bb("nhwc.oh.latch");
        let latch_n = bb("nhwc.n.latch");
        let exit = bb("nhwc.exit");

        macro_rules! phi {
            ($bb:expr) => {{
                self.builder.position_at_end($bb);
                let p = self.builder.build_phi(i64_ty, "ind")?;
                (p, p.as_basic_value().into_int_value())
            }};
        }
        macro_rules! latch {
            ($phi:expr, $prev_bb:expr, $val:expr, $bound:expr, $next:expr, $exit:expr, $latch_bb:expr) => {{
                self.builder.position_at_end($latch_bb);
                let next_val =
                    self.builder
                        .build_int_add($val, i64_ty.const_int(1, false), "next")?;
                let ec = self.builder.build_int_compare(
                    inkwell::IntPredicate::EQ,
                    next_val,
                    i64_ty.const_int($bound, false),
                    "ec",
                )?;
                self.builder.build_conditional_branch(ec, $exit, $next)?;
                $phi.add_incoming(&[(&i64_ty.const_zero(), $prev_bb), (&next_val, $latch_bb)]);
            }};
        }

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(hdr_n)?;

        let (phi_n, ind_n) = phi!(hdr_n);
        self.builder.build_unconditional_branch(hdr_oh)?;

        let (phi_oh, ind_oh) = phi!(hdr_oh);
        self.builder.build_unconditional_branch(hdr_ow)?;

        let (phi_ow, ind_ow) = phi!(hdr_ow);
        self.builder.build_unconditional_branch(hdr_kh)?;

        let (phi_kh, ind_kh) = phi!(hdr_kh);
        self.builder.build_unconditional_branch(hdr_kw)?;

        let (phi_kw, ind_kw) = phi!(hdr_kw);
        self.builder.build_unconditional_branch(hdr_c)?;

        let (phi_c, ind_c) = phi!(hdr_c);
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);

        let ih = {
            let a = self.builder.build_int_mul(ind_oh, i64_ty.const_int(stride_h, false), "oh_s")?;
            let b = self.builder.build_int_mul(ind_kh, i64_ty.const_int(dilation_h, false), "kh_d")?;
            let c = self.builder.build_int_add(a, b, "ih_raw")?;
            self.builder.build_int_sub(c, i64_ty.const_int(pad_h, false), "ih")?
        };
        let iw = {
            let a = self.builder.build_int_mul(ind_ow, i64_ty.const_int(stride_w, false), "ow_s")?;
            let b = self.builder.build_int_mul(ind_kw, i64_ty.const_int(dilation_w, false), "kw_d")?;
            let c = self.builder.build_int_add(a, b, "iw_raw")?;
            self.builder.build_int_sub(c, i64_ty.const_int(pad_w, false), "iw")?
        };

        let oob = {
            let ih_neg = self.builder.build_int_compare(inkwell::IntPredicate::SLT, ih, i64_ty.const_zero(), "ih_neg")?;
            let ih_big = self.builder.build_int_compare(inkwell::IntPredicate::SGE, ih, i64_ty.const_int(h_in, false), "ih_big")?;
            let iw_neg = self.builder.build_int_compare(inkwell::IntPredicate::SLT, iw, i64_ty.const_zero(), "iw_neg")?;
            let iw_big = self.builder.build_int_compare(inkwell::IntPredicate::SGE, iw, i64_ty.const_int(w_in, false), "iw_big")?;
            let a = self.builder.build_or(ih_neg, ih_big, "oob_h")?;
            let b = self.builder.build_or(iw_neg, iw_big, "oob_w")?;
            self.builder.build_or(a, b, "oob")?
        };

        let bb_load = bb("nhwc.load");
        let bb_pad = bb("nhwc.pad");
        let bb_store = bb("nhwc.store");
        self.builder.build_conditional_branch(oob, bb_pad, bb_load)?;

        self.builder.position_at_end(bb_load);
        let src_offset = match layout {
            operator::Layout::NCHW => {
                let o = self.builder.build_int_mul(ind_n, i64_ty.const_int(c_in * h_in * w_in, false), "so_n")?;
                let o = self.builder.build_int_add(o, self.builder.build_int_mul(ind_c, i64_ty.const_int(h_in * w_in, false), "so_c")?, "so_nc")?;
                let o = self.builder.build_int_add(o, self.builder.build_int_mul(ih, i64_ty.const_int(w_in, false), "so_h")?, "so_nch")?;
                self.builder.build_int_add(o, iw, "src_off")?
            }
            operator::Layout::NHWC => {
                let o = self.builder.build_int_mul(ind_n, i64_ty.const_int(h_in * w_in * c_in, false), "so_n")?;
                let o = self.builder.build_int_add(o, self.builder.build_int_mul(ih, i64_ty.const_int(w_in * c_in, false), "so_h")?, "so_nh")?;
                let o = self.builder.build_int_add(o, self.builder.build_int_mul(iw, i64_ty.const_int(c_in, false), "so_w")?, "so_nhw")?;
                self.builder.build_int_add(o, ind_c, "src_off")?
            }
        };
        let src_val = self.build_load(&src.clone().set_offset(src_offset))?.into_float_value();
        self.builder.build_unconditional_branch(bb_store)?;

        self.builder.position_at_end(bb_pad);
        self.builder.build_unconditional_branch(bb_store)?;

        self.builder.position_at_end(bb_store);
        let val = self.builder.build_phi(elem_ty, "val")?;
        val.add_incoming(&[(&src_val, bb_load), (&pad_val, bb_pad)]);

        let dst_row = {
            let o = self.builder.build_int_mul(ind_n, i64_ty.const_int(h_out * w_out, false), "dr_n")?;
            let o = self.builder.build_int_add(o, self.builder.build_int_mul(ind_oh, i64_ty.const_int(w_out, false), "dr_oh")?, "dr_noh")?;
            self.builder.build_int_add(o, ind_ow, "dst_row")?
        };
        let dst_col = {
            let o = self.builder.build_int_mul(ind_c, i64_ty.const_int(kh * kw, false), "dc_c")?;
            let o = self.builder.build_int_add(o, self.builder.build_int_mul(ind_kh, i64_ty.const_int(kw, false), "dc_kh")?, "dc_ckh")?;
            self.builder.build_int_add(o, ind_kw, "dst_col")?
        };
        let dst_offset = {
            let o = self.builder.build_int_mul(dst_row, i64_ty.const_int(dst_cols, false), "do_row")?;
            self.builder.build_int_add(o, dst_col, "dst_off")?
        };
        self.build_store(&dst.clone().set_offset(dst_offset), val.as_basic_value())?;
        self.builder.build_unconditional_branch(latch_c)?;

        latch!(phi_c, hdr_kw, ind_c, c_in, hdr_c, latch_kw, latch_c);
        latch!(phi_kw, hdr_kh, ind_kw, kw, hdr_kw, latch_kh, latch_kw);
        latch!(phi_kh, hdr_ow, ind_kh, kh, hdr_kh, latch_ow, latch_kh);
        latch!(phi_ow, hdr_oh, ind_ow, w_out, hdr_ow, latch_oh, latch_ow);
        latch!(phi_oh, hdr_n, ind_oh, h_out, hdr_oh, latch_n, latch_oh);
        latch!(phi_n, entry, ind_n, nbatch, hdr_n, exit, latch_n);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_im2col_split(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        im2col: &operator::Im2Col,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        // src[N, C, H_in, W_in] (NCHW, Channel::Split)
        // dst[N * C * H_out * W_out, kH * kW]
        //
        // for n in 0..N:
        //   for c in 0..C:
        //     for oh in 0..H_out:
        //       for ow in 0..W_out:
        //         for kh in 0..kH:
        //           for kw in 0..kW:
        //             ih = oh * stride_h + kh * dilation_h - pad_h
        //             iw = ow * stride_w + kw * dilation_w - pad_w
        //             row = n*C*H_out*W_out + c*H_out*W_out + oh*W_out + ow
        //             col = kh * kW + kw
        //             if oob(ih, iw):
        //               dst[row, col] = pad_val
        //             else:
        //               dst[row, col] = src[n, c, ih, iw]
        assert_eq!(im2col.one_fm_shape.ndim(), 2);
        assert!(matches!(im2col.channel, operator::Channel::Split(_)));
        assert!(matches!(im2col.layout, operator::Layout::NCHW));

        let i64_ty = self.context.i64_type();
        let elem_ty = src.ty.elem_type.llvm_type(self.context);

        let nbatch = im2col.nbatch as u64;
        let h_out = im2col.one_fm_shape[0] as u64;
        let w_out = im2col.one_fm_shape[1] as u64;
        let kh = im2col.one_kernel_shape[0] as u64;
        let kw = im2col.one_kernel_shape[1] as u64;
        let c_in = im2col.channel.inner() as u64;
        let h_in = src.ty.dims[2] as u64;
        let w_in = src.ty.dims[3] as u64;
        let stride_h = im2col.strides[0] as u64;
        let stride_w = im2col.strides[1] as u64;
        let dilation_h = im2col.dilations[0] as u64;
        let dilation_w = im2col.dilations[1] as u64;
        let calc_pad = |dim: usize| -> u64 {
            match &im2col.pad {
                operator::ConvPad::NotSet(pad) => pad[dim].0 as u64,
                operator::ConvPad::Valid => 0,
                operator::ConvPad::SameLower | operator::ConvPad::SameUpper => {
                    let padded_len = im2col.padded_len(dim) as u64;
                    let src_dim = src.ty.dims[dim + 2] as u64;
                    let pad_len = padded_len - src_dim;
                    let pad_left = pad_len / 2;
                    let add_left = pad_len % 2 == 1 && matches!(im2col.pad, operator::ConvPad::SameLower);
                    pad_left + add_left as u64
                }
            }
        };
        let pad_h = calc_pad(0);
        let pad_w = calc_pad(1);

        let pad_val = match (src.ty.elem_type, im2col.pad_val) {
            (_, operator::PadVal::Zero) => elem_ty.const_zero(),
            (DataType::Float(ft), operator::PadVal::NInf) => {
                let v = match ft {
                    FloatType::F32 => f32::NEG_INFINITY as f64,
                    FloatType::F64 => f64::NEG_INFINITY,
                };
                ft.llvm_type(self.context)
                    .const_float(v)
                    .as_basic_value_enum()
            }
            _ => unimplemented!(),
        };

        let dst_cols = kh * kw;

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);
        let hdr_n = bb("split.n.hdr");
        let hdr_c = bb("split.c.hdr");
        let hdr_oh = bb("split.oh.hdr");
        let hdr_ow = bb("split.ow.hdr");
        let hdr_kh = bb("split.kh.hdr");
        let hdr_kw = bb("split.kw.hdr");
        let body = bb("split.body");
        let latch_kw = bb("split.kw.latch");
        let latch_kh = bb("split.kh.latch");
        let latch_ow = bb("split.ow.latch");
        let latch_oh = bb("split.oh.latch");
        let latch_c = bb("split.c.latch");
        let latch_n = bb("split.n.latch");
        let exit = bb("split.exit");

        macro_rules! phi {
            ($bb:expr) => {{
                self.builder.position_at_end($bb);
                let p = self.builder.build_phi(i64_ty, "ind")?;
                (p, p.as_basic_value().into_int_value())
            }};
        }
        macro_rules! latch {
            ($phi:expr, $prev_bb:expr, $val:expr, $bound:expr, $next:expr, $exit:expr, $latch_bb:expr) => {{
                self.builder.position_at_end($latch_bb);
                let next_val =
                    self.builder
                        .build_int_add($val, i64_ty.const_int(1, false), "next")?;
                let ec = self.builder.build_int_compare(
                    inkwell::IntPredicate::EQ,
                    next_val,
                    i64_ty.const_int($bound, false),
                    "ec",
                )?;
                self.builder.build_conditional_branch(ec, $exit, $next)?;
                $phi.add_incoming(&[(&i64_ty.const_zero(), $prev_bb), (&next_val, $latch_bb)]);
            }};
        }

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(hdr_n)?;

        let (phi_n, ind_n) = phi!(hdr_n);
        self.builder.build_unconditional_branch(hdr_c)?;

        let (phi_c, ind_c) = phi!(hdr_c);
        self.builder.build_unconditional_branch(hdr_oh)?;

        let (phi_oh, ind_oh) = phi!(hdr_oh);
        self.builder.build_unconditional_branch(hdr_ow)?;

        let (phi_ow, ind_ow) = phi!(hdr_ow);
        self.builder.build_unconditional_branch(hdr_kh)?;

        let (phi_kh, ind_kh) = phi!(hdr_kh);
        self.builder.build_unconditional_branch(hdr_kw)?;

        let (phi_kw, ind_kw) = phi!(hdr_kw);
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);

        let ih = {
            let a = self.builder.build_int_mul(ind_oh, i64_ty.const_int(stride_h, false), "oh_s")?;
            let b = self.builder.build_int_mul(ind_kh, i64_ty.const_int(dilation_h, false), "kh_d")?;
            let c = self.builder.build_int_add(a, b, "ih_raw")?;
            self.builder.build_int_sub(c, i64_ty.const_int(pad_h, false), "ih")?
        };
        let iw = {
            let a = self.builder.build_int_mul(ind_ow, i64_ty.const_int(stride_w, false), "ow_s")?;
            let b = self.builder.build_int_mul(ind_kw, i64_ty.const_int(dilation_w, false), "kw_d")?;
            let c = self.builder.build_int_add(a, b, "iw_raw")?;
            self.builder.build_int_sub(c, i64_ty.const_int(pad_w, false), "iw")?
        };

        let oob = {
            let ih_neg = self.builder.build_int_compare(inkwell::IntPredicate::SLT, ih, i64_ty.const_zero(), "ih_neg")?;
            let ih_big = self.builder.build_int_compare(inkwell::IntPredicate::SGE, ih, i64_ty.const_int(h_in, false), "ih_big")?;
            let iw_neg = self.builder.build_int_compare(inkwell::IntPredicate::SLT, iw, i64_ty.const_zero(), "iw_neg")?;
            let iw_big = self.builder.build_int_compare(inkwell::IntPredicate::SGE, iw, i64_ty.const_int(w_in, false), "iw_big")?;
            let a = self.builder.build_or(ih_neg, ih_big, "oob_h")?;
            let b = self.builder.build_or(iw_neg, iw_big, "oob_w")?;
            self.builder.build_or(a, b, "oob")?
        };

        let bb_load = bb("split.load");
        let bb_pad = bb("split.pad");
        let bb_store = bb("split.store");
        self.builder.build_conditional_branch(oob, bb_pad, bb_load)?;

        self.builder.position_at_end(bb_load);
        let src_offset = {
            let o = self.builder.build_int_mul(ind_n, i64_ty.const_int(c_in * h_in * w_in, false), "so_n")?;
            let o = self.builder.build_int_add(o, self.builder.build_int_mul(ind_c, i64_ty.const_int(h_in * w_in, false), "so_c")?, "so_nc")?;
            let o = self.builder.build_int_add(o, self.builder.build_int_mul(ih, i64_ty.const_int(w_in, false), "so_h")?, "so_nch")?;
            self.builder.build_int_add(o, iw, "src_off")?
        };
        let src_val = self.build_load(&src.clone().set_offset(src_offset))?.into_float_value();
        self.builder.build_unconditional_branch(bb_store)?;

        self.builder.position_at_end(bb_pad);
        self.builder.build_unconditional_branch(bb_store)?;

        self.builder.position_at_end(bb_store);
        let val = self.builder.build_phi(elem_ty, "val")?;
        val.add_incoming(&[(&src_val, bb_load), (&pad_val, bb_pad)]);

        let dst_row = {
            let o = self.builder.build_int_mul(ind_n, i64_ty.const_int(c_in * h_out * w_out, false), "dr_n")?;
            let o = self.builder.build_int_add(o, self.builder.build_int_mul(ind_c, i64_ty.const_int(h_out * w_out, false), "dr_c")?, "dr_nc")?;
            let o = self.builder.build_int_add(o, self.builder.build_int_mul(ind_oh, i64_ty.const_int(w_out, false), "dr_oh")?, "dr_ncoh")?;
            self.builder.build_int_add(o, ind_ow, "dst_row")?
        };
        let dst_col = {
            let o = self.builder.build_int_mul(ind_kh, i64_ty.const_int(kw, false), "dc_kh")?;
            self.builder.build_int_add(o, ind_kw, "dst_col")?
        };
        let dst_offset = {
            let o = self.builder.build_int_mul(dst_row, i64_ty.const_int(dst_cols, false), "do_row")?;
            self.builder.build_int_add(o, dst_col, "dst_off")?
        };
        self.build_store(&dst.clone().set_offset(dst_offset), val.as_basic_value())?;
        self.builder.build_unconditional_branch(latch_kw)?;

        latch!(phi_kw, hdr_kh, ind_kw, kw, hdr_kw, latch_kh, latch_kw);
        latch!(phi_kh, hdr_ow, ind_kh, kh, hdr_kh, latch_ow, latch_kh);
        latch!(phi_ow, hdr_oh, ind_ow, w_out, hdr_ow, latch_oh, latch_ow);
        latch!(phi_oh, hdr_c, ind_oh, h_out, hdr_oh, latch_c, latch_oh);
        latch!(phi_c, hdr_n, ind_c, c_in, hdr_c, latch_n, latch_c);
        latch!(phi_n, entry, ind_n, nbatch, hdr_n, exit, latch_n);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_gemm(
        &self,
        dst: &TensorPtr<'ctx>,
        a: &TensorPtr<'ctx>,
        b: &TensorPtr<'ctx>,
        c: Option<&TensorPtr<'ctx>>,
        entry: BasicBlock<'ctx>,
        gemm: &operator::Gemm,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        // Copy C into output buffer so BLAS can compute alpha*A*B + beta*C in-place.
        let entry = if let Some(c) = c {
            assert_eq!(c.ty.dims, dst.ty.dims);
            self.build_contiguous(dst, c.clone(), entry, &[])?
        } else {
            entry
        };

        let m = dst.ty.dims[0] as u64;
        let n = dst.ty.dims[1] as u64;
        let k = a.ty.dims[1] as u64;
        let beta = if c.is_some() { gemm.beta } else { 0.0 };
        let mut trans_a = gemm.trans_a;
        let mut trans_b = gemm.trans_b;
        assert!(a.ty.dims.ndim() == 2);
        assert!(b.ty.dims.ndim() == 2);
        if a.ty.stride(0) < a.ty.stride(1) {
            trans_a = !trans_a;
        }
        if b.ty.stride(0) < b.ty.stride(1) {
            trans_b = !trans_b;
        }
        let gemm = GemmArgs {
            a: (a.ptr, trans_a),
            b: (b.ptr, trans_b),
            c: dst.ptr,
            alpha: gemm.alpha,
            beta,
            m,
            n,
            k,
        };

        self.builder.position_at_end(entry);
        self.blas
            .call_gemm(dst.ty.elem_type.float_type().unwrap(), &gemm, self.builder)?;
        Ok(entry)
    }

    pub fn build_batched_gemm(
        &self,
        dst: &TensorPtr<'ctx>,
        a: &TensorPtr<'ctx>,
        b: &TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        gemm: &operator::BatchedGemm,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let ndim = a.ty.dims.ndim();
        assert!(ndim >= 3);
        assert!(a.ty.is_contiguous());
        assert!(b.ty.is_contiguous());
        assert!(dst.ty.is_contiguous());
        let fp_ty = a.ty.elem_type.float_type().unwrap();
        let i64_ty = self.context.i64_type();

        let (m, k) = if gemm.trans_a {
            (a.ty.dims[ndim - 1], a.ty.dims[ndim - 2])
        } else {
            (a.ty.dims[ndim - 2], a.ty.dims[ndim - 1])
        };
        let n = if gemm.trans_b {
            b.ty.dims[ndim - 2]
        } else {
            b.ty.dims[ndim - 1]
        };
        let batch_count = a.ty.dims.size() / (a.ty.dims[ndim - 2] * a.ty.dims[ndim - 1]);
        let stride_a = (a.ty.dims[ndim - 2] * a.ty.dims[ndim - 1]) as u64;
        let stride_b = (b.ty.dims[ndim - 2] * b.ty.dims[ndim - 1]) as u64;
        let stride_c = (m * n) as u64;

        self.builder.position_at_end(entry);
        let a_ptr = self.build_gep(a)?;
        let b_ptr = self.build_gep(b)?;
        let dst_ptr = self.build_gep(dst)?;

        if self.blas.has_batch_strided() {
            self.blas.call_gemm_batch_strided(
                fp_ty,
                &BatchedGemmArgs {
                    a: (a_ptr, gemm.trans_a),
                    b: (b_ptr, gemm.trans_b),
                    c: dst_ptr,
                    m: m as u64,
                    n: n as u64,
                    k: k as u64,
                    alpha: gemm.alpha,
                    beta: gemm.beta,
                    stride_a,
                    stride_b,
                    stride_c,
                    batch_count: batch_count as u64,
                },
                self.builder,
            )?;
        } else {
            let body = self.context.append_basic_block(*self.func, "bgemm.body");
            let exit = self.context.append_basic_block(*self.func, "bgemm.exit");

            let guard = self.builder.build_int_compare(
                inkwell::IntPredicate::EQ,
                i64_ty.const_int(batch_count as u64, false),
                i64_ty.const_zero(),
                "bgemm.guard",
            )?;
            self.builder.build_conditional_branch(guard, exit, body)?;
            self.builder.position_at_end(body);
            let ind = self.builder.build_phi(i64_ty, "bgemm.i")?;
            let idx = ind.as_basic_value().into_int_value();

            let a_off =
                self.builder
                    .build_int_mul(idx, i64_ty.const_int(stride_a, false), "a.off")?;
            let b_off =
                self.builder
                    .build_int_mul(idx, i64_ty.const_int(stride_b, false), "b.off")?;
            let c_off =
                self.builder
                    .build_int_mul(idx, i64_ty.const_int(stride_c, false), "c.off")?;

            let a_base = self.builder.build_int_add(a.offset, a_off, "a.base")?;
            let b_base = self.builder.build_int_add(b.offset, b_off, "b.base")?;
            let c_base = self.builder.build_int_add(dst.offset, c_off, "c.base")?;
            let a_slice = self.build_gep(&a.clone().set_offset(a_base))?;
            let b_slice = self.build_gep(&b.clone().set_offset(b_base))?;
            let c_slice = self.build_gep(&dst.clone().set_offset(c_base))?;

            self.blas.call_gemm(
                fp_ty,
                &GemmArgs {
                    a: (a_slice, gemm.trans_a),
                    b: (b_slice, gemm.trans_b),
                    c: c_slice,
                    m: m as u64,
                    n: n as u64,
                    k: k as u64,
                    alpha: gemm.alpha,
                    beta: gemm.beta,
                },
                self.builder,
            )?;

            let ind_next =
                self.builder
                    .build_int_add(idx, i64_ty.const_int(1, false), "bgemm.next")?;
            let ec = self.builder.build_int_compare(
                inkwell::IntPredicate::EQ,
                ind_next,
                i64_ty.const_int(batch_count as u64, false),
                "bgemm.ec",
            )?;
            self.builder.build_conditional_branch(ec, exit, body)?;
            ind.add_incoming(&[(&i64_ty.const_zero(), entry), (&ind_next, body)]);

            self.builder.position_at_end(exit);
            return Ok(exit);
        }
        Ok(entry)
    }

    pub fn build_matmul(
        &self,
        dst: &TensorPtr<'ctx>,
        a: &TensorPtr<'ctx>,
        b: &TensorPtr<'ctx>,
        c: Option<&TensorPtr<'ctx>>,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(3 <= a.ty.dims.ndim());
        let (m, k) = (
            a.ty.dims[a.ty.dims.ndim() - 2],
            a.ty.dims[a.ty.dims.ndim() - 1],
        );
        let n = b.ty.dims.last().copied().unwrap();
        assert!(b.ty.dims[b.ty.dims.ndim() - 2] == k);
        assert!(dst.ty.dims[dst.ty.dims.ndim() - 2] == m);
        assert!(dst.ty.dims[dst.ty.dims.ndim() - 1] == n);
        assert!(a.ty.is_contiguous());
        assert!(b.ty.is_contiguous());
        let alpha = 1.0;
        let beta = if c.is_some() { 1.0 } else { 0.0 };

        let bound = a.ty.dims.size() / (m * k);
        let gemm = operator::Gemm {
            trans_a: false,
            trans_b: false,
            alpha,
            beta,
        };

        let header = self.context.append_basic_block(*self.func, "matmul.header");
        let latch = self.context.append_basic_block(*self.func, "matmul.latch");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.build_unconditional_branch(header)?;

        self.builder.position_at_end(header);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let a = {
            let offset = self.builder.build_int_mul(
                ind.as_basic_value().into_int_value(),
                self.context.i64_type().const_int((m * k) as u64, false),
                "offset.a",
            )?;
            let a = a.clone().set_offset(offset);
            let ptr = self.build_gep(&a)?;
            TensorPtr {
                ptr,
                ty: ResolvedTensorType::new(a.ty.elem_type, ResolvedTensorDims::new(&[m, k])),
                offset,
                name: a.name.clone(),
            }
        };
        let b = {
            let offset = self.builder.build_int_mul(
                ind.as_basic_value().into_int_value(),
                self.context.i64_type().const_int((k * n) as u64, false),
                "offset.b",
            )?;
            let b = b.clone().set_offset(offset);
            let ptr = self.build_gep(&b)?;
            TensorPtr {
                ptr,
                ty: ResolvedTensorType::new(b.ty.elem_type, ResolvedTensorDims::new(&[k, n])),
                offset,
                name: b.name.clone(),
            }
        };
        let c = if let Some(c) = c {
            let offset = self.builder.build_int_mul(
                ind.as_basic_value().into_int_value(),
                self.context.i64_type().const_int((m * n) as u64, false),
                "offset.c",
            )?;
            let c = c.clone().set_offset(offset);
            let ptr = self.build_gep(&c)?;
            Some(TensorPtr {
                ptr,
                ty: ResolvedTensorType::new(c.ty.elem_type, ResolvedTensorDims::new(&[m, n])),
                offset,
                name: c.name.clone(),
            })
        } else {
            None
        };
        let dst = {
            let offset = self.builder.build_int_mul(
                ind.as_basic_value().into_int_value(),
                self.context.i64_type().const_int((m * n) as u64, false),
                "offset.dst",
            )?;
            let dst = dst.clone().set_offset(offset);
            let ptr = self.build_gep(&dst)?;
            TensorPtr {
                ptr,
                ty: ResolvedTensorType::new(dst.ty.elem_type, ResolvedTensorDims::new(&[m, n])),
                offset,
                name: dst.name.clone(),
            }
        };
        self.build_gemm(&dst, &a, &b, c.as_ref(), header, &gemm)?;
        self.builder.build_unconditional_branch(latch)?;

        self.builder.position_at_end(latch);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            ind_next,
            self.context.i64_type().const_int(bound as u64, false),
            "ec",
        )?;
        self.builder.build_conditional_branch(ec, exit, header)?;
        ind.add_incoming(&[
            (&ind_next, latch),
            (&self.context.i64_type().const_zero(), entry),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_matrix_reduce(
        &self,
        ptrs: &[TensorPtr<'ctx>],
        elem_ty: DataType,
        mn: (u64, u64),
        op: operator::ReduceOp,
        preheader: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let (row, col) = mn;

        let header0 = self.context.append_basic_block(*self.func, "header0");
        let exiting0 = self.context.append_basic_block(*self.func, "exiting0");
        let body = self.context.append_basic_block(*self.func, "body");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.build_unconditional_branch(header0)?;

        self.builder.position_at_end(header0);
        let ind0 = self.builder.build_phi(self.context.i64_type(), "ind0")?;
        let offset0 = self.builder.build_phi(self.context.i64_type(), "offset0")?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ind1 = self.builder.build_phi(self.context.i64_type(), "ind1")?;
        let fp_ty = match elem_ty {
            DataType::Float(t) => t.llvm_type(self.context),
            _ => todo!(),
        };
        let acc = self.builder.build_phi(fp_ty, "acc")?;
        let offset1 = self.builder.build_int_mul(
            ind1.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(ptrs[1].stride(1).try_into().unwrap(), false),
            "offset1",
        )?;
        let offset1 = self.builder.build_int_add(
            offset0.as_basic_value().into_int_value(),
            offset1,
            "offset1",
        )?;

        macro_rules! load {
            () => {{
                self.build_raw_load(fp_ty, ptrs[1].ptr, offset1)?
                    .into_float_value()
            }};
        }

        let (id_v, res) = match op {
            operator::ReduceOp::Max => {
                let ty = elem_ty.float_type().unwrap();
                let fmax = self.intrinsics.fmax.get(ty);
                let id_v = match ty {
                    FloatType::F32 => f32::MIN as f64,
                    FloatType::F64 => f64::MIN,
                };
                let ty = ty.llvm_type(self.context);
                let id_v = ty.const_float(id_v);
                let val = load!();
                let res = self
                    .build_tail_call(fmax, &[acc.as_basic_value().into(), val.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap();
                (id_v.as_basic_value_enum(), res)
            }
            operator::ReduceOp::Sum | operator::ReduceOp::Mean | operator::ReduceOp::Variance => {
                let fp_ty = match elem_ty {
                    DataType::Float(t) => t.llvm_type(self.context),
                    _ => todo!(),
                };
                let zero = fp_ty.const_zero();
                let val = load!();
                let res = match op {
                    operator::ReduceOp::Sum | operator::ReduceOp::Mean => self
                        .builder
                        .build_float_add(acc.as_basic_value().into_float_value(), val, "res"),
                    operator::ReduceOp::Variance => {
                        // TODO?: stride
                        let mean = self
                            .build_raw_load(
                                fp_ty,
                                ptrs[2].ptr,
                                ind0.as_basic_value().into_int_value(),
                            )?
                            .into_float_value();
                        let diff = self.builder.build_float_sub(val, mean, "diff")?;
                        self.builder.build_float_mul(diff, diff, "diff.squared")
                    }
                    _ => unreachable!(),
                }?;
                (zero.as_basic_value_enum(), res.as_basic_value_enum())
            }
        };
        let ind1_next = self.builder.build_int_add(
            ind1.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind1.next",
        )?;
        let cond1 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind1_next,
            self.context.i64_type().const_int(col, false),
            "cond1",
        )?;
        self.builder
            .build_conditional_branch(cond1, body, exiting0)?;
        ind1.add_incoming(&[
            (&ind1_next, body),
            (&self.context.i64_type().const_zero(), header0),
        ]);
        acc.add_incoming(&[(&res, body), (&id_v, header0)]);

        self.builder.position_at_end(exiting0);
        let res = match op {
            operator::ReduceOp::Mean | operator::ReduceOp::Variance => {
                let div = fp_ty.const_float(col as f64);
                self.builder
                    .build_float_div(res.into_float_value(), div, "res")?
                    .as_basic_value_enum()
            }
            operator::ReduceOp::Max | operator::ReduceOp::Sum => res,
        };
        self.build_raw_store(
            fp_ty,
            ptrs[0].ptr,
            ind0.as_basic_value().into_int_value(),
            res,
        )?;
        let ind0_next = self.builder.build_int_add(
            ind0.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind0.next",
        )?;
        let offset0_next = self.builder.build_int_add(
            offset0.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(ptrs[1].stride(0).try_into().unwrap(), false),
            "offset0.next",
        )?;
        let cond0 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind0_next,
            self.context.i64_type().const_int(row, false),
            "cond0",
        )?;
        self.builder
            .build_conditional_branch(cond0, header0, exit)?;
        ind0.add_incoming(&[
            (&ind0_next, exiting0),
            (&self.context.i64_type().const_zero(), preheader),
        ]);
        offset0.add_incoming(&[
            (&offset0_next, exiting0),
            (&self.context.i64_type().const_zero(), preheader),
        ]);

        self.builder.position_at_end(exit);

        Ok(exit)
    }

    fn omp_parallel<F>(
        &self,
        captures: &[BasicValueEnum<'ctx>],
        build_body: F,
    ) -> Result<(), BuilderError>
    where
        F: FnOnce(
            &FunctionTranslator<'_, 'ctx>,
            &OMPContext<'ctx>,
            &[BasicValueEnum<'ctx>],
            LoopBB<'ctx>,
        ) -> Result<(), BuilderError>,
    {
        let mut alloca_ptrs = Vec::with_capacity(captures.len());
        for (i, cap) in captures.iter().enumerate() {
            let alloca = self
                .builder
                .build_alloca(cap.get_type(), &format!("cap.{i}"))?;
            self.builder.build_store(alloca, *cap)?;
            alloca_ptrs.push(alloca);
        }

        // (global_tid, bound_tid, ptr0, ptr1, ...) -> void
        let ptr_type = self.context.ptr_type(inkwell::AddressSpace::default());
        let outlined_fn = self
            .context
            .void_type()
            .fn_type(&vec![ptr_type.into(); 2 + captures.len()], false);
        let outlined_fn = self.module.add_function(
            &format!("{}.omp_outlined", self.func.get_name().to_str().unwrap()),
            outlined_fn,
            None,
        );

        let new_builder = self.context.create_builder();
        let entry = self.context.append_basic_block(outlined_fn, "omp.entry");
        let body = self.context.append_basic_block(outlined_fn, "omp.body");
        let exit = self.context.append_basic_block(outlined_fn, "omp.exit");

        new_builder.position_at_end(entry);
        let i32_type = self.context.i32_type();
        let is_last = new_builder.build_alloca(i32_type, "is.last")?;
        let lb = new_builder.build_alloca(i32_type, "lb")?;
        let ub = new_builder.build_alloca(i32_type, "ub")?;
        let stride = new_builder.build_alloca(i32_type, "stride")?;
        let omp_ctx = OMPContext {
            global_tid: outlined_fn.get_nth_param(0).unwrap().into_pointer_value(),
            is_last,
            lb,
            ub,
            stride,
        };
        let mut loaded = Vec::with_capacity(captures.len());
        for (i, cap) in captures.iter().enumerate() {
            let param = outlined_fn
                .get_nth_param((i + 2) as u32)
                .unwrap()
                .into_pointer_value();
            let val = new_builder.build_load(cap.get_type(), param, &format!("cap.{i}"))?;
            loaded.push(val);
        }
        new_builder.build_unconditional_branch(body)?;

        new_builder.position_at_end(exit);
        new_builder.build_return(None)?;

        new_builder.position_at_end(body);
        let translator = {
            let mut t = self.clone();
            t.builder = &new_builder;
            t.func = &outlined_fn;
            t
        };
        let loop_bb = LoopBB {
            preheader: entry,
            header: body,
            exit,
        };
        build_body(&translator, &omp_ctx, &loaded, loop_bb)?;

        let args = ForkCallArgs {
            outlined: outlined_fn,
            args: alloca_ptrs,
        };
        self.omp.fork_call(self.builder, &args)?;

        Ok(())
    }

    fn omp_for_static(
        &self,
        omp_ctx: &OMPContext<'ctx>,
        bound: u64,
        exit_bb: BasicBlock<'ctx>,
    ) -> Result<OMPForRange<'ctx>, BuilderError> {
        let body_bb = self.context.append_basic_block(*self.func, "omp.for.body");
        let epilog_bb = self
            .context
            .append_basic_block(*self.func, "omp.for.epilog");

        let i32_type = self.context.i32_type();
        let len = i32_type.const_int(bound - 1, false);

        let tid = self
            .builder
            .build_load(i32_type, omp_ctx.global_tid, "global.tid")?
            .into_int_value();
        let one = i32_type.const_int(1, false);
        for (ptr, val) in [
            (omp_ctx.is_last, i32_type.const_zero()),
            (omp_ctx.lb, i32_type.const_zero()),
            (omp_ctx.ub, len),
            (omp_ctx.stride, one),
        ] {
            self.builder.build_store(ptr, val)?;
        }

        let args = StaticInitArgs {
            tid,
            sched: ScheduleType::UnorderedStatic,
            is_last: omp_ctx.is_last,
            lb: omp_ctx.lb,
            ub: omp_ctx.ub,
            stride: omp_ctx.stride,
            incr: one,
        };
        self.omp.static_init(self.builder, &args)?;

        let lb = self
            .builder
            .build_load(i32_type, omp_ctx.lb, "lb")?
            .into_int_value();
        let ub = self
            .builder
            .build_load(i32_type, omp_ctx.ub, "ub_omp")?
            .into_int_value();
        let ub = self
            .builder
            .build_call(self.intrinsics.smin_i32, &[ub.into(), len.into()], "ub_min")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let ub = self.builder.build_int_add(ub, one, "ub_open")?;
        let cond = self
            .builder
            .build_int_compare(inkwell::IntPredicate::SLT, lb, ub, "cond")?;
        self.builder
            .build_conditional_branch(cond, body_bb, epilog_bb)?;

        self.builder.position_at_end(epilog_bb);
        self.omp
            .static_fini(self.builder, &StaticFiniArgs { tid })?;
        self.builder.build_unconditional_branch(exit_bb)?;

        self.builder.position_at_end(body_bb);
        let i64_type = self.context.i64_type();
        let lb = self.builder.build_int_s_extend(lb, i64_type, "lb.i64")?;
        let ub = self.builder.build_int_s_extend(ub, i64_type, "ub.i64")?;

        Ok(OMPForRange {
            lb,
            ub,
            body_bb,
            epilog_bb,
        })
    }

    pub fn build_flat_loop(
        &self,
        op: Operation<'ctx>,
        preheader: BasicBlock<'ctx>,
        use_omp: bool,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let output_size: u64 = op.result_dims().size().try_into().unwrap();
        let operand_info: Vec<Vec<(u64, u64)>> = op
            .operands
            .iter()
            .map(|ptr| {
                (0..ptr.ty.dims.ndim())
                    .map(|d| (ptr.ty.dims[d] as u64, ptr.ty.stride(d) as u64))
                    .collect()
            })
            .collect();
        let output_dims: Vec<u64> = op.result_dims().iter().map(|d| *d as u64).collect();
        let dst_ty = op.dst_operand().ty.clone();
        let tensor_info: Vec<_> = op
            .operands
            .iter()
            .map(|t| (t.ty.clone(), t.name.clone()))
            .collect();
        let opcode = op.opcode.clone();
        let captures: Vec<BasicValueEnum<'ctx>> = op
            .operands
            .iter()
            .flat_map(|t| [t.ptr.as_basic_value_enum(), t.offset.as_basic_value_enum()])
            .collect();

        if use_omp {
            let exit = self.context.append_basic_block(*self.func, "flat.exit");
            self.omp_parallel(&captures, |translator, omp_ctx, loaded, loop_bb| {
                let operands: smallvec::SmallVec<[TensorPtr<'ctx>; 4]> = tensor_info
                    .iter()
                    .enumerate()
                    .map(|(i, (ty, name))| TensorPtr {
                        ptr: loaded[i * 2].into_pointer_value(),
                        ty: ty.clone(),
                        offset: loaded[i * 2 + 1].into_int_value(),
                        name: name.clone(),
                    })
                    .collect();
                let op = Operation {
                    opcode: opcode.clone(),
                    operands,
                };
                let range = translator.omp_for_static(omp_ctx, output_size, loop_bb.exit)?;
                translator.build_flat_loop_body(
                    op,
                    range.body_bb,
                    range.lb,
                    range.ub,
                    &operand_info,
                    &output_dims,
                    &dst_ty,
                )?;
                translator
                    .builder
                    .build_unconditional_branch(range.epilog_bb)?;
                Ok(())
            })?;
            self.builder.build_unconditional_branch(exit)?;
            self.builder.position_at_end(exit);
            Ok(exit)
        } else {
            let i64_type = self.context.i64_type();
            let lb = i64_type.const_zero();
            let ub = i64_type.const_int(output_size, false);
            self.build_flat_loop_body(op, preheader, lb, ub, &operand_info, &output_dims, &dst_ty)
        }
    }

    fn build_flat_loop_body(
        &self,
        mut op: Operation<'ctx>,
        preheader: BasicBlock<'ctx>,
        lb: IntValue<'ctx>,
        ub: IntValue<'ctx>,
        operand_info: &[Vec<(u64, u64)>],
        output_dims: &[u64],
        dst_ty: &ResolvedTensorType,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let i64_type = self.context.i64_type();

        let header = self.context.append_basic_block(*self.func, "flat.header");
        let body = self.context.append_basic_block(*self.func, "flat.body");
        let latch = self.context.append_basic_block(*self.func, "flat.latch");
        let exit = self.context.append_basic_block(*self.func, "flat.exit");

        let base_offsets: Vec<IntValue<'ctx>> = op.operands.iter().map(|ptr| ptr.offset).collect();

        self.builder.build_unconditional_branch(header)?;
        self.builder.position_at_end(header);

        let ind = self.builder.build_phi(i64_type, "flat.ind")?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind.as_basic_value().into_int_value(),
            ub,
            "flat.cond",
        )?;
        self.builder.build_conditional_branch(cond, body, exit)?;

        self.builder.position_at_end(body);
        let flat_idx = ind.as_basic_value().into_int_value();

        for (op_idx, ptr) in op.operands.iter_mut().enumerate() {
            if ptr.ty == *dst_ty && ptr.ty.is_contiguous() {
                ptr.offset = self.builder.build_int_add(
                    base_offsets[op_idx],
                    flat_idx,
                    &format!("off.{}", op_idx),
                )?;
            } else {
                let info = &operand_info[op_idx];
                let mut offset = base_offsets[op_idx];
                let mut remaining = flat_idx;

                for (dim_idx, out_dim) in output_dims.iter().enumerate().rev() {
                    let dim_const = i64_type.const_int(*out_dim, false);
                    let idx = self.builder.build_int_unsigned_rem(
                        remaining,
                        dim_const,
                        &format!("idx.{}.{}", op_idx, dim_idx),
                    )?;
                    remaining = self.builder.build_int_unsigned_div(
                        remaining,
                        dim_const,
                        &format!("rem.{}.{}", op_idx, dim_idx),
                    )?;

                    let (_, stride) = info[dim_idx];
                    if stride != 0 {
                        let stride_const = i64_type.const_int(stride, false);
                        let contrib = self.builder.build_int_mul(
                            idx,
                            stride_const,
                            &format!("contrib.{}.{}", op_idx, dim_idx),
                        )?;
                        offset = self.builder.build_int_add(
                            offset,
                            contrib,
                            &format!("off.{}.{}", op_idx, dim_idx),
                        )?;
                    }
                }

                ptr.offset = offset;
            }
        }

        self.build_operation(&op)?;

        self.builder.build_unconditional_branch(latch)?;
        self.builder.position_at_end(latch);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            i64_type.const_int(1, false),
            "flat.ind.next",
        )?;
        self.builder.build_unconditional_branch(header)?;
        ind.add_incoming(&[(&lb, preheader), (&ind_next, latch)]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    fn build_resize_rec(&self, param: ResizeParam<'_, 'ctx>) -> Result<(), BuilderError> {
        let ResizeParam {
            dst,
            src,
            axes,
            loop_bb,
            dim,
            resize,
        } = param;

        self.builder.position_at_end(loop_bb.header);

        if dim == dst.ty.dims.ndim() {
            let val = self.build_load(&src)?;
            self.build_store(&dst, val)?;
            self.builder.build_unconditional_branch(loop_bb.exit)?;
            return Ok(());
        }

        let next_preheader = loop_bb.header;
        let next_header = self.context.append_basic_block(*self.func, "header");
        let next_exit = self.context.append_basic_block(*self.func, "exit");

        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let dst_offset_add = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(dst.ty.stride(dim).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset_add, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        let nth_resize = axes.iter().position(|&x| x == dim);
        let x_original = match nth_resize {
            Some(n) => {
                let Some(scale) = resize.scale.as_ref() else {
                    unreachable!();
                };
                let scale = match scale {
                    operator::ResizeScale::Scales(s) => self.context.f32_type().const_float(s[n]),
                    operator::ResizeScale::Sizes(resized) => {
                        let resized = self.context.i64_type().const_int(resized[n] as u64, false);
                        let resized = self.builder.build_signed_int_to_float(
                            resized,
                            self.context.f32_type(),
                            "resized",
                        )?;
                        self.builder.build_float_div(
                            resized,
                            self.context.f32_type().const_float(src.ty.dims[dim] as f64),
                            "scale",
                        )?
                    }
                };
                let x_resized = self.builder.build_signed_int_to_float(
                    ind.as_basic_value().into_int_value(),
                    self.context.f32_type(),
                    "x_resized",
                )?;
                let half = self.context.f32_type().const_float(0.5);
                let x_original = match resize.coordinate_transformation_mode {
                    operator::ResizeCoordinateTransformationMode::HalfPixel => {
                        // (x_resized + 0.5) / scale - 0.5
                        let res = self.builder.build_float_add(x_resized, half, "res")?;
                        let res = self.builder.build_float_div(res, scale, "res")?;
                        self.builder.build_float_sub(res, half, "res")?
                    }
                };
                let x_original = match resize.mode {
                    operator::ResizeMode::Nearest(nearest) => {
                        let res = match nearest {
                            operator::ResizeNearestMode::RoundPreferFloor => {
                                let x =
                                    self.builder
                                        .build_float_sub(x_original, half, "x_original")?;
                                self.build_tail_call(
                                    self.intrinsics.ceil.get(FloatType::F32),
                                    &[x.into()],
                                    "x_original",
                                )
                            }
                            operator::ResizeNearestMode::Floor => self.build_tail_call(
                                self.intrinsics.floor.get(FloatType::F32),
                                &[x_original.into()],
                                "x_original",
                            ),
                            operator::ResizeNearestMode::Ceil => self.build_tail_call(
                                self.intrinsics.ceil.get(FloatType::F32),
                                &[x_original.into()],
                                "x_original",
                            ),
                            _ => unimplemented!(),
                        }?;
                        res.try_as_basic_value().left().unwrap().into_float_value()
                    }
                };
                let x_original = self.builder.build_float_to_signed_int(
                    x_original,
                    self.context.i64_type(),
                    "x_original",
                )?;
                let x_original = self
                    .build_tail_call(
                        self.intrinsics.smin_i64,
                        &[
                            x_original.into(),
                            self.context
                                .i64_type()
                                .const_int((src.ty.dims[dim] - 1).try_into().unwrap(), false)
                                .into(),
                        ],
                        "x_original",
                    )?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_int_value();
                let x_original = self
                    .build_tail_call(
                        self.intrinsics.smax_i64,
                        &[
                            x_original.into(),
                            self.context.i64_type().const_zero().into(),
                        ],
                        "x_original",
                    )?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
                    .into_int_value();
                x_original
            }
            None => ind.as_basic_value().into_int_value(),
        };
        let src_offset_add = self.builder.build_int_mul(
            x_original,
            self.context
                .i64_type()
                .const_int(src.ty.stride(dim).try_into().unwrap(), false),
            "src.offset",
        )?;
        let src_offset = self
            .builder
            .build_int_add(src.offset, src_offset_add, "src.offset")?;
        let src = src.set_offset(src_offset);
        self.builder.build_unconditional_branch(next_header)?;

        self.builder.position_at_end(next_exit);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind_next,
            self.context
                .i64_type()
                .const_int(dst.ty.dims[dim].try_into().unwrap(), false),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, loop_bb.header, loop_bb.exit)?;
        ind.add_incoming(&[
            (&self.context.i64_type().const_zero(), loop_bb.preheader),
            (&ind_next, next_exit),
        ]);

        let loop_bb = LoopBB {
            preheader: next_preheader,
            header: next_header,
            exit: next_exit,
        };

        let param = ResizeParam {
            dst,
            src,
            axes,
            loop_bb,
            dim: dim + 1,
            resize,
        };

        self.build_resize_rec(param)
    }

    pub fn build_resize(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        resize: &operator::Resize,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        if resize.scale.is_none() {
            unimplemented!();
        }
        let ndim = dst.ty.dims.ndim();
        let axes: Vec<_> = match resize.axes {
            Some(ref axes) => axes.iter().map(|x| x.index(ndim)).collect(),
            None => (0..ndim).collect(),
        };

        let preheader = entry;
        let header = self.context.append_basic_block(*self.func, "header");
        let exit = self.context.append_basic_block(*self.func, "exit");
        let loop_bb = LoopBB {
            preheader,
            header,
            exit,
        };

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(header)?;

        let param = ResizeParam {
            dst,
            src,
            axes,
            loop_bb,
            dim: 0,
            resize,
        };
        self.build_resize_rec(param)?;
        Ok(exit)
    }

    pub fn build_concat(
        &self,
        dst: TensorPtr<'ctx>,
        srcs: &[TensorPtr<'ctx>],
        entry: BasicBlock<'ctx>,
        axis: usize,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let mut acc = 0;
        let mut entry = entry;
        let stride = dst.ty.stride(axis);
        for src in srcs {
            let offset = self.builder.build_int_add(
                dst.offset,
                self.context
                    .i64_type()
                    .const_int(acc.try_into().unwrap(), false),
                "dst.offset",
            )?;
            let dst = {
                let mut new_ty = dst.ty.clone();
                new_ty.dims[axis] = src.ty.dims[axis];
                dst.clone().set_offset(offset).set_type(new_ty)
            };
            let op = Operation {
                opcode: SingleOpcode::Transfer.into(),
                operands: smallvec![dst, src.clone()],
            };
            entry = self.build_flat_loop(op, entry, false)?;
            acc += src.ty.dims[axis] * stride;
        }
        Ok(entry)
    }

    fn scalar_to_llvm_value(
        &self,
        value: &ScalarData,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let res = match value {
            ScalarData::SInt(ty, v) => {
                let ty = match ty {
                    SIntType::I32 => self.context.i32_type(),
                    SIntType::I64 => self.context.i64_type(),
                };
                ty.const_int(*v as u64, true).into()
            }
            ScalarData::UInt(ty, v) => {
                let ty = match ty {
                    UIntType::U64 => self.context.i64_type(),
                };
                ty.const_int(*v, false).into()
            }
            ScalarData::Float(ty, v) => {
                let ty = match ty {
                    FloatType::F32 => self.context.f32_type(),
                    FloatType::F64 => self.context.f64_type(),
                };
                ty.const_float(*v).into()
            }
        };
        Ok(res)
    }

    pub fn build_one_hot(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        one_hot: &operator::OneHot,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(one_hot.axis == -1 || one_hot.axis == (src.ty.dims.ndim() as isize - 1));
        assert!(src.ty.is_contiguous() && dst.ty.is_contiguous());
        let Some(depth) = one_hot.depth else {
            unimplemented!();
        };
        let Some(on_value) = one_hot.on_value else {
            unimplemented!();
        };
        let Some(off_value) = one_hot.off_value else {
            unimplemented!();
        };
        let on_value = self.scalar_to_llvm_value(&on_value)?;
        let off_value = self.scalar_to_llvm_value(&off_value)?;
        let depth = self.context.i64_type().const_int(depth as u64, false);

        let outer_header = self.context.append_basic_block(*self.func, "outer.header");
        let inner = self.context.append_basic_block(*self.func, "inner");
        let outer_latch = self.context.append_basic_block(*self.func, "outer.latch");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(outer_header)?;

        self.builder.position_at_end(outer_header);
        let outer_i = self.builder.build_phi(self.context.i64_type(), "i.outer")?;
        let src = src.set_offset(outer_i.as_basic_value().into_int_value());
        let index = self.build_load(&src)?.into_int_value();
        let index = {
            let add = self.builder.build_int_add(index, depth, "index.add")?;
            let is_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                index,
                self.context.i64_type().const_zero(),
                "is_neg",
            )?;
            self.builder
                .build_select(is_neg, add, index, "index")?
                .into_int_value()
        };
        self.builder.build_unconditional_branch(inner)?;

        self.builder.position_at_end(inner);
        let inner_i = self.builder.build_phi(self.context.i64_type(), "i.inner")?;
        let is_on = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i.as_basic_value().into_int_value(),
            index,
            "is_on",
        )?;
        let val = self
            .builder
            .build_select(is_on, on_value, off_value, "val")?;
        let offset = self.builder.build_int_mul(
            outer_i.as_basic_value().into_int_value(),
            depth,
            "offset",
        )?;
        let offset = self.builder.build_int_add(
            offset,
            inner_i.as_basic_value().into_int_value(),
            "offset",
        )?;
        let dst = dst.set_offset(offset);
        self.build_store(&dst, val)?;
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            depth,
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, outer_latch, inner)?;
        inner_i.add_incoming(&[
            (&self.context.i64_type().const_zero(), outer_header),
            (&inner_i_next, inner),
        ]);

        self.builder.position_at_end(outer_latch);
        let outer_i_next = self.builder.build_int_add(
            outer_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "outer.i.next",
        )?;
        let outer_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            outer_i_next,
            self.context
                .i64_type()
                .const_int(src.ty.dims.size() as u64, false),
            "ec.outer",
        )?;
        self.builder
            .build_conditional_branch(outer_ec, exit, outer_header)?;
        outer_i.add_incoming(&[
            (&self.context.i64_type().const_zero(), entry),
            (&outer_i_next, outer_latch),
        ]);
        self.builder.position_at_end(exit);

        Ok(exit)
    }

    pub fn build_softmax(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        softmax: &operator::Softmax,
        use_omp: bool,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(src.ty.is_contiguous() && dst.ty.is_contiguous());

        let axis = softmax.axis.index(src.ty.dims.ndim());
        let val_ty = match src.ty.elem_type {
            DataType::Float(t) => t,
            _ => unimplemented!(),
        };
        let fmax_id = match val_ty {
            FloatType::F32 => f32::MIN as f64,
            FloatType::F64 => f64::MIN,
        };

        let outer_bound = {
            let mut acc = 1;
            for d in src.ty.dims.iter().take(axis) {
                acc *= *d as u64;
            }
            acc
        };
        let inner_bound = src.ty.dims[axis] as u64;
        let middle_bound = src.ty.dims.size() as u64 / (outer_bound * inner_bound);

        let src_ty = src.ty.clone();
        let dst_ty = dst.ty.clone();

        if use_omp {
            let exit = self.context.append_basic_block(*self.func, "softmax.exit");
            let captures: Vec<BasicValueEnum<'ctx>> = vec![
                src.ptr.as_basic_value_enum(),
                src.offset.as_basic_value_enum(),
                dst.ptr.as_basic_value_enum(),
                dst.offset.as_basic_value_enum(),
            ];
            self.omp_parallel(&captures, |translator, omp_ctx, loaded, loop_bb| {
                let src = TensorPtr {
                    ptr: loaded[0].into_pointer_value(),
                    ty: src_ty.clone(),
                    offset: loaded[1].into_int_value(),
                    name: "src".to_string(),
                };
                let dst = TensorPtr {
                    ptr: loaded[2].into_pointer_value(),
                    ty: dst_ty.clone(),
                    offset: loaded[3].into_int_value(),
                    name: "dst".to_string(),
                };
                let range = translator.omp_for_static(omp_ctx, outer_bound, loop_bb.exit)?;
                translator.build_softmax_outer_loop(
                    dst,
                    src,
                    range.body_bb,
                    range.lb,
                    range.ub,
                    val_ty,
                    fmax_id,
                    inner_bound,
                    middle_bound,
                )?;
                translator
                    .builder
                    .build_unconditional_branch(range.epilog_bb)?;
                Ok(())
            })?;
            self.builder.position_at_end(entry);
            self.builder.build_unconditional_branch(exit)?;
            self.builder.position_at_end(exit);
            Ok(exit)
        } else {
            let i64_type = self.context.i64_type();
            let lb = i64_type.const_zero();
            let ub = i64_type.const_int(outer_bound, false);
            self.build_softmax_outer_loop(
                dst,
                src,
                entry,
                lb,
                ub,
                val_ty,
                fmax_id,
                inner_bound,
                middle_bound,
            )
        }
    }

    fn build_softmax_outer_loop(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        preheader: BasicBlock<'ctx>,
        lb: IntValue<'ctx>,
        ub: IntValue<'ctx>,
        val_ty: FloatType,
        fmax_id: f64,
        inner_bound: u64,
        middle_bound: u64,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let fmax = self.intrinsics.fmax.get(val_ty);
        let fexp = self.intrinsics.exp.get(val_ty);
        let val_ty = val_ty.llvm_type(self.context);
        let i64_type = self.context.i64_type();

        let outer_header = self.context.append_basic_block(*self.func, "outer.header");
        let middle_header = self.context.append_basic_block(*self.func, "middle.header");
        let max_inner = self.context.append_basic_block(*self.func, "max.inner");
        let sum_inner = self.context.append_basic_block(*self.func, "sum.inner");
        let div_inner = self.context.append_basic_block(*self.func, "div.inner");
        let middle_latch = self.context.append_basic_block(*self.func, "middle.latch");
        let outer_latch = self.context.append_basic_block(*self.func, "outer.latch");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.position_at_end(preheader);
        self.builder.build_unconditional_branch(outer_header)?;

        self.builder.position_at_end(outer_header);
        let outer_i = self.builder.build_phi(i64_type, "i.outer")?;
        let outer_offset = self.builder.build_int_mul(
            outer_i.as_basic_value().into_int_value(),
            i64_type.const_int(middle_bound * inner_bound, false),
            "outer.offset",
        )?;
        self.builder.build_unconditional_branch(middle_header)?;

        self.builder.position_at_end(middle_header);
        let middle_i = self.builder.build_phi(i64_type, "i.middle")?;
        self.builder.build_unconditional_branch(max_inner)?;

        macro_rules! get_offset {
            ($inner_i: expr) => {{
                let inner_offset = self.builder.build_int_mul(
                    $inner_i.as_basic_value().into_int_value(),
                    i64_type.const_int(middle_bound, false),
                    "inner.offset",
                )?;
                let offset = self
                    .builder
                    .build_int_add(outer_offset, inner_offset, "offset")?;
                let offset = self.builder.build_int_add(
                    offset,
                    middle_i.as_basic_value().into_int_value(),
                    "offset",
                )?;
                offset
            }};
        }

        self.builder.position_at_end(max_inner);
        let inner_i = self.builder.build_phi(i64_type, "i.inner")?;
        let max_val = self.builder.build_phi(val_ty, "max.val")?;
        let offset = get_offset!(inner_i);
        let src = src.set_offset(offset);
        let src_val = self.build_load(&src)?.into_float_value();
        let max_val_next = self
            .build_tail_call(
                fmax,
                &[max_val.as_basic_value().into(), src_val.into()],
                "max.val",
            )?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            i64_type.const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            i64_type.const_int(inner_bound, false),
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, sum_inner, max_inner)?;
        inner_i.add_incoming(&[
            (&i64_type.const_zero(), middle_header),
            (&inner_i_next, max_inner),
        ]);
        max_val.add_incoming(&[
            (&val_ty.const_float(fmax_id), middle_header),
            (&max_val_next, max_inner),
        ]);

        self.builder.position_at_end(sum_inner);
        let max_val = max_val_next;
        let inner_i = self.builder.build_phi(i64_type, "i.inner")?;
        let sum_val = self.builder.build_phi(val_ty, "sum.val")?;
        let offset = get_offset!(inner_i);
        let src = src.set_offset(offset);
        let src_val = self.build_load(&src)?.into_float_value();
        let src_val = self.builder.build_float_sub(src_val, max_val, "src.sub")?;
        let exp_src = self
            .build_tail_call(fexp, &[src_val.into()], "exp.src")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let sum_val_next = self.builder.build_float_add(
            sum_val.as_basic_value().into_float_value(),
            exp_src,
            "sum.val",
        )?;
        let dst = dst.set_offset(offset);
        self.build_store(&dst, exp_src.as_basic_value_enum())?;
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            i64_type.const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            i64_type.const_int(inner_bound, false),
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, div_inner, sum_inner)?;
        inner_i.add_incoming(&[
            (&i64_type.const_zero(), max_inner),
            (&inner_i_next, sum_inner),
        ]);
        sum_val.add_incoming(&[
            (&val_ty.const_float(0.0), max_inner),
            (&sum_val_next, sum_inner),
        ]);

        self.builder.position_at_end(div_inner);
        let sum_val = sum_val_next;
        let inner_i = self.builder.build_phi(i64_type, "i.inner")?;
        let offset = get_offset!(inner_i);
        let dst = dst.set_offset(offset);
        let dst_val = self.build_load(&dst)?.into_float_value();
        let dst_val = self.builder.build_float_div(dst_val, sum_val, "dst.val")?;
        self.build_store(&dst, dst_val.as_basic_value_enum())?;
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            i64_type.const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            i64_type.const_int(inner_bound, false),
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, middle_latch, div_inner)?;
        inner_i.add_incoming(&[
            (&i64_type.const_zero(), sum_inner),
            (&inner_i_next, div_inner),
        ]);

        self.builder.position_at_end(middle_latch);
        let middle_i_next = self.builder.build_int_add(
            middle_i.as_basic_value().into_int_value(),
            i64_type.const_int(1, false),
            "middle.i.next",
        )?;
        let middle_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            middle_i_next,
            i64_type.const_int(middle_bound, false),
            "ec.middle",
        )?;
        self.builder
            .build_conditional_branch(middle_ec, outer_latch, middle_header)?;
        middle_i.add_incoming(&[
            (&i64_type.const_zero(), outer_header),
            (&middle_i_next, middle_latch),
        ]);

        self.builder.position_at_end(outer_latch);
        let outer_i_next = self.builder.build_int_add(
            outer_i.as_basic_value().into_int_value(),
            i64_type.const_int(1, false),
            "outer.i.next",
        )?;
        let outer_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            outer_i_next,
            ub,
            "ec.outer",
        )?;
        self.builder
            .build_conditional_branch(outer_ec, exit, outer_header)?;
        outer_i.add_incoming(&[(&lb, preheader), (&outer_i_next, outer_latch)]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_layer_norm(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        scale: TensorPtr<'ctx>,
        bias: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        ln: &operator::LayerNormalization,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(src.ty.is_contiguous() && dst.ty.is_contiguous());

        let axis = ln.axis.index(src.ty.dims.ndim());
        assert_eq!(axis, src.ty.dims.ndim() - 1);

        let val_ty = match src.ty.elem_type {
            DataType::Float(t) => t,
            _ => unimplemented!(),
        };
        let sqrt = self.intrinsics.sqrt.get(val_ty);
        let fma = self.intrinsics.fma.get(val_ty);
        let val_ty = val_ty.llvm_type(self.context);

        let inner_bound = src.ty.dims[axis] as u64;
        let outer_bound = src.ty.dims.size() as u64 / inner_bound;

        let outer_header = self.context.append_basic_block(*self.func, "outer.header");
        let mean_body = self.context.append_basic_block(*self.func, "mean.body");
        let mean_done = self.context.append_basic_block(*self.func, "mean.done");
        let var_body = self.context.append_basic_block(*self.func, "var.body");
        let var_done = self.context.append_basic_block(*self.func, "var.done");
        let norm_body = self.context.append_basic_block(*self.func, "norm.body");
        let outer_latch = self.context.append_basic_block(*self.func, "outer.latch");
        let exit = self.context.append_basic_block(*self.func, "exit");

        let epsilon = val_ty.const_float(ln.epsilon);
        let inner_bound_f = val_ty.const_float(inner_bound as f64);

        // entry -> outer_header
        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(outer_header)?;

        // Outer loop header
        self.builder.position_at_end(outer_header);
        let outer_i = self.builder.build_phi(self.context.i64_type(), "i.outer")?;
        let outer_offset = self.builder.build_int_mul(
            outer_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(inner_bound, false),
            "outer.offset",
        )?;
        self.builder.build_unconditional_branch(mean_body)?;

        // Helper: src offset = outer_offset + inner_i
        macro_rules! src_offset {
            ($inner_i:expr) => {
                self.builder.build_int_add(
                    outer_offset,
                    $inner_i.as_basic_value().into_int_value(),
                    "offset",
                )?
            };
        }

        // Pass 1: Sum for mean
        self.builder.position_at_end(mean_body);
        let inner_i = self.builder.build_phi(self.context.i64_type(), "i.inner")?;
        let sum_acc = self.builder.build_phi(val_ty, "sum.acc")?;
        let offset = src_offset!(inner_i);
        let src_val = self
            .build_load(&src.clone().set_offset(offset))?
            .into_float_value();
        let sum_next = self.builder.build_float_add(
            sum_acc.as_basic_value().into_float_value(),
            src_val,
            "sum.next",
        )?;
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            self.context.i64_type().const_int(inner_bound, false),
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, mean_done, mean_body)?;
        inner_i.add_incoming(&[
            (&self.context.i64_type().const_zero(), outer_header),
            (&inner_i_next, mean_body),
        ]);
        sum_acc.add_incoming(&[
            (&val_ty.const_float(0.0), outer_header),
            (&sum_next, mean_body),
        ]);

        // mean_done: compute mean, branch to var_body
        self.builder.position_at_end(mean_done);
        let mean = self
            .builder
            .build_float_div(sum_next, inner_bound_f, "mean")?;
        self.builder.build_unconditional_branch(var_body)?;

        // Pass 2: Sum of (x - mean)^2
        self.builder.position_at_end(var_body);
        let inner_i = self.builder.build_phi(self.context.i64_type(), "i.inner")?;
        let var_acc = self.builder.build_phi(val_ty, "var.acc")?;
        let offset = src_offset!(inner_i);
        let src_val = self
            .build_load(&src.clone().set_offset(offset))?
            .into_float_value();
        let diff = self.builder.build_float_sub(src_val, mean, "diff")?;
        let diff_sq = self.builder.build_float_mul(diff, diff, "diff.sq")?;
        let var_next = self.builder.build_float_add(
            var_acc.as_basic_value().into_float_value(),
            diff_sq,
            "var.next",
        )?;
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            self.context.i64_type().const_int(inner_bound, false),
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, var_done, var_body)?;
        inner_i.add_incoming(&[
            (&self.context.i64_type().const_zero(), mean_done),
            (&inner_i_next, var_body),
        ]);
        var_acc.add_incoming(&[(&val_ty.const_float(0.0), mean_done), (&var_next, var_body)]);

        // var_done: compute inv_std = 1 / sqrt(variance + epsilon), branch to norm_body
        self.builder.position_at_end(var_done);
        let variance = self
            .builder
            .build_float_div(var_next, inner_bound_f, "variance")?;
        let var_eps = self.builder.build_float_add(variance, epsilon, "var.eps")?;
        let std_dev = self
            .build_tail_call(sqrt, &[var_eps.into()], "std.dev")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let inv_std = self
            .builder
            .build_float_div(val_ty.const_float(1.0), std_dev, "inv.std")?;
        self.builder.build_unconditional_branch(norm_body)?;

        // Pass 3: Normalize, scale, bias
        self.builder.position_at_end(norm_body);
        let inner_i = self.builder.build_phi(self.context.i64_type(), "i.inner")?;
        let offset = src_offset!(inner_i);
        let src_val = self
            .build_load(&src.clone().set_offset(offset))?
            .into_float_value();
        let diff = self.builder.build_float_sub(src_val, mean, "diff")?;
        let scaled = self.builder.build_float_mul(diff, inv_std, "scaled")?;
        let scale_val = self
            .build_load(
                &scale
                    .clone()
                    .set_offset(inner_i.as_basic_value().into_int_value()),
            )?
            .into_float_value();
        let bias_val = self
            .build_load(
                &bias
                    .clone()
                    .set_offset(inner_i.as_basic_value().into_int_value()),
            )?
            .into_float_value();
        let result = self
            .build_tail_call(
                fma,
                &[scaled.into(), scale_val.into(), bias_val.into()],
                "result",
            )?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.build_store(
            &dst.clone().set_offset(offset),
            result.as_basic_value_enum(),
        )?;
        let inner_i_next = self.builder.build_int_add(
            inner_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "inner.i.next",
        )?;
        let inner_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_next,
            self.context.i64_type().const_int(inner_bound, false),
            "ec.inner",
        )?;
        self.builder
            .build_conditional_branch(inner_ec, outer_latch, norm_body)?;
        inner_i.add_incoming(&[
            (&self.context.i64_type().const_zero(), var_done),
            (&inner_i_next, norm_body),
        ]);

        // Outer latch
        self.builder.position_at_end(outer_latch);
        let outer_i_next = self.builder.build_int_add(
            outer_i.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "outer.i.next",
        )?;
        let outer_ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            outer_i_next,
            self.context.i64_type().const_int(outer_bound, false),
            "ec.outer",
        )?;
        self.builder
            .build_conditional_branch(outer_ec, exit, outer_header)?;
        outer_i.add_incoming(&[
            (&self.context.i64_type().const_zero(), entry),
            (&outer_i_next, outer_latch),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_gather(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        indicies: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        gather: &operator::Gather,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let axis = gather.axis.index(src.ty.dims.ndim());
        let exit = self.context.append_basic_block(*self.func, "gather.exit");
        let param = Gather {
            dst,
            src,
            indicies,
            entry,
            exit,
            axis,
            depth: 0,
        };
        self.build_gather_outer(param)?;
        Ok(exit)
    }

    fn build_gather_outer(&self, param: Gather<'ctx>) -> Result<(), BuilderError> {
        if param.axis as u64 == param.depth {
            return self.build_gather_middle(param);
        }

        let Gather {
            dst,
            src,
            indicies,
            entry,
            exit,
            axis,
            depth,
        } = param;
        let header = self.context.append_basic_block(*self.func, "header");
        let latch = self.context.append_basic_block(*self.func, "latch");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(header)?;

        self.builder.position_at_end(header);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let bound = src.ty.dims[depth as usize] as u64;
        let src_offset = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(src.ty.stride(depth as usize).try_into().unwrap(), false),
            "src.offset",
        )?;
        let src_offset = self
            .builder
            .build_int_add(src.offset, src_offset, "src.offset")?;
        let src = src.set_offset(src_offset);
        let dst_offset = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(dst.ty.stride(depth as usize).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        self.builder.position_at_end(latch);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            ind_next,
            self.context.i64_type().const_int(bound, false),
            "ec",
        )?;
        self.builder.build_conditional_branch(ec, exit, header)?;
        ind.add_incoming(&[
            (&ind_next, latch),
            (&self.context.i64_type().const_zero(), entry),
        ]);

        self.build_gather_outer(Gather {
            dst,
            src,
            indicies,
            entry: header,
            exit: latch,
            axis,
            depth: depth + 1,
        })
    }

    fn build_gather_middle(&self, param: Gather<'ctx>) -> Result<(), BuilderError> {
        self.builder.position_at_end(param.entry);
        if param.axis + param.indicies.ty.dims.ndim() == param.depth as usize {
            let pred = self.context.append_basic_block(*self.func, "pred");
            self.builder.build_unconditional_branch(pred)?;

            self.builder.position_at_end(pred);
            let ind = self.build_load(&param.indicies)?.into_int_value();
            let ind = self.builder.build_int_s_extend_or_bit_cast(
                ind,
                self.context.i64_type(),
                "ind.i64",
            )?;
            let is_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ind,
                self.context.i64_type().const_zero(),
                "is_neg",
            )?;
            let ind = self
                .builder
                .build_select(
                    is_neg,
                    self.builder.build_int_add(
                        ind,
                        self.context
                            .i64_type()
                            .const_int(param.src.ty.dims[param.axis] as u64, false),
                        "ind.add",
                    )?,
                    ind,
                    "ind",
                )?
                .into_int_value();
            let offset = self.builder.build_int_mul(
                ind,
                self.context
                    .i64_type()
                    .const_int(param.src.ty.stride(param.axis).try_into().unwrap(), false),
                "ind.offset",
            )?;
            let offset = self
                .builder
                .build_int_add(param.src.offset, offset, "ind.offset")?;
            let mut param = param;
            param.entry = pred;
            param.src = param.src.set_offset(offset);
            return self.build_gather_inner(param);
        }

        let Gather {
            dst,
            src,
            indicies,
            entry,
            exit,
            axis,
            depth,
        } = param;
        let header = self.context.append_basic_block(*self.func, "header");
        let latch = self.context.append_basic_block(*self.func, "latch");
        let indices_idx = depth as usize - axis;
        let bound = indicies.ty.dims[indices_idx] as u64;

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(header)?;

        self.builder.position_at_end(header);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let offset = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(indicies.ty.stride(indices_idx).try_into().unwrap(), false),
            "ind.offset",
        )?;
        let offset = self
            .builder
            .build_int_add(indicies.offset, offset, "ind.offset")?;
        let indicies = indicies.set_offset(offset);
        let dst_offset = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(dst.ty.stride(depth as usize).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        self.builder.position_at_end(latch);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            ind_next,
            self.context.i64_type().const_int(bound, false),
            "ec",
        )?;
        self.builder.build_conditional_branch(ec, exit, header)?;
        ind.add_incoming(&[
            (&ind_next, latch),
            (&self.context.i64_type().const_zero(), entry),
        ]);

        self.build_gather_middle(Gather {
            dst,
            src,
            indicies,
            entry: header,
            exit: latch,
            axis,
            depth: depth + 1,
        })
    }

    fn build_gather_inner(&self, param: Gather<'ctx>) -> Result<(), BuilderError> {
        let Gather {
            dst,
            src,
            indicies,
            entry,
            exit,
            depth,
            ..
        } = param;

        self.builder.position_at_end(entry);
        if dst.ty.dims.ndim() == depth as usize {
            let val = self.build_load(&src)?;
            self.build_store(&dst, val)?;
            self.builder.build_unconditional_branch(exit)?;
            return Ok(());
        }

        let header = self.context.append_basic_block(*self.func, "header");
        let latch = self.context.append_basic_block(*self.func, "latch");
        let src_idx = depth as usize - indicies.ty.dims.ndim() + 1;
        let bound = dst.ty.dims[depth as usize];
        assert!(src.ty.dims[src_idx] == bound);
        self.builder.build_unconditional_branch(header)?;

        self.builder.position_at_end(header);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let src_offset = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(src.ty.stride(src_idx).try_into().unwrap(), false),
            "src.offset",
        )?;
        let src_offset = self
            .builder
            .build_int_add(src.offset, src_offset, "src.offset")?;
        let src = src.set_offset(src_offset);
        let dst_offset = self.builder.build_int_mul(
            ind.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(dst.ty.stride(depth as usize).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        self.builder.position_at_end(latch);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            ind_next,
            self.context.i64_type().const_int(bound as u64, false),
            "ec",
        )?;
        self.builder.build_conditional_branch(ec, exit, header)?;
        ind.add_incoming(&[
            (&ind_next, latch),
            (&self.context.i64_type().const_zero(), entry),
        ]);

        self.build_gather_inner(Gather {
            dst,
            src,
            indicies,
            entry: header,
            exit: latch,
            axis: param.axis,
            depth: depth + 1,
        })
    }

    pub fn build_split(
        &self,
        dsts: &[TensorPtr<'ctx>],
        src: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        split: &operator::Split,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(dsts.iter().all(|dst| dst.ty.is_contiguous()));

        let axis = split.axis.index(src.ty.dims.ndim());
        let dst_dims_acc = dsts
            .iter()
            .map(|dst| dst.ty.dims[axis])
            .scan(0, |acc, d| {
                let res = *acc;
                *acc += d;
                Some(res)
            })
            .skip(1)
            .collect::<Vec<_>>();

        let bb = self.context.append_basic_block(*self.func, "loop");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(bb)?;

        self.builder.position_at_end(bb);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let indexes = izip!(src.ty.dims.iter(), src.ty.strides().iter())
            .map(|(d, s)| {
                let i = self.builder.build_int_unsigned_div(
                    ind.as_basic_value().into_int_value(),
                    self.context.i64_type().const_int((*s as u64).max(1), false),
                    "index",
                )?;
                self.builder.build_int_unsigned_rem(
                    i,
                    self.context.i64_type().const_int(*d as u64, false),
                    "index",
                )
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
        let dim = indexes[axis];
        let mut dst_ptr = dsts[0].ptr;
        let mut dst_axis_dim = self
            .context
            .i64_type()
            .const_int(dsts[0].ty.dims[axis] as u64, false);
        let mut dst_axis_idx = dim;
        for i in 1..dsts.len() {
            let bound = dst_dims_acc[i - 1];
            let exceed = self.builder.build_int_compare(
                inkwell::IntPredicate::ULE,
                self.context.i64_type().const_int(bound as u64, false),
                dim,
                "exceed",
            )?;
            dst_ptr = self
                .builder
                .build_select(exceed, dsts[i].ptr, dst_ptr, "dst_ptr")?
                .into_pointer_value();
            dst_axis_dim = self
                .builder
                .build_select(
                    exceed,
                    self.context
                        .i64_type()
                        .const_int(dsts[i].ty.dims[axis] as u64, false),
                    dst_axis_dim,
                    "dst_axis_dim",
                )?
                .into_int_value();
            dst_axis_idx = self
                .builder
                .build_select(
                    exceed,
                    self.builder.build_int_sub(
                        dim,
                        self.context.i64_type().const_int(bound as u64, false),
                        "dst_axis_idx",
                    )?,
                    dst_axis_idx,
                    "dst_axis_idx",
                )?
                .into_int_value();
        }
        let offset = indexes.iter().enumerate().try_fold(
            self.context.i64_type().const_zero(),
            |acc, (i, idx)| {
                let dim = if i == axis {
                    dst_axis_dim
                } else {
                    self.context
                        .i64_type()
                        .const_int(src.ty.dims[i] as u64, false)
                };
                let res = self.builder.build_int_mul(acc, dim, "offset")?;

                let add = if i == axis { dst_axis_idx } else { *idx };
                self.builder.build_int_add(res, add, "offset")
            },
        )?;
        let src = src.set_offset(ind.as_basic_value().into_int_value());
        let src_val = self.build_load(&src)?;
        self.build_raw_store(
            src.ty.elem_type.llvm_type(self.context),
            dst_ptr,
            offset,
            src_val,
        )?;
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            ind_next,
            self.context
                .i64_type()
                .const_int(src.ty.dims.size() as u64, false),
            "ec",
        )?;
        self.builder.build_conditional_branch(ec, exit, bb)?;
        ind.add_incoming(&[
            (&self.context.i64_type().const_zero(), entry),
            (&ind_next, bb),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    // Assumes dst is not a slice of a larger tensor (i.e., dst strides must be
    // consistent with dst dims) because `ind` is used directly as the store offset.
    pub fn build_contiguous(
        &self,
        dst: &TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        ops: &[ReinterpretType],
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let body = self.context.append_basic_block(*self.func, "body");
        let exit = self.context.append_basic_block(*self.func, "exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let mut offset = {
            let mut indexes = Vec::new();
            for (d, s) in izip!(dst.ty.dims.iter(), dst.ty.strides().iter()) {
                let idx = self.builder.build_int_unsigned_div(
                    ind.as_basic_value().into_int_value(),
                    self.context.i64_type().const_int((*s as u64).max(1), false),
                    "index",
                )?;
                let idx = self.builder.build_int_unsigned_rem(
                    idx,
                    self.context.i64_type().const_int(*d as u64, false),
                    "index",
                )?;
                indexes.push(idx);
            }
            let mut acc = self.context.i64_type().const_zero();
            for (i, d) in indexes.iter().zip(dst.ty.dims.iter()) {
                acc = self.builder.build_int_mul(
                    acc,
                    self.context.i64_type().const_int(*d as u64, false),
                    "offset",
                )?;
                acc = self.builder.build_int_add(acc, *i, "offset")?;
            }
            acc
        };
        let mut shape = dst.ty.dims.iter().map(|d| *d as u64).collect_vec();
        for op in ops.iter().rev() {
            match op {
                ReinterpretType::Reshape { before, .. } => {
                    shape = before.iter().map(|d| *d as u64).collect_vec();
                }
                ReinterpretType::Transpose(transpose) => {
                    let mut cur = offset;
                    let mut new_indexes = Vec::new();
                    for d in shape.iter().rev() {
                        let idx = self.builder.build_int_unsigned_rem(
                            cur,
                            self.context.i64_type().const_int(*d, false),
                            "index",
                        )?;
                        new_indexes.push(idx);
                        cur = self.builder.build_int_unsigned_div(
                            cur,
                            self.context.i64_type().const_int(*d, false),
                            "index",
                        )?;
                    }
                    new_indexes.reverse();

                    let perm = transpose
                        .perm
                        .clone()
                        .unwrap_or_else(|| (0..shape.len()).collect_vec());
                    let new_shape = shape.clone();
                    let mut indexes = vec![self.context.i64_type().const_zero(); shape.len()];
                    for (i, p) in perm.iter().enumerate() {
                        // forward perm: new_idx[i] = idx[perm[i]]
                        // inverse perm: idx[perm[i]] = new_idx[i]
                        indexes[*p] = new_indexes[i];
                        shape[*p] = new_shape[i];
                    }

                    offset = self.context.i64_type().const_zero();
                    for (i, d) in izip!(indexes.iter(), shape.iter()) {
                        offset = self.builder.build_int_mul(
                            offset,
                            self.context.i64_type().const_int(*d, false),
                            "offset",
                        )?;
                        offset = self.builder.build_int_add(offset, *i, "offset")?;
                    }
                }
                ReinterpretType::Broadcast { before, .. } => {
                    // Compute multi-index from output offset, then mod each dim
                    // by the before-shape to get the source index
                    let mut cur = offset;
                    let mut indexes = Vec::new();
                    for d in shape.iter().rev() {
                        let idx = self.builder.build_int_unsigned_rem(
                            cur,
                            self.context.i64_type().const_int(*d, false),
                            "index",
                        )?;
                        indexes.push(idx);
                        cur = self.builder.build_int_unsigned_div(
                            cur,
                            self.context.i64_type().const_int(*d, false),
                            "index",
                        )?;
                    }
                    indexes.reverse();

                    // Map output index to input index via broadcast (mod by before dim)
                    shape = before.iter().map(|d| *d as u64).collect_vec();
                    offset = self.context.i64_type().const_zero();
                    for (idx, d) in izip!(indexes.iter(), before.iter()) {
                        let d_val = self.context.i64_type().const_int(*d as u64, false);
                        let src_idx = if *d == 1 {
                            self.context.i64_type().const_zero()
                        } else {
                            *idx
                        };
                        offset = self.builder.build_int_mul(offset, d_val, "offset")?;
                        offset = self.builder.build_int_add(offset, src_idx, "offset")?;
                    }
                }
            }
        }

        let offset = {
            let mut indexes = Vec::new();
            let mut cur = offset;
            for d in shape.iter().rev() {
                let idx = self.builder.build_int_unsigned_rem(
                    cur,
                    self.context.i64_type().const_int(*d, false),
                    "index",
                )?;
                indexes.push(idx);
                cur = self.builder.build_int_unsigned_div(
                    cur,
                    self.context.i64_type().const_int(*d, false),
                    "index",
                )?;
            }
            indexes.reverse();
            let mut acc = self.context.i64_type().const_zero();
            for (i, s) in indexes.iter().zip(src.ty.strides().iter()) {
                let add = self.builder.build_int_mul(
                    *i,
                    self.context.i64_type().const_int(*s as u64, false),
                    "add",
                )?;
                acc = self.builder.build_int_add(acc, add, "offset")?;
            }
            acc
        };

        let elem_type = dst.ty.elem_type.llvm_type(self.context);
        let src_offset = self
            .builder
            .build_int_add(src.offset, offset, "src.offset")?;
        let src = src.set_offset(src_offset);
        let src_val = self.build_load(&src)?;
        let dst_offset = self.builder.build_int_add(
            dst.offset,
            ind.as_basic_value().into_int_value(),
            "dst.offset",
        )?;
        self.build_raw_store(elem_type, dst.ptr, dst_offset, src_val)?;

        let ind_inc = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let ec = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            ind_inc,
            self.context
                .i64_type()
                .const_int(dst.ty.dims.size().max(1) as u64, false),
            "ec",
        )?;
        self.builder.build_conditional_branch(ec, exit, body)?;
        ind.add_incoming(&[
            (&self.context.i64_type().const_zero(), entry),
            (&ind_inc, body),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }
}

#[derive(Clone)]
struct OMPContext<'ctx> {
    global_tid: PointerValue<'ctx>,
    is_last: PointerValue<'ctx>,
    lb: PointerValue<'ctx>,
    ub: PointerValue<'ctx>,
    stride: PointerValue<'ctx>,
}

struct OMPForRange<'ctx> {
    lb: IntValue<'ctx>,
    ub: IntValue<'ctx>,
    body_bb: BasicBlock<'ctx>,
    epilog_bb: BasicBlock<'ctx>,
}

#[derive(Debug)]
struct LoopBB<'ctx> {
    preheader: BasicBlock<'ctx>,
    header: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,
}

#[derive(Debug)]
struct ResizeParam<'a, 'ctx> {
    dst: TensorPtr<'ctx>,
    src: TensorPtr<'ctx>,
    axes: Vec<usize>,
    loop_bb: LoopBB<'ctx>,
    dim: usize,
    resize: &'a operator::Resize,
}

struct Gather<'ctx> {
    dst: TensorPtr<'ctx>,
    src: TensorPtr<'ctx>,
    indicies: TensorPtr<'ctx>,
    entry: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,
    axis: usize,
    depth: u64,
}
