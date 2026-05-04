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
use crate::graph::operator;
use crate::graph::operator::Layout;
use crate::graph::operator::ReinterpretType;
use crate::schedule::ElementwiseOpArg;
use crate::tensor::data::ScalarData;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::UIntType;

#[derive(Clone, Copy)]
pub enum PoolMode {
    Max,
    Avg,
}

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

struct Im2Col {
    nbatch: usize,
    one_fm_shape: ResolvedTensorDims,
    pad: operator::ConvPad,
    channel: usize,
    dilations: operator::OptionalVec<usize>,
    one_kernel_shape: ResolvedTensorDims,
    strides: operator::OptionalVec<usize>,
    pad_val: operator::PadVal,
    layout: operator::Layout,
}

impl Im2Col {
    fn padded_len(&self, dim: usize) -> usize {
        let unit = self.dilations[dim] * (self.one_kernel_shape[dim] - 1) + 1;
        self.strides[dim] * (self.one_fm_shape[dim] - 1) + unit
    }
}

impl<'ctx> FunctionTranslator<'_, 'ctx> {
    fn init_counted_loop(
        &self,
        header: BasicBlock<'ctx>,
    ) -> Result<(PhiValue<'ctx>, IntValue<'ctx>), BuilderError> {
        self.builder.position_at_end(header);
        let phi = self.builder.build_phi(self.context.i64_type(), "ind")?;
        Ok((phi, phi.as_basic_value().into_int_value()))
    }

    fn finalize_counted_loop(
        &self,
        phi: PhiValue<'ctx>,
        entry: BasicBlock<'ctx>,
        bound: IntValue<'ctx>,
        header: BasicBlock<'ctx>,
        exit: BasicBlock<'ctx>,
        latch: BasicBlock<'ctx>,
    ) -> Result<(), BuilderError> {
        let i64_ty = self.context.i64_type();
        let ind = phi.as_basic_value().into_int_value();
        self.builder.position_at_end(latch);
        let next = self
            .builder
            .build_int_add(ind, i64_ty.const_int(1, false), "next")?;
        let ec = self
            .builder
            .build_int_compare(inkwell::IntPredicate::EQ, next, bound, "ec")?;
        self.builder.build_conditional_branch(ec, exit, header)?;
        phi.add_incoming(&[(&i64_ty.const_zero(), entry), (&next, latch)]);
        Ok(())
    }

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
        let raw = self.builder.build_load(
            ptr.ty.elem_type.llvm_type(self.context),
            gep,
            format!("load.{}", ptr.name).as_str(),
        )?;
        if matches!(ptr.ty.elem_type, DataType::Float(FloatType::BF16)) {
            Ok(self.bf16_bits_to_f32(raw.into_int_value())?.into())
        } else {
            Ok(raw)
        }
    }

    fn build_store<V: BasicValue<'ctx>>(
        &self,
        ptr: &TensorPtr<'ctx>,
        val: V,
    ) -> Result<(), BuilderError> {
        let gep = self.build_gep(ptr)?;
        if matches!(ptr.ty.elem_type, DataType::Float(FloatType::BF16)) {
            let bits = self.f32_to_bf16_bits(val.as_basic_value_enum().into_float_value())?;
            self.builder.build_store(gep, bits).map(|_| ())
        } else {
            self.builder.build_store(gep, val).map(|_| ())
        }
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

    // bf16 bit pattern (i16) -> f32 value via shift-and-bitcast.
    fn bf16_bits_to_f32(&self, bits: IntValue<'ctx>) -> Result<FloatValue<'ctx>, BuilderError> {
        let i32_ty = self.context.i32_type();
        let zext = self.builder.build_int_z_extend(bits, i32_ty, "bf16.zext")?;
        let shifted =
            self.builder
                .build_left_shift(zext, i32_ty.const_int(16, false), "bf16.shl")?;
        let bc = self
            .builder
            .build_bit_cast(shifted, self.context.f32_type(), "bf16.f32")?;
        Ok(bc.into_float_value())
    }

    // f32 value -> bf16 bit pattern (i16) with round-to-nearest-even.
    fn f32_to_bf16_bits(&self, v: FloatValue<'ctx>) -> Result<IntValue<'ctx>, BuilderError> {
        let i32_ty = self.context.i32_type();
        let i16_ty = self.context.i16_type();
        let bits = self
            .builder
            .build_bit_cast(v, i32_ty, "bf16.bits")?
            .into_int_value();
        // RNE: bias = 0x7FFF + ((bits >> 16) & 1)
        let lsb = self.builder.build_right_shift(
            bits,
            i32_ty.const_int(16, false),
            false,
            "bf16.lsb.shr",
        )?;
        let lsb = self
            .builder
            .build_and(lsb, i32_ty.const_int(1, false), "bf16.lsb")?;
        let bias = self
            .builder
            .build_int_add(lsb, i32_ty.const_int(0x7FFF, false), "bf16.bias")?;
        let rounded = self.builder.build_int_add(bits, bias, "bf16.rounded")?;
        let high = self.builder.build_right_shift(
            rounded,
            i32_ty.const_int(16, false),
            false,
            "bf16.high",
        )?;
        Ok(self
            .builder
            .build_int_truncate(high, i16_ty, "bf16.trunc")?)
    }

    pub fn build_dequantize_linear(
        &self,
        dst: &TensorPtr<'ctx>,
        x: &TensorPtr<'ctx>,
        scale: &TensorPtr<'ctx>,
        axis: usize,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let total = dst.ty.dims.size() as u64;
        let axis_dim = dst.ty.dims[axis] as u64;
        let inner_size: u64 = dst.ty.dims[axis + 1..]
            .iter()
            .map(|d| *d as u64)
            .product::<u64>()
            .max(1);
        let i64_ty = self.context.i64_type();
        let preheader = self.builder.get_insert_block().unwrap_or(entry);
        self.builder.position_at_end(entry);
        let header = self.context.append_basic_block(*self.func, "dequant.h");
        let after = self.context.append_basic_block(*self.func, "dequant.x");
        self.builder.build_unconditional_branch(header)?;
        let (phi, idx) = self.init_counted_loop(header)?;

        let x_loaded = self.build_load(&x.clone().set_offset(idx))?;
        let f_val = match x.ty.elem_type {
            DataType::SInt(_) | DataType::Bool => self
                .builder
                .build_signed_int_to_float(
                    x_loaded.into_int_value(),
                    self.context.f32_type(),
                    "dequant.x.f",
                )?
                .as_basic_value_enum(),
            DataType::UInt(_) => self
                .builder
                .build_unsigned_int_to_float(
                    x_loaded.into_int_value(),
                    self.context.f32_type(),
                    "dequant.x.f",
                )?
                .as_basic_value_enum(),
            DataType::Float(_) => x_loaded,
        };

        let axis_idx = if axis_dim == 1 {
            i64_ty.const_zero()
        } else {
            let div = self.builder.build_int_unsigned_div(
                idx,
                i64_ty.const_int(inner_size, false),
                "dequant.div",
            )?;
            self.builder.build_int_unsigned_rem(
                div,
                i64_ty.const_int(axis_dim, false),
                "dequant.axis",
            )?
        };
        let scale_v = self.build_load(&scale.clone().set_offset(axis_idx))?;
        let prod = self.builder.build_float_mul(
            f_val.into_float_value(),
            scale_v.into_float_value(),
            "dequant.prod",
        )?;
        self.build_store(&dst.clone().set_offset(idx), prod)?;

        self.finalize_counted_loop(
            phi,
            preheader,
            i64_ty.const_int(total, false),
            header,
            after,
            header,
        )?;
        self.builder.position_at_end(after);
        Ok(after)
    }

    fn load_tensor_f32(
        &self,
        storage_ty: BasicTypeEnum<'ctx>,
        ptr: PointerValue<'ctx>,
        is_bf16: bool,
        name: &str,
    ) -> Result<FloatValue<'ctx>, BuilderError> {
        let raw = self.builder.build_load(storage_ty, ptr, name)?;
        if is_bf16 {
            self.bf16_bits_to_f32(raw.into_int_value())
        } else {
            Ok(raw.into_float_value())
        }
    }

    fn store_tensor_f32(
        &self,
        ptr: PointerValue<'ctx>,
        val: FloatValue<'ctx>,
        is_bf16: bool,
    ) -> Result<(), BuilderError> {
        if is_bf16 {
            let bits = self.f32_to_bf16_bits(val)?;
            self.builder.build_store(ptr, bits)?;
        } else {
            self.builder.build_store(ptr, val)?;
        }
        Ok(())
    }

    // Returns ptr = workspace.ptr + workspace.offset + offset_elems (in f32 elements).
    fn workspace_offset(
        &self,
        workspace: &TensorPtr<'ctx>,
        offset_elems: u64,
        name: &str,
    ) -> Result<PointerValue<'ctx>, BuilderError> {
        let i64_ty = self.context.i64_type();
        let off = self.builder.build_int_add(
            workspace.offset,
            i64_ty.const_int(offset_elems, false),
            "ws.off",
        )?;
        unsafe {
            self.builder
                .build_in_bounds_gep(self.context.f32_type(), workspace.ptr, &[off], name)
        }
    }

    fn bf16_buf_to_f32(
        &self,
        src_bf16: PointerValue<'ctx>,
        dst_f32: PointerValue<'ctx>,
        count: u64,
    ) -> Result<(), BuilderError> {
        if count == 0 {
            return Ok(());
        }
        let i16_ty = self.context.i16_type();
        let f32_ty = self.context.f32_type();
        let i64_ty = self.context.i64_type();
        let preheader = self.builder.get_insert_block().unwrap();
        let header = self.context.append_basic_block(*self.func, "bf16.h2f.h");
        let after = self.context.append_basic_block(*self.func, "bf16.h2f.x");
        self.builder.build_unconditional_branch(header)?;
        let (phi, idx) = self.init_counted_loop(header)?;
        let src_gep = unsafe {
            self.builder
                .build_in_bounds_gep(i16_ty, src_bf16, &[idx], "h2f.src")?
        };
        let bits = self
            .builder
            .build_load(i16_ty, src_gep, "h2f.bits")?
            .into_int_value();
        let f = self.bf16_bits_to_f32(bits)?;
        let dst_gep = unsafe {
            self.builder
                .build_in_bounds_gep(f32_ty, dst_f32, &[idx], "h2f.dst")?
        };
        self.builder.build_store(dst_gep, f)?;
        self.finalize_counted_loop(
            phi,
            preheader,
            i64_ty.const_int(count, false),
            header,
            after,
            header,
        )?;
        self.builder.position_at_end(after);
        Ok(())
    }

    fn f32_buf_to_bf16(
        &self,
        src_f32: PointerValue<'ctx>,
        dst_bf16: PointerValue<'ctx>,
        count: u64,
    ) -> Result<(), BuilderError> {
        if count == 0 {
            return Ok(());
        }
        let i16_ty = self.context.i16_type();
        let f32_ty = self.context.f32_type();
        let i64_ty = self.context.i64_type();
        let preheader = self.builder.get_insert_block().unwrap();
        let header = self.context.append_basic_block(*self.func, "bf16.f2h.h");
        let after = self.context.append_basic_block(*self.func, "bf16.f2h.x");
        self.builder.build_unconditional_branch(header)?;
        let (phi, idx) = self.init_counted_loop(header)?;
        let src_gep = unsafe {
            self.builder
                .build_in_bounds_gep(f32_ty, src_f32, &[idx], "f2h.src")?
        };
        let f = self
            .builder
            .build_load(f32_ty, src_gep, "f2h.val")?
            .into_float_value();
        let bits = self.f32_to_bf16_bits(f)?;
        let dst_gep = unsafe {
            self.builder
                .build_in_bounds_gep(i16_ty, dst_bf16, &[idx], "f2h.dst")?
        };
        self.builder.build_store(dst_gep, bits)?;
        self.finalize_counted_loop(
            phi,
            preheader,
            i64_ty.const_int(count, false),
            header,
            after,
            header,
        )?;
        self.builder.position_at_end(after);
        Ok(())
    }

    fn build_tail_call(
        &self,
        function: FunctionValue<'ctx>,
        args: &[BasicMetadataValueEnum<'ctx>],
        name: &str,
    ) -> Result<CallSiteValue<'ctx>, BuilderError> {
        let call = self.builder.build_call(function, args, name)?;
        call.set_tail_call(false);
        Ok(call)
    }

    fn sigmoid(
        &self,
        src: FloatValue<'ctx>,
        ty: FloatType,
    ) -> Result<FloatValue<'ctx>, BuilderError> {
        let exp = self.intrinsics.exp.get(ty);
        let ty = ty.llvm_type(self.context);
        let src = self.builder.build_float_neg(src, "neg")?;
        let exp = self
            .build_tail_call(exp, &[src.into()], "exp")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let one = ty.const_float(1.0);
        let den = self.builder.build_float_add(one, exp, "den")?;
        self.builder.build_float_div(one, den, "res")
    }

    fn build_single_op(
        &self,
        opcode: SingleOpcode,
        ty: DataType,
        input_ty: Option<DataType>,
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

            SingleOpcode::And => {
                let (lhs, rhs) = binary_op!(operands);
                self.builder
                    .build_and(lhs.into_int_value(), rhs.into_int_value(), "and")?
                    .as_basic_value_enum()
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

                    (DataType::Bool, DataType::SInt(sto)) => self
                        .builder
                        .build_int_cast(src.into_int_value(), sto.llvm_type(self.context), "cast")?
                        .as_basic_value_enum(),
                    (DataType::Bool, DataType::UInt(uto)) => self
                        .builder
                        .build_int_cast(src.into_int_value(), uto.llvm_type(self.context), "cast")?
                        .as_basic_value_enum(),
                    (DataType::Bool, DataType::Float(fto)) => self
                        .builder
                        .build_unsigned_int_to_float(
                            src.into_int_value(),
                            fto.llvm_type(self.context),
                            "cast",
                        )?
                        .as_basic_value_enum(),
                    (DataType::Bool, DataType::Bool) => src,

                    (DataType::SInt(_) | DataType::UInt(_), DataType::Bool) => {
                        let zero = src.into_int_value().get_type().const_zero();
                        let cmp = self.builder.build_int_compare(
                            inkwell::IntPredicate::NE,
                            src.into_int_value(),
                            zero,
                            "cast",
                        )?;
                        self.builder
                            .build_int_z_extend(cmp, self.context.i8_type(), "cast")?
                            .as_basic_value_enum()
                    }
                    (DataType::Float(_), DataType::Bool) => {
                        let zero = src.into_float_value().get_type().const_zero();
                        let cmp = self.builder.build_float_compare(
                            inkwell::FloatPredicate::ONE,
                            src.into_float_value(),
                            zero,
                            "cast",
                        )?;
                        self.builder
                            .build_int_z_extend(cmp, self.context.i8_type(), "cast")?
                            .as_basic_value_enum()
                    }

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

            SingleOpcode::Equal => {
                let is_float = matches!(input_ty.unwrap(), DataType::Float(_));
                let (lhs, rhs) = binary_op!(operands);
                if is_float {
                    self.builder
                        .build_float_compare(
                            inkwell::FloatPredicate::OEQ,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "cmp",
                        )?
                        .as_basic_value_enum()
                } else {
                    self.builder
                        .build_int_compare(
                            inkwell::IntPredicate::EQ,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "cmp",
                        )?
                        .as_basic_value_enum()
                }
            }

            SingleOpcode::LessOrEqual => {
                let is_float = matches!(input_ty.unwrap(), DataType::Float(_));
                let (lhs, rhs) = binary_op!(operands);
                if is_float {
                    self.builder
                        .build_float_compare(
                            inkwell::FloatPredicate::OLE,
                            lhs.into_float_value(),
                            rhs.into_float_value(),
                            "cmp",
                        )?
                        .as_basic_value_enum()
                } else {
                    self.builder
                        .build_int_compare(
                            inkwell::IntPredicate::SLE,
                            lhs.into_int_value(),
                            rhs.into_int_value(),
                            "cmp",
                        )?
                        .as_basic_value_enum()
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

            SingleOpcode::Neg => {
                let src = unary_op!(operands).into_float_value();
                self.builder
                    .build_float_neg(src, "neg")?
                    .as_basic_value_enum()
            }

            SingleOpcode::IsNaN => {
                assert!(matches!(input_ty.unwrap(), DataType::Float(_)));
                let val = unary_op!(operands).into_float_value();
                let is_nan = self.builder.build_float_compare(
                    inkwell::FloatPredicate::UNO,
                    val,
                    val,
                    "isnan",
                )?;
                self.builder
                    .build_int_z_extend(is_nan, self.context.i8_type(), "isnan_i8")?
                    .as_basic_value_enum()
            }

            opcode @ (SingleOpcode::Cos |
            SingleOpcode::Exp |
            SingleOpcode::Log |
            SingleOpcode::Sin |
            SingleOpcode::Sqrt) => {
                let src = unary_op!(operands);
                let ty = ty.float_type().unwrap();
                let f = match opcode {
                    SingleOpcode::Cos => self.intrinsics.cos.get(ty),
                    SingleOpcode::Exp => self.intrinsics.exp.get(ty),
                    SingleOpcode::Log => self.intrinsics.log.get(ty),
                    SingleOpcode::Sin => self.intrinsics.sin.get(ty),
                    SingleOpcode::Sqrt => self.intrinsics.sqrt.get(ty),
                    _ => unreachable!(),
                };
                let src = src.into_float_value();
                self.build_tail_call(f, &[src.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }

            SingleOpcode::Clip(operator::Clip { min, max }) => {
                let min = min.expect("Clip min must be constant-folded");
                let max = max.expect("Clip max must be constant-folded");
                let fp_ty = ty.float_type().unwrap();
                let llvm_ty = fp_ty.llvm_type(self.context);
                let src = unary_op!(operands).into_float_value();
                let clamped_low = self
                    .builder
                    .build_call(
                        self.intrinsics.fmax.get(fp_ty),
                        &[src.into(), llvm_ty.const_float(min).into()],
                        "clamped_low",
                    )?
                    .try_as_basic_value()
                    .left()
                    .unwrap();
                self.builder
                    .build_call(
                        self.intrinsics.fmin.get(fp_ty),
                        &[clamped_low.into(), llvm_ty.const_float(max).into()],
                        "res",
                    )?
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
                    DataType::Bool | DataType::SInt(_) => {
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
                    DataType::Bool => self
                        .builder
                        .build_float_to_signed_int(
                            res.into_float_value(),
                            self.context.i8_type(),
                            "res",
                        )?
                        .as_basic_value_enum(),
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
                let src = unary_op!(operands).into_float_value();
                let ty = ty.float_type().unwrap();
                self.sigmoid(src, ty)?.into()
            }

            SingleOpcode::Swish(operator::Swish { alpha }) => {
                let ty = ty.float_type().unwrap();
                let src = unary_op!(operands).into_float_value();
                let alpha = ty.llvm_type(self.context).const_float(alpha as f64);
                let x = self.builder.build_float_mul(src, alpha, "x")?;
                let sigmoid = self.sigmoid(x, ty)?;
                self.builder.build_float_mul(src, sigmoid, "res")?.into()
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
        let input_ty = op.src_operands().get(0).map(|op| op.ty.elem_type);
        let res = match op.opcode {
            Opcode::Single(opcode) => {
                let operands = op
                    .src_operands()
                    .iter()
                    .map(|op| self.build_load(op))
                    .collect::<Result<Vec<_>, _>>()?;
                self.build_single_op(opcode, ty, input_ty, &operands)?
            }
            Opcode::Fused(ref ops) => {
                let mut intermediates = Vec::with_capacity(ops.len());
                let srcs = op.src_operands();
                let ty = op.result_type();
                for (opcode, args) in ops.iter() {
                    let opcode = *opcode;
                    let typed_operands = args
                        .iter()
                        .map(|arg| match arg {
                            ElementwiseOpArg::Input(i) => {
                                Ok((self.build_load(&srcs[*i])?, srcs[*i].ty.elem_type))
                            }
                            ElementwiseOpArg::NthResult(i) => Ok(intermediates[*i]),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let operands = typed_operands
                        .iter()
                        .map(|(val, _)| *val)
                        .collect::<Vec<_>>();
                    let input_ty = typed_operands.first().map(|(_, ty)| *ty);
                    let res = self.build_single_op(opcode, ty, input_ty, &operands)?;
                    intermediates.push((res, ty));
                }
                intermediates.last().unwrap().0
            }
        };
        self.build_store(op.dst_operand(), res)
    }

    fn build_im2col(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        im2col: &Im2Col,
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
        let elem_ty = src.ty.elem_type.llvm_type(self.context);

        let nbatch = im2col.nbatch as u64;
        let h_out = im2col.one_fm_shape[0] as u64;
        let w_out = im2col.one_fm_shape[1] as u64;
        let kh = im2col.one_kernel_shape[0] as u64;
        let kw = im2col.one_kernel_shape[1] as u64;
        let c_in = im2col.channel as u64;
        let (h_in, w_in, src_c_stride) = match layout {
            operator::Layout::NCHW => (src.ty.dims[2] as u64, src.ty.dims[3] as u64, c_in),
            operator::Layout::NHWC => (
                src.ty.dims[1] as u64,
                src.ty.dims[2] as u64,
                src.ty.dims[3] as u64,
            ),
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
                    let add_left =
                        pad_len % 2 == 1 && matches!(im2col.pad, operator::ConvPad::SameLower);
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
                    FloatType::BF16 => f32::NEG_INFINITY as f64,
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

        let i64_ty = self.context.i64_type();

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(hdr_n)?;

        let (phi_n, ind_n) = self.init_counted_loop(hdr_n)?;
        self.builder.build_unconditional_branch(hdr_oh)?;

        let (phi_oh, ind_oh) = self.init_counted_loop(hdr_oh)?;
        self.builder.build_unconditional_branch(hdr_ow)?;

        let (phi_ow, ind_ow) = self.init_counted_loop(hdr_ow)?;
        self.builder.build_unconditional_branch(hdr_kh)?;

        let (phi_kh, ind_kh) = self.init_counted_loop(hdr_kh)?;
        self.builder.build_unconditional_branch(hdr_kw)?;

        let (phi_kw, ind_kw) = self.init_counted_loop(hdr_kw)?;
        self.builder.build_unconditional_branch(hdr_c)?;

        let (phi_c, ind_c) = self.init_counted_loop(hdr_c)?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);

        let ih = {
            let a =
                self.builder
                    .build_int_mul(ind_oh, i64_ty.const_int(stride_h, false), "oh_s")?;
            let b =
                self.builder
                    .build_int_mul(ind_kh, i64_ty.const_int(dilation_h, false), "kh_d")?;
            let c = self.builder.build_int_add(a, b, "ih_raw")?;
            self.builder
                .build_int_sub(c, i64_ty.const_int(pad_h, false), "ih")?
        };
        let iw = {
            let a =
                self.builder
                    .build_int_mul(ind_ow, i64_ty.const_int(stride_w, false), "ow_s")?;
            let b =
                self.builder
                    .build_int_mul(ind_kw, i64_ty.const_int(dilation_w, false), "kw_d")?;
            let c = self.builder.build_int_add(a, b, "iw_raw")?;
            self.builder
                .build_int_sub(c, i64_ty.const_int(pad_w, false), "iw")?
        };

        let oob = {
            let ih_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ih,
                i64_ty.const_zero(),
                "ih_neg",
            )?;
            let ih_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                ih,
                i64_ty.const_int(h_in, false),
                "ih_big",
            )?;
            let iw_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                iw,
                i64_ty.const_zero(),
                "iw_neg",
            )?;
            let iw_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                iw,
                i64_ty.const_int(w_in, false),
                "iw_big",
            )?;
            let a = self.builder.build_or(ih_neg, ih_big, "oob_h")?;
            let b = self.builder.build_or(iw_neg, iw_big, "oob_w")?;
            self.builder.build_or(a, b, "oob")?
        };

        let bb_load = bb("nhwc.load");
        let bb_pad = bb("nhwc.pad");
        let bb_store = bb("nhwc.store");
        self.builder
            .build_conditional_branch(oob, bb_pad, bb_load)?;

        self.builder.position_at_end(bb_load);
        let src_offset = match layout {
            operator::Layout::NCHW => {
                let o = self.builder.build_int_mul(
                    ind_n,
                    i64_ty.const_int(c_in * h_in * w_in, false),
                    "so_n",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder.build_int_mul(
                        ind_c,
                        i64_ty.const_int(h_in * w_in, false),
                        "so_c",
                    )?,
                    "so_nc",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder
                        .build_int_mul(ih, i64_ty.const_int(w_in, false), "so_h")?,
                    "so_nch",
                )?;
                self.builder.build_int_add(o, iw, "src_off")?
            }
            operator::Layout::NHWC => {
                let o = self.builder.build_int_mul(
                    ind_n,
                    i64_ty.const_int(h_in * w_in * src_c_stride, false),
                    "so_n",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder.build_int_mul(
                        ih,
                        i64_ty.const_int(w_in * src_c_stride, false),
                        "so_h",
                    )?,
                    "so_nh",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder.build_int_mul(
                        iw,
                        i64_ty.const_int(src_c_stride, false),
                        "so_w",
                    )?,
                    "so_nhw",
                )?;
                self.builder.build_int_add(o, ind_c, "src_off")?
            }
        };
        let src_val = self
            .build_load(&src.clone().add_offset(self.builder, src_offset)?)?
            .into_float_value();
        self.builder.build_unconditional_branch(bb_store)?;

        self.builder.position_at_end(bb_pad);
        self.builder.build_unconditional_branch(bb_store)?;

        self.builder.position_at_end(bb_store);
        let val = self.builder.build_phi(elem_ty, "val")?;
        val.add_incoming(&[(&src_val, bb_load), (&pad_val, bb_pad)]);

        let dst_row = {
            let o = self.builder.build_int_mul(
                ind_n,
                i64_ty.const_int(h_out * w_out, false),
                "dr_n",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder
                    .build_int_mul(ind_oh, i64_ty.const_int(w_out, false), "dr_oh")?,
                "dr_noh",
            )?;
            self.builder.build_int_add(o, ind_ow, "dst_row")?
        };
        let dst_col = {
            let o = self
                .builder
                .build_int_mul(ind_c, i64_ty.const_int(kh * kw, false), "dc_c")?;
            let o = self.builder.build_int_add(
                o,
                self.builder
                    .build_int_mul(ind_kh, i64_ty.const_int(kw, false), "dc_kh")?,
                "dc_ckh",
            )?;
            self.builder.build_int_add(o, ind_kw, "dst_col")?
        };
        let dst_offset = {
            let o =
                self.builder
                    .build_int_mul(dst_row, i64_ty.const_int(dst_cols, false), "do_row")?;
            self.builder.build_int_add(o, dst_col, "dst_off")?
        };
        self.build_store(&dst.clone().set_offset(dst_offset), val.as_basic_value())?;
        self.builder.build_unconditional_branch(latch_c)?;

        let c = |v: u64| i64_ty.const_int(v, false);
        self.finalize_counted_loop(phi_c, hdr_kw, c(c_in), hdr_c, latch_kw, latch_c)?;
        self.finalize_counted_loop(phi_kw, hdr_kh, c(kw), hdr_kw, latch_kh, latch_kw)?;
        self.finalize_counted_loop(phi_kh, hdr_ow, c(kh), hdr_kh, latch_ow, latch_kh)?;
        self.finalize_counted_loop(phi_ow, hdr_oh, c(w_out), hdr_ow, latch_oh, latch_ow)?;
        self.finalize_counted_loop(phi_oh, hdr_n, c(h_out), hdr_oh, latch_n, latch_oh)?;
        self.finalize_counted_loop(phi_n, entry, c(nbatch), hdr_n, exit, latch_n)?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    /// groups=1 only:
    ///   im2col(src, workspace)
    ///   if bias: broadcast_copy(bias, dst); gemm(dst = workspace * weight^T + dst)
    ///   else: gemm(dst = workspace * weight^T)
    ///   if NCHW: transpose dst [N,H,W,C] -> [N,C,H,W] via workspace
    #[allow(clippy::too_many_arguments)]
    pub fn build_conv(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        weight: &TensorPtr<'ctx>,
        bias: Option<&TensorPtr<'ctx>>,
        workspace: &TensorPtr<'ctx>,
        conv: &operator::Conv,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(conv.group == 1);
        let i64_ty = self.context.i64_type();
        let c_in = weight.ty.dims[1];
        let c_out = weight.ty.dims[0];
        let kh = conv.kernel_shape[0];
        let kw = conv.kernel_shape[1];

        let output_shape = conv.output_shape(&src.ty.dims, &weight.ty.dims);
        let n_batch = src.ty.dims[0];
        let (h_out, w_out) = match conv.output_layout {
            Layout::NCHW => (2, 3),
            Layout::NHWC => (1, 2),
        };
        let h_out = output_shape[h_out];
        let w_out = output_shape[w_out];
        let m = n_batch * h_out * w_out;
        let k = c_in * kh * kw;

        let im2col_op = Im2Col {
            nbatch: n_batch,
            one_fm_shape: ResolvedTensorDims::new(&[h_out, w_out]),
            pad: conv.pad.clone(),
            channel: c_in,
            dilations: conv.dilations.clone(),
            one_kernel_shape: ResolvedTensorDims::new(&[kh, kw]),
            strides: conv.strides.clone(),
            pad_val: operator::PadVal::Zero,
            layout: conv.input_layout,
        };

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);

        let ty = ResolvedTensorType::new(src.ty.elem_type, ResolvedTensorDims::new(&[m, k]));
        let im2col_ptr = TensorPtr {
            ptr: workspace.ptr,
            ty,
            offset: i64_ty.const_zero(),
            name: "conv_im2col".to_string(),
        };

        let cur_bb = self.build_im2col(&im2col_ptr, src, &im2col_op, entry)?;

        let ty = ResolvedTensorType::new(weight.ty.elem_type, ResolvedTensorDims::new(&[c_out, k]));
        let weight_ptr = TensorPtr {
            ptr: weight.ptr,
            ty,
            offset: i64_ty.const_zero(),
            name: "conv_weight".to_string(),
        };

        let ty = ResolvedTensorType::new(dst.ty.elem_type, ResolvedTensorDims::new(&[m, c_out]));
        let dst_ptr = TensorPtr {
            ptr: dst.ptr,
            ty,
            offset: i64_ty.const_zero(),
            name: "conv_gemm_dst".to_string(),
        };

        let bias_broadcast = bias.map(|b| {
            let ty = ResolvedTensorType::new(b.ty.elem_type, ResolvedTensorDims::new(&[c_out]));
            let broadcast_ty = ty.broadcast(&ResolvedTensorDims::new(&[m, c_out]));
            TensorPtr {
                ptr: b.ptr,
                ty: broadcast_ty,
                offset: i64_ty.const_zero(),
                name: "conv_bias_broadcast".to_string(),
            }
        });

        let gemm_op = operator::Gemm {
            trans_a: false,
            trans_b: true,
            alpha: 1.0,
            beta: if bias.is_some() { 1.0 } else { 0.0 },
        };
        let cur_bb = self.build_gemm(
            &dst_ptr,
            &im2col_ptr,
            &weight_ptr,
            bias_broadcast.as_ref(),
            None,
            cur_bb,
            &gemm_op,
        )?;

        // GEMM output is [M, C_out] = [N*H*W, C] (NHWC-like)
        let exit = match conv.output_layout {
            Layout::NHWC => cur_bb,
            Layout::NCHW => {
                let n_hdr = bb("conv.trans_n.hdr");
                let save_body = bb("conv.save_hwc");
                let trans_c_entry = bb("conv.trans_c.entry");
                let trans_hw = bb("conv.trans_hw");
                let trans_c_latch = bb("conv.trans_c.latch");
                let n_latch = bb("conv.trans_n.latch");
                let exit = bb("conv.exit");
                self.builder.build_unconditional_branch(n_hdr)?;

                let (phi_n, ind_n) = self.init_counted_loop(n_hdr)?;
                let hwc = (h_out * w_out * c_out) as u64;
                let n_base_src = self.builder.build_int_mul(
                    ind_n,
                    i64_ty.const_int(hwc, false),
                    "n_base_src",
                )?;
                let n_base_dst = self.builder.build_int_mul(
                    ind_n,
                    i64_ty.const_int((c_out * h_out * w_out) as u64, false),
                    "n_base_dst",
                )?;
                self.builder.build_unconditional_branch(save_body)?;

                self.builder.position_at_end(save_body);
                let bound = h_out * w_out * c_out;
                let (phi_hwc, ind_hwc) = self.init_counted_loop(save_body)?;
                let ty =
                    ResolvedTensorType::new(dst.ty.elem_type, ResolvedTensorDims::new(&[bound]));
                let offset = self
                    .builder
                    .build_int_add(n_base_src, ind_hwc, "save_src_off")?;
                let src_ptr = TensorPtr {
                    ptr: dst.ptr,
                    offset,
                    ty: ty.clone(),
                    name: "conv_save_src".to_string(),
                };
                let dst_ws = TensorPtr {
                    ptr: workspace.ptr,
                    offset: ind_hwc,
                    ty,
                    name: "conv_save_dst".to_string(),
                };
                let load = self.build_load(&src_ptr)?;
                self.build_store(&dst_ws, load)?;
                self.finalize_counted_loop(
                    phi_hwc,
                    n_hdr,
                    i64_ty.const_int(bound as u64, false),
                    save_body,
                    trans_c_entry,
                    save_body,
                )?;

                self.builder.position_at_end(trans_c_entry);
                let (phi_c, ind_c) = self.init_counted_loop(trans_c_entry)?;
                self.builder.build_unconditional_branch(trans_hw)?;

                self.builder.position_at_end(trans_hw);
                let (phi_hw, ind_hw) = self.init_counted_loop(trans_hw)?;
                let offset = self.builder.build_int_mul(
                    ind_hw,
                    i64_ty.const_int(c_out as u64, false),
                    "ws_hw_off",
                )?;
                let offset = self.builder.build_int_add(offset, ind_c, "ws_off")?;
                let ws_src = TensorPtr {
                    ptr: workspace.ptr,
                    offset,
                    ty: ResolvedTensorType::new(
                        dst.ty.elem_type,
                        ResolvedTensorDims::new(&[bound]),
                    ),
                    name: "conv_trans_src".to_string(),
                };
                let offset = self.builder.build_int_mul(
                    ind_c,
                    i64_ty.const_int((h_out * w_out) as u64, false),
                    "dst_c_off",
                )?;
                let offset = self.builder.build_int_add(offset, ind_hw, "dst_chw")?;
                let offset = self.builder.build_int_add(n_base_dst, offset, "dst_off")?;
                let dst_nchw = TensorPtr {
                    ptr: dst.ptr,
                    offset,
                    ty: ResolvedTensorType::new(
                        dst.ty.elem_type,
                        ResolvedTensorDims::new(&[c_out * h_out * w_out]),
                    ),
                    name: "conv_trans_dst".to_string(),
                };
                let val = self.build_load(&ws_src)?;
                self.build_store(&dst_nchw, val)?;
                self.finalize_counted_loop(
                    phi_hw,
                    trans_c_entry,
                    i64_ty.const_int((h_out * w_out) as u64, false),
                    trans_hw,
                    trans_c_latch,
                    trans_hw,
                )?;

                self.builder.position_at_end(trans_c_latch);
                self.finalize_counted_loop(
                    phi_c,
                    save_body,
                    i64_ty.const_int(c_out as u64, false),
                    trans_c_entry,
                    n_latch,
                    trans_c_latch,
                )?;

                self.builder.position_at_end(n_latch);
                self.finalize_counted_loop(
                    phi_n,
                    cur_bb,
                    i64_ty.const_int(n_batch as u64, false),
                    n_hdr,
                    exit,
                    n_latch,
                )?;
                exit
            }
        };

        Ok(exit)
    }

    // depthwise conv (groups=C_in=C_out, c_out_per_group=1):
    // for n, oh, ow:
    //   for c in 0..C:
    //     val = bias[c] or 0
    //     for kh, kw:
    //       ih = oh*sh + kh*dh - pad_h; iw = ow*sw + kw*dw - pad_w
    //       if !oob: val += src[n,c,ih,iw] * weight[c,0,kh,kw]
    //     dst[...] = val   (NCHW or NHWC)
    pub fn build_depthwise_conv(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        weight: &TensorPtr<'ctx>,
        bias: Option<&TensorPtr<'ctx>>,
        conv: &operator::Conv,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(conv.group > 1);
        // TODO: support c_out_per_group > 1 (currently assumes groups == C_in == C_out)
        assert!(conv.group == weight.ty.dims[0]);
        let i64_ty = self.context.i64_type();
        let c = weight.ty.dims[0];
        let kh = conv.kernel_shape[0];
        let kw = conv.kernel_shape[1];

        let output_shape = conv.output_shape(&src.ty.dims, &weight.ty.dims);
        let n_batch = src.ty.dims[0];
        let (h_out_idx, w_out_idx) = match conv.output_layout {
            Layout::NCHW => (2, 3),
            Layout::NHWC => (1, 2),
        };
        let h_out = output_shape[h_out_idx];
        let w_out = output_shape[w_out_idx];
        let (h_in, w_in) = match conv.input_layout {
            Layout::NCHW => (src.ty.dims[2], src.ty.dims[3]),
            Layout::NHWC => (src.ty.dims[1], src.ty.dims[2]),
        };
        let stride_h = conv.strides[0] as u64;
        let stride_w = conv.strides[1] as u64;
        let dilation_h = conv.dilations[0] as u64;
        let dilation_w = conv.dilations[1] as u64;
        let (pad_h, pad_w) = match &conv.pad {
            operator::ConvPad::NotSet(pad) => (pad[0].0 as u64, pad[1].0 as u64),
            operator::ConvPad::Valid => (0, 0),
            _ => unimplemented!(),
        };

        let fp_ty = match src.ty.elem_type {
            DataType::Float(t) => t,
            _ => unimplemented!(),
        };
        let elem_ty = fp_ty.llvm_type(self.context);

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);
        let hdr_n = bb("dw.n.hdr");
        let hdr_oh = bb("dw.oh.hdr");
        let hdr_ow = bb("dw.ow.hdr");
        let hdr_c = bb("dw.c.hdr");
        let hdr_kh = bb("dw.kh.hdr");
        let hdr_kw = bb("dw.kw.hdr");
        let body = bb("dw.body");
        let bb_load = bb("dw.load");
        let bb_pad = bb("dw.pad");
        let bb_acc = bb("dw.acc");
        let latch_kw = bb("dw.kw.latch");
        let latch_kh = bb("dw.kh.latch");
        let store_bb = bb("dw.store");
        let latch_c = bb("dw.c.latch");
        let latch_ow = bb("dw.ow.latch");
        let latch_oh = bb("dw.oh.latch");
        let latch_n = bb("dw.n.latch");
        let exit = bb("dw.exit");

        let ci = |v: u64| i64_ty.const_int(v, false);

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(hdr_n)?;

        let (phi_n, ind_n) = self.init_counted_loop(hdr_n)?;
        self.builder.build_unconditional_branch(hdr_oh)?;
        let (phi_oh, ind_oh) = self.init_counted_loop(hdr_oh)?;
        self.builder.build_unconditional_branch(hdr_ow)?;
        let (phi_ow, ind_ow) = self.init_counted_loop(hdr_ow)?;
        self.builder.build_unconditional_branch(hdr_c)?;

        let (phi_c, ind_c) = self.init_counted_loop(hdr_c)?;
        let init_val = if let Some(b) = bias {
            self.build_load(&b.clone().add_offset(self.builder, ind_c)?)?
                .into_float_value()
        } else {
            elem_ty.const_float(0.0)
        };
        self.builder.build_unconditional_branch(hdr_kh)?;

        let (phi_kh, ind_kh) = self.init_counted_loop(hdr_kh)?;
        let acc_kh = self.builder.build_phi(elem_ty, "acc_kh")?;
        self.builder.build_unconditional_branch(hdr_kw)?;

        let (phi_kw, ind_kw) = self.init_counted_loop(hdr_kw)?;
        let acc = self.builder.build_phi(elem_ty, "acc")?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ih = {
            let a = self.builder.build_int_mul(ind_oh, ci(stride_h), "oh_s")?;
            let b = self.builder.build_int_mul(ind_kh, ci(dilation_h), "kh_d")?;
            let v = self.builder.build_int_add(a, b, "ih_raw")?;
            self.builder.build_int_sub(v, ci(pad_h), "ih")?
        };
        let iw = {
            let a = self.builder.build_int_mul(ind_ow, ci(stride_w), "ow_s")?;
            let b = self.builder.build_int_mul(ind_kw, ci(dilation_w), "kw_d")?;
            let v = self.builder.build_int_add(a, b, "iw_raw")?;
            self.builder.build_int_sub(v, ci(pad_w), "iw")?
        };

        let oob = {
            let h_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ih,
                i64_ty.const_zero(),
                "h_neg",
            )?;
            let h_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                ih,
                ci(h_in as u64),
                "h_big",
            )?;
            let w_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                iw,
                i64_ty.const_zero(),
                "w_neg",
            )?;
            let w_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                iw,
                ci(w_in as u64),
                "w_big",
            )?;
            let h_oob = self.builder.build_or(h_neg, h_big, "h_oob")?;
            let w_oob = self.builder.build_or(w_neg, w_big, "w_oob")?;
            self.builder.build_or(h_oob, w_oob, "oob")?
        };
        self.builder
            .build_conditional_branch(oob, bb_pad, bb_load)?;

        // load src and weight, multiply-accumulate
        self.builder.position_at_end(bb_load);
        let src_offset = match conv.input_layout {
            Layout::NCHW => {
                let o = self
                    .builder
                    .build_int_mul(ind_n, ci((c * h_in * w_in) as u64), "so_n")?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder
                        .build_int_mul(ind_c, ci((h_in * w_in) as u64), "so_c")?,
                    "so_nc",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder.build_int_mul(ih, ci(w_in as u64), "so_h")?,
                    "so_nch",
                )?;
                self.builder.build_int_add(o, iw, "src_off")?
            }
            Layout::NHWC => {
                let o = self
                    .builder
                    .build_int_mul(ind_n, ci((h_in * w_in * c) as u64), "so_n")?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder
                        .build_int_mul(ih, ci((w_in * c) as u64), "so_h")?,
                    "so_nh",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder.build_int_mul(iw, ci(c as u64), "so_w")?,
                    "so_nhw",
                )?;
                self.builder.build_int_add(o, ind_c, "src_off")?
            }
        };
        let src_val = self
            .build_load(&src.clone().add_offset(self.builder, src_offset)?)?
            .into_float_value();
        let w_offset = {
            let o = self
                .builder
                .build_int_mul(ind_c, ci((kh * kw) as u64), "wo_c")?;
            let o = self.builder.build_int_add(
                o,
                self.builder.build_int_mul(ind_kh, ci(kw as u64), "wo_kh")?,
                "wo_ckh",
            )?;
            self.builder.build_int_add(o, ind_kw, "w_off")?
        };
        let w_val = self
            .build_load(&weight.clone().add_offset(self.builder, w_offset)?)?
            .into_float_value();

        let prod = self.builder.build_float_mul(src_val, w_val, "prod")?;
        let new_acc = self.builder.build_float_add(
            acc.as_basic_value().into_float_value(),
            prod,
            "new_acc",
        )?;
        self.builder.build_unconditional_branch(bb_acc)?;

        // pad: skip (acc unchanged)
        self.builder.position_at_end(bb_pad);
        self.builder.build_unconditional_branch(bb_acc)?;

        // merge
        self.builder.position_at_end(bb_acc);
        let merged = self.builder.build_phi(elem_ty, "merged")?;
        merged.add_incoming(&[(&new_acc, bb_load), (&acc.as_basic_value(), bb_pad)]);
        self.builder.build_unconditional_branch(latch_kw)?;

        self.finalize_counted_loop(phi_kw, hdr_kh, ci(kw as u64), hdr_kw, latch_kh, latch_kw)?;
        acc.add_incoming(&[
            (&acc_kh.as_basic_value(), hdr_kh),
            (&merged.as_basic_value(), latch_kw),
        ]);

        self.finalize_counted_loop(phi_kh, hdr_c, ci(kh as u64), hdr_kh, store_bb, latch_kh)?;
        acc_kh.add_incoming(&[(&init_val, hdr_c), (&merged.as_basic_value(), latch_kh)]);

        self.builder.position_at_end(store_bb);
        let final_val = merged.as_basic_value().into_float_value();
        let dst_offset = match conv.output_layout {
            Layout::NCHW => {
                let o =
                    self.builder
                        .build_int_mul(ind_n, ci((c * h_out * w_out) as u64), "do_n")?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder
                        .build_int_mul(ind_c, ci((h_out * w_out) as u64), "do_c")?,
                    "do_nc",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder
                        .build_int_mul(ind_oh, ci(w_out as u64), "do_h")?,
                    "do_nch",
                )?;
                self.builder.build_int_add(o, ind_ow, "dst_off")?
            }
            Layout::NHWC => {
                let o =
                    self.builder
                        .build_int_mul(ind_n, ci((h_out * w_out * c) as u64), "do_n")?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder
                        .build_int_mul(ind_oh, ci((w_out * c) as u64), "do_h")?,
                    "do_nh",
                )?;
                let o = self.builder.build_int_add(
                    o,
                    self.builder.build_int_mul(ind_ow, ci(c as u64), "do_w")?,
                    "do_nhw",
                )?;
                self.builder.build_int_add(o, ind_c, "dst_off")?
            }
        };
        self.build_store(&dst.clone().set_offset(dst_offset), final_val)?;
        self.builder.build_unconditional_branch(latch_c)?;

        self.finalize_counted_loop(phi_c, hdr_ow, ci(c as u64), hdr_c, latch_ow, latch_c)?;
        self.finalize_counted_loop(phi_ow, hdr_oh, ci(w_out as u64), hdr_ow, latch_oh, latch_ow)?;
        self.finalize_counted_loop(phi_oh, hdr_n, ci(h_out as u64), hdr_oh, latch_n, latch_oh)?;
        self.finalize_counted_loop(phi_n, entry, ci(n_batch as u64), hdr_n, exit, latch_n)?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_pool_nchw(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        pooling: &operator::Pooling,
        mode: PoolMode,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert_eq!(pooling.kernel_shape.ndim(), 2);

        let i64_ty = self.context.i64_type();
        let nbatch = src.ty.dims[0] as u64;
        let c_in = src.ty.dims[1] as u64;
        let h_in = src.ty.dims[2] as u64;
        let w_in = src.ty.dims[3] as u64;
        let h_out = dst.ty.dims[2] as u64;
        let w_out = dst.ty.dims[3] as u64;
        let kh = pooling.kernel_shape[0] as u64;
        let kw = pooling.kernel_shape[1] as u64;
        let stride_h = pooling.strides[0] as u64;
        let stride_w = pooling.strides[1] as u64;
        let dilation_h = pooling.dilations[0] as u64;
        let dilation_w = pooling.dilations[1] as u64;
        let calc_pad = |dim: usize| -> u64 {
            match &pooling.pad {
                operator::ConvPad::NotSet(pad) => pad[dim].0 as u64,
                operator::ConvPad::Valid => 0,
                operator::ConvPad::SameLower | operator::ConvPad::SameUpper => unimplemented!(),
            }
        };
        let pad_h = calc_pad(0);
        let pad_w = calc_pad(1);

        let fp_ty = match src.ty.elem_type {
            DataType::Float(t) => t,
            _ => unimplemented!(),
        };
        let elem_ty = fp_ty.llvm_type(self.context);
        let init_val = match mode {
            PoolMode::Avg => elem_ty.const_float(0.0).as_basic_value_enum(),
            PoolMode::Max => elem_ty
                .const_float(match fp_ty {
                    FloatType::F32 => f32::NEG_INFINITY as f64,
                    FloatType::F64 => f64::NEG_INFINITY,
                    FloatType::BF16 => f32::NEG_INFINITY as f64,
                })
                .as_basic_value_enum(),
        };

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);
        let hdr_n = bb("pool.n.hdr");
        let hdr_c = bb("pool.c.hdr");
        let hdr_oh = bb("pool.oh.hdr");
        let hdr_ow = bb("pool.ow.hdr");
        let hdr_kh = bb("pool.kh.hdr");
        let hdr_kw = bb("pool.kw.hdr");
        let body = bb("pool.body");
        let bb_update = bb("pool.update");
        let latch_kw = bb("pool.kw.latch");
        let latch_kh = bb("pool.kh.latch");
        let store_bb = bb("pool.store");
        let latch_ow = bb("pool.ow.latch");
        let latch_oh = bb("pool.oh.latch");
        let latch_c = bb("pool.c.latch");
        let latch_n = bb("pool.n.latch");
        let exit = bb("pool.exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(hdr_n)?;

        let (phi_n, ind_n) = self.init_counted_loop(hdr_n)?;
        self.builder.build_unconditional_branch(hdr_c)?;

        let (phi_c, ind_c) = self.init_counted_loop(hdr_c)?;
        self.builder.build_unconditional_branch(hdr_oh)?;

        let (phi_oh, ind_oh) = self.init_counted_loop(hdr_oh)?;
        self.builder.build_unconditional_branch(hdr_ow)?;

        let (phi_ow, ind_ow) = self.init_counted_loop(hdr_ow)?;
        self.builder.build_unconditional_branch(hdr_kh)?;

        self.builder.position_at_end(hdr_kh);
        let phi_kh = self.builder.build_phi(i64_ty, "ind")?;
        let ind_kh = phi_kh.as_basic_value().into_int_value();
        let acc_kh = self.builder.build_phi(elem_ty, "acc")?;
        let cnt_acc_kh = self.builder.build_phi(i64_ty, "cnt_acc")?;
        self.builder.build_unconditional_branch(hdr_kw)?;

        self.builder.position_at_end(hdr_kw);
        let phi_kw = self.builder.build_phi(i64_ty, "ind")?;
        let ind_kw = phi_kw.as_basic_value().into_int_value();
        let acc = self.builder.build_phi(elem_ty, "acc")?;
        let cnt_acc = self.builder.build_phi(i64_ty, "cnt_acc")?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ih = {
            let a =
                self.builder
                    .build_int_mul(ind_oh, i64_ty.const_int(stride_h, false), "oh_s")?;
            let b =
                self.builder
                    .build_int_mul(ind_kh, i64_ty.const_int(dilation_h, false), "kh_d")?;
            let c = self.builder.build_int_add(a, b, "ih_raw")?;
            self.builder
                .build_int_sub(c, i64_ty.const_int(pad_h, false), "ih")?
        };
        let iw = {
            let a =
                self.builder
                    .build_int_mul(ind_ow, i64_ty.const_int(stride_w, false), "ow_s")?;
            let b =
                self.builder
                    .build_int_mul(ind_kw, i64_ty.const_int(dilation_w, false), "kw_d")?;
            let c = self.builder.build_int_add(a, b, "iw_raw")?;
            self.builder
                .build_int_sub(c, i64_ty.const_int(pad_w, false), "iw")?
        };

        let oob = {
            let ih_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ih,
                i64_ty.const_zero(),
                "ih_neg",
            )?;
            let ih_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                ih,
                i64_ty.const_int(h_in, false),
                "ih_big",
            )?;
            let iw_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                iw,
                i64_ty.const_zero(),
                "iw_neg",
            )?;
            let iw_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                iw,
                i64_ty.const_int(w_in, false),
                "iw_big",
            )?;
            let a = self.builder.build_or(ih_neg, ih_big, "oob_h")?;
            let b = self.builder.build_or(iw_neg, iw_big, "oob_w")?;
            self.builder.build_or(a, b, "oob")?
        };
        self.builder
            .build_conditional_branch(oob, latch_kw, bb_update)?;

        self.builder.position_at_end(bb_update);
        let src_offset = {
            let o = self.builder.build_int_mul(
                ind_n,
                i64_ty.const_int(c_in * h_in * w_in, false),
                "so_n",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder
                    .build_int_mul(ind_c, i64_ty.const_int(h_in * w_in, false), "so_c")?,
                "so_nc",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder
                    .build_int_mul(ih, i64_ty.const_int(w_in, false), "so_h")?,
                "so_nch",
            )?;
            self.builder.build_int_add(o, iw, "src_off")?
        };
        let src_val = self
            .build_load(&src.clone().set_offset(src_offset))?
            .into_float_value();
        let cur_acc = acc.as_basic_value().into_float_value();
        let new_acc = match mode {
            PoolMode::Avg => self
                .builder
                .build_float_add(cur_acc, src_val, "new_acc")?
                .as_basic_value_enum(),
            PoolMode::Max => {
                let fmax = self.intrinsics.fmax.get(fp_ty);
                self.build_tail_call(fmax, &[cur_acc.into(), src_val.into()], "new_acc")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }
        };
        let cur_cnt = cnt_acc.as_basic_value().into_int_value();
        let new_cnt = self
            .builder
            .build_int_add(cur_cnt, i64_ty.const_int(1, false), "new_cnt")?;
        self.builder.build_unconditional_branch(latch_kw)?;

        self.builder.position_at_end(latch_kw);
        let merged_acc = self.builder.build_phi(elem_ty, "merged_acc")?;
        merged_acc.add_incoming(&[(&new_acc, bb_update), (&cur_acc, body)]);
        let merged_cnt = self.builder.build_phi(i64_ty, "merged_cnt")?;
        merged_cnt.add_incoming(&[(&new_cnt, bb_update), (&cur_cnt, body)]);
        let c = |v: u64| i64_ty.const_int(v, false);
        let ind_kw_next = self.builder.build_int_add(ind_kw, c(1), "next")?;
        let kw_done =
            self.builder
                .build_int_compare(inkwell::IntPredicate::EQ, ind_kw_next, c(kw), "ec")?;
        self.builder
            .build_conditional_branch(kw_done, latch_kh, hdr_kw)?;
        phi_kw.add_incoming(&[(&i64_ty.const_zero(), hdr_kh), (&ind_kw_next, latch_kw)]);
        acc.add_incoming(&[
            (&acc_kh.as_basic_value(), hdr_kh),
            (&merged_acc.as_basic_value(), latch_kw),
        ]);
        cnt_acc.add_incoming(&[
            (&cnt_acc_kh.as_basic_value(), hdr_kh),
            (&merged_cnt.as_basic_value(), latch_kw),
        ]);

        self.builder.position_at_end(latch_kh);
        let ind_kh_next = self.builder.build_int_add(ind_kh, c(1), "next")?;
        let kh_done =
            self.builder
                .build_int_compare(inkwell::IntPredicate::EQ, ind_kh_next, c(kh), "ec")?;
        self.builder
            .build_conditional_branch(kh_done, store_bb, hdr_kh)?;
        phi_kh.add_incoming(&[(&i64_ty.const_zero(), hdr_ow), (&ind_kh_next, latch_kh)]);
        acc_kh.add_incoming(&[
            (&init_val, hdr_ow),
            (&merged_acc.as_basic_value(), latch_kh),
        ]);
        cnt_acc_kh.add_incoming(&[
            (&i64_ty.const_zero(), hdr_ow),
            (&merged_cnt.as_basic_value(), latch_kh),
        ]);

        self.builder.position_at_end(store_bb);
        let final_val = match mode {
            PoolMode::Avg => {
                let final_acc = merged_acc.as_basic_value().into_float_value();
                let final_cnt = merged_cnt.as_basic_value().into_int_value();
                let cnt_fp = self
                    .builder
                    .build_signed_int_to_float(final_cnt, elem_ty, "cnt_fp")?;
                self.builder
                    .build_float_div(final_acc, cnt_fp, "avg")?
                    .as_basic_value_enum()
            }
            PoolMode::Max => merged_acc.as_basic_value(),
        };
        let dst_offset = {
            let o = self.builder.build_int_mul(
                ind_n,
                i64_ty.const_int(c_in * h_out * w_out, false),
                "do_n",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder.build_int_mul(
                    ind_c,
                    i64_ty.const_int(h_out * w_out, false),
                    "do_c",
                )?,
                "do_nc",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder
                    .build_int_mul(ind_oh, i64_ty.const_int(w_out, false), "do_oh")?,
                "do_ncoh",
            )?;
            self.builder.build_int_add(o, ind_ow, "dst_off")?
        };
        self.build_store(&dst.clone().set_offset(dst_offset), final_val)?;
        self.builder.build_unconditional_branch(latch_ow)?;

        self.finalize_counted_loop(phi_ow, hdr_oh, c(w_out), hdr_ow, latch_oh, latch_ow)?;
        self.finalize_counted_loop(phi_oh, hdr_c, c(h_out), hdr_oh, latch_c, latch_oh)?;
        self.finalize_counted_loop(phi_c, hdr_n, c(c_in), hdr_c, latch_n, latch_c)?;
        self.finalize_counted_loop(phi_n, entry, c(nbatch), hdr_n, exit, latch_n)?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_pool_nhwc(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        pooling: &operator::Pooling,
        mode: PoolMode,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert_eq!(pooling.kernel_shape.ndim(), 2);

        let i64_ty = self.context.i64_type();
        let ndim = src.ty.dims.ndim();
        let nbatch = src.ty.dims[0] as u64;
        let h_in = src.ty.dims[1] as u64;
        let w_in = src.ty.dims[2] as u64;
        let c_in = src.ty.dims[ndim - 1] as u64;
        let h_out = dst.ty.dims[1] as u64;
        let w_out = dst.ty.dims[2] as u64;
        let kh = pooling.kernel_shape[0] as u64;
        let kw = pooling.kernel_shape[1] as u64;
        let stride_h = pooling.strides[0] as u64;
        let stride_w = pooling.strides[1] as u64;
        let dilation_h = pooling.dilations[0] as u64;
        let dilation_w = pooling.dilations[1] as u64;
        let calc_pad = |dim: usize| -> u64 {
            match &pooling.pad {
                operator::ConvPad::NotSet(pad) => pad[dim].0 as u64,
                operator::ConvPad::Valid => 0,
                operator::ConvPad::SameLower | operator::ConvPad::SameUpper => unimplemented!(),
            }
        };
        let pad_h = calc_pad(0);
        let pad_w = calc_pad(1);

        let fp_ty = match src.ty.elem_type {
            DataType::Float(t) => t,
            _ => unimplemented!(),
        };
        let elem_ty = fp_ty.llvm_type(self.context);
        let init_float = match mode {
            PoolMode::Avg => 0.0,
            PoolMode::Max => match fp_ty {
                FloatType::F32 => f32::NEG_INFINITY as f64,
                FloatType::F64 => f64::NEG_INFINITY,
                FloatType::BF16 => f32::NEG_INFINITY as f64,
            },
        };

        let acc_vals_ptr =
            self.builder
                .build_array_alloca(elem_ty, i64_ty.const_int(c_in, false), "acc_vals")?;
        let cnt_ptr = self
            .builder
            .build_array_alloca(i64_ty, i64_ty.const_int(1, false), "cnt")?;

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);
        let hdr_n = bb("pool.n.hdr");
        let hdr_oh = bb("pool.oh.hdr");
        let hdr_ow = bb("pool.ow.hdr");
        let init_loop = bb("pool.init");
        let hdr_kh = bb("pool.kh.hdr");
        let hdr_kw = bb("pool.kw.hdr");
        let body = bb("pool.body");
        let bb_update = bb("pool.update");
        let update_loop = bb("pool.update_c");
        let latch_kw = bb("pool.kw.latch");
        let latch_kh = bb("pool.kh.latch");
        let store_loop = bb("pool.store_c");
        let latch_ow = bb("pool.ow.latch");
        let latch_oh = bb("pool.oh.latch");
        let latch_n = bb("pool.n.latch");
        let exit = bb("pool.exit");

        let c = |v: u64| i64_ty.const_int(v, false);

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(hdr_n)?;

        let (phi_n, ind_n) = self.init_counted_loop(hdr_n)?;
        self.builder.build_unconditional_branch(hdr_oh)?;

        let (phi_oh, ind_oh) = self.init_counted_loop(hdr_oh)?;
        self.builder.build_unconditional_branch(hdr_ow)?;

        let (phi_ow, ind_ow) = self.init_counted_loop(hdr_ow)?;

        self.builder.build_unconditional_branch(init_loop)?;
        let (phi_init, ind_init) = self.init_counted_loop(init_loop)?;
        let gep = unsafe {
            self.builder
                .build_in_bounds_gep(elem_ty, acc_vals_ptr, &[ind_init], "acc_vals_gep")?
        };
        self.builder
            .build_store(gep, elem_ty.const_float(init_float))?;
        let init_done = bb("pool.init_done");
        self.finalize_counted_loop(phi_init, hdr_ow, c(c_in), init_loop, init_done, init_loop)?;

        self.builder.position_at_end(init_done);
        self.builder.build_store(cnt_ptr, i64_ty.const_zero())?;
        self.builder.build_unconditional_branch(hdr_kh)?;

        let (phi_kh, ind_kh) = self.init_counted_loop(hdr_kh)?;
        self.builder.build_unconditional_branch(hdr_kw)?;

        let (phi_kw, ind_kw) = self.init_counted_loop(hdr_kw)?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ih = {
            let a = self.builder.build_int_mul(ind_oh, c(stride_h), "oh_s")?;
            let b = self.builder.build_int_mul(ind_kh, c(dilation_h), "kh_d")?;
            let v = self.builder.build_int_add(a, b, "ih_raw")?;
            self.builder.build_int_sub(v, c(pad_h), "ih")?
        };
        let iw = {
            let a = self.builder.build_int_mul(ind_ow, c(stride_w), "ow_s")?;
            let b = self.builder.build_int_mul(ind_kw, c(dilation_w), "kw_d")?;
            let v = self.builder.build_int_add(a, b, "iw_raw")?;
            self.builder.build_int_sub(v, c(pad_w), "iw")?
        };
        let oob = {
            let ih_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                ih,
                i64_ty.const_zero(),
                "ih_neg",
            )?;
            let ih_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                ih,
                c(h_in),
                "ih_big",
            )?;
            let iw_neg = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                iw,
                i64_ty.const_zero(),
                "iw_neg",
            )?;
            let iw_big = self.builder.build_int_compare(
                inkwell::IntPredicate::SGE,
                iw,
                c(w_in),
                "iw_big",
            )?;
            let a = self.builder.build_or(ih_neg, ih_big, "oob_h")?;
            let b = self.builder.build_or(iw_neg, iw_big, "oob_w")?;
            self.builder.build_or(a, b, "oob")?
        };
        self.builder
            .build_conditional_branch(oob, latch_kw, bb_update)?;

        self.builder.position_at_end(bb_update);
        let cur_cnt = self
            .builder
            .build_load(i64_ty, cnt_ptr, "cur_cnt")?
            .into_int_value();
        let new_cnt = self.builder.build_int_add(cur_cnt, c(1), "new_cnt")?;
        self.builder.build_store(cnt_ptr, new_cnt)?;
        self.builder.build_unconditional_branch(update_loop)?;

        let (phi_uc, ind_uc) = self.init_counted_loop(update_loop)?;
        let src_offset = {
            let o = self
                .builder
                .build_int_mul(ind_n, c(h_in * w_in * c_in), "so_n")?;
            let o = self.builder.build_int_add(
                o,
                self.builder.build_int_mul(ih, c(w_in * c_in), "so_h")?,
                "so_nh",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder.build_int_mul(iw, c(c_in), "so_w")?,
                "so_nhw",
            )?;
            self.builder.build_int_add(o, ind_uc, "src_off")?
        };
        let src_val = self
            .build_load(&src.clone().set_offset(src_offset))?
            .into_float_value();
        let gep = unsafe {
            self.builder
                .build_in_bounds_gep(elem_ty, acc_vals_ptr, &[ind_uc], "acc_gep")?
        };
        let cur_acc = self
            .builder
            .build_load(elem_ty, gep, "cur_acc")?
            .into_float_value();
        let new_acc = match mode {
            PoolMode::Avg => self
                .builder
                .build_float_add(cur_acc, src_val, "new_acc")?
                .as_basic_value_enum(),
            PoolMode::Max => {
                let fmax = self.intrinsics.fmax.get(fp_ty);
                self.build_tail_call(fmax, &[cur_acc.into(), src_val.into()], "new_acc")?
                    .try_as_basic_value()
                    .left()
                    .unwrap()
            }
        };
        self.builder.build_store(gep, new_acc)?;
        self.finalize_counted_loop(
            phi_uc,
            bb_update,
            c(c_in),
            update_loop,
            latch_kw,
            update_loop,
        )?;

        self.finalize_counted_loop(phi_kw, hdr_kh, c(kw), hdr_kw, latch_kh, latch_kw)?;
        self.finalize_counted_loop(phi_kh, init_done, c(kh), hdr_kh, store_loop, latch_kh)?;

        self.builder.position_at_end(store_loop);
        let (phi_sc, ind_sc) = self.init_counted_loop(store_loop)?;
        let gep = unsafe {
            self.builder
                .build_in_bounds_gep(elem_ty, acc_vals_ptr, &[ind_sc], "store_gep")?
        };
        let val = self
            .builder
            .build_load(elem_ty, gep, "val")?
            .into_float_value();
        let final_val = match mode {
            PoolMode::Avg => {
                let final_cnt = self
                    .builder
                    .build_load(i64_ty, cnt_ptr, "final_cnt")?
                    .into_int_value();
                let cnt_fp = self
                    .builder
                    .build_signed_int_to_float(final_cnt, elem_ty, "cnt_fp")?;
                self.builder
                    .build_float_div(val, cnt_fp, "avg")?
                    .as_basic_value_enum()
            }
            PoolMode::Max => val.as_basic_value_enum(),
        };
        let dst_offset = {
            let o = self
                .builder
                .build_int_mul(ind_n, c(h_out * w_out * c_in), "do_n")?;
            let o = self.builder.build_int_add(
                o,
                self.builder
                    .build_int_mul(ind_oh, c(w_out * c_in), "do_oh")?,
                "do_noh",
            )?;
            let o = self.builder.build_int_add(
                o,
                self.builder.build_int_mul(ind_ow, c(c_in), "do_ow")?,
                "do_nohow",
            )?;
            self.builder.build_int_add(o, ind_sc, "dst_off")?
        };
        self.build_store(&dst.clone().set_offset(dst_offset), final_val)?;
        self.finalize_counted_loop(phi_sc, latch_kh, c(c_in), store_loop, latch_ow, store_loop)?;

        self.finalize_counted_loop(phi_ow, hdr_oh, c(w_out), hdr_ow, latch_oh, latch_ow)?;
        self.finalize_counted_loop(phi_oh, hdr_n, c(h_out), hdr_oh, latch_n, latch_oh)?;
        self.finalize_counted_loop(phi_n, entry, c(nbatch), hdr_n, exit, latch_n)?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_gemm(
        &self,
        dst: &TensorPtr<'ctx>,
        a: &TensorPtr<'ctx>,
        b: &TensorPtr<'ctx>,
        c: Option<&TensorPtr<'ctx>>,
        workspace: Option<&TensorPtr<'ctx>>,
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
        self.builder.position_at_end(entry);
        let a_gep = self.build_gep(a)?;
        let b_gep = self.build_gep(b)?;
        let c_gep = self.build_gep(dst)?;

        let fp_ty = dst.ty.elem_type.float_type().unwrap();
        if fp_ty == FloatType::BF16 {
            let workspace = workspace.expect("bf16 Gemm requires GEMM_WORKSPACE input");
            let ws_a = self.build_gep(workspace)?;
            let ws_b = self.workspace_offset(workspace, m * k, "gemm.ws_b")?;
            let ws_c = self.workspace_offset(workspace, m * k + k * n, "gemm.ws_c")?;
            self.bf16_buf_to_f32(a_gep, ws_a, m * k)?;
            self.bf16_buf_to_f32(b_gep, ws_b, k * n)?;
            if beta != 0.0 {
                self.bf16_buf_to_f32(c_gep, ws_c, m * n)?;
            }
            let gemm_args = GemmArgs {
                a: (ws_a, trans_a),
                b: (ws_b, trans_b),
                c: ws_c,
                alpha: gemm.alpha,
                beta,
                m,
                n,
                k,
            };
            self.blas
                .call_gemm(FloatType::F32, &gemm_args, self.builder)?;
            self.f32_buf_to_bf16(ws_c, c_gep, m * n)?;
            return Ok(self.builder.get_insert_block().unwrap());
        }
        let gemm_args = GemmArgs {
            a: (a_gep, trans_a),
            b: (b_gep, trans_b),
            c: c_gep,
            alpha: gemm.alpha,
            beta,
            m,
            n,
            k,
        };
        self.blas.call_gemm(fp_ty, &gemm_args, self.builder)?;
        Ok(entry)
    }

    pub fn build_batched_gemm(
        &self,
        dst: &TensorPtr<'ctx>,
        a: &TensorPtr<'ctx>,
        b: &TensorPtr<'ctx>,
        workspace: Option<&TensorPtr<'ctx>>,
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

        if fp_ty == FloatType::BF16 {
            let total_a = batch_count as u64 * stride_a;
            let total_b = batch_count as u64 * stride_b;
            let total_c = batch_count as u64 * stride_c;
            let workspace =
                workspace.expect("bf16 BatchedGemm requires BATCHED_GEMM_WORKSPACE input");
            let ws_a = self.build_gep(workspace)?;
            let ws_b = self.workspace_offset(workspace, total_a, "bgemm.ws_b")?;
            let ws_c = self.workspace_offset(workspace, total_a + total_b, "bgemm.ws_c")?;
            self.bf16_buf_to_f32(a_ptr, ws_a, total_a)?;
            self.bf16_buf_to_f32(b_ptr, ws_b, total_b)?;
            if gemm.beta != 0.0 {
                self.bf16_buf_to_f32(dst_ptr, ws_c, total_c)?;
            }

            let body = self
                .context
                .append_basic_block(*self.func, "bgemm.bf16.body");
            let after = self
                .context
                .append_basic_block(*self.func, "bgemm.bf16.after");
            let guard = self.builder.build_int_compare(
                inkwell::IntPredicate::EQ,
                i64_ty.const_int(batch_count as u64, false),
                i64_ty.const_zero(),
                "bgemm.bf16.guard",
            )?;
            self.builder.build_conditional_branch(guard, after, body)?;
            let (ind, idx) = self.init_counted_loop(body)?;
            let f32_ty = self.context.f32_type();
            let a_slice = unsafe {
                self.builder.build_in_bounds_gep(
                    f32_ty,
                    ws_a,
                    &[self.builder.build_int_mul(
                        idx,
                        i64_ty.const_int(stride_a, false),
                        "a.off",
                    )?],
                    "a.slice",
                )?
            };
            let b_slice = unsafe {
                self.builder.build_in_bounds_gep(
                    f32_ty,
                    ws_b,
                    &[self.builder.build_int_mul(
                        idx,
                        i64_ty.const_int(stride_b, false),
                        "b.off",
                    )?],
                    "b.slice",
                )?
            };
            let c_slice = unsafe {
                self.builder.build_in_bounds_gep(
                    f32_ty,
                    ws_c,
                    &[self.builder.build_int_mul(
                        idx,
                        i64_ty.const_int(stride_c, false),
                        "c.off",
                    )?],
                    "c.slice",
                )?
            };
            self.blas.call_gemm(
                FloatType::F32,
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
            self.finalize_counted_loop(
                ind,
                entry,
                i64_ty.const_int(batch_count as u64, false),
                body,
                after,
                body,
            )?;
            self.builder.position_at_end(after);
            self.f32_buf_to_bf16(ws_c, dst_ptr, total_c)?;
            return Ok(self.builder.get_insert_block().unwrap());
        }

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
            let (ind, idx) = self.init_counted_loop(body)?;

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

            self.finalize_counted_loop(
                ind,
                entry,
                i64_ty.const_int(batch_count as u64, false),
                body,
                exit,
                body,
            )?;

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

        let (ind, ind_val) = self.init_counted_loop(header)?;
        let a = {
            let offset = self.builder.build_int_mul(
                ind_val,
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
                ind_val,
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
                ind_val,
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
                ind_val,
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
        self.build_gemm(&dst, &a, &b, c.as_ref(), None, header, &gemm)?;
        self.builder.build_unconditional_branch(latch)?;

        self.finalize_counted_loop(
            ind,
            entry,
            self.context.i64_type().const_int(bound as u64, false),
            header,
            exit,
            latch,
        )?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    // out[..., m, n] = scale[n] * sum_k (f32)act[..., m, k] * (f32)wq[n, k]
    //
    //   act:       [..., M, K] float
    //   wq:        [N, K]      i8
    //   scale:     [N]         same float type as out
    //   out:       [..., M, N] float
    //   workspace: [M*K + N*K + M*N] f32
    //
    // Pre-dequantize wq into workspace[M*K..M*K+N*K] as f32, optionally promote
    // act to f32 (workspace[..M*K]), then dispatch BLAS sgemm and (if needed)
    // round the f32 output back to bf16. Layout matches build_gemm's bf16 path.
    pub fn build_dequant_matmul(
        &self,
        out: &TensorPtr<'ctx>,
        act: &TensorPtr<'ctx>,
        wq: &TensorPtr<'ctx>,
        scale: &TensorPtr<'ctx>,
        workspace: &TensorPtr<'ctx>,
        axis: usize,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert_eq!(axis, 0, "DequantMatMul kernel assumes axis=0");
        assert!(act.ty.is_contiguous());
        assert!(wq.ty.is_contiguous());
        assert!(out.ty.is_contiguous());
        assert_eq!(wq.ty.dims.ndim(), 2);
        let n_dim = wq.ty.dims[0];
        let k = wq.ty.dims[1];
        assert_eq!(act.ty.dims.last().copied().unwrap(), k);
        let m_total = act.ty.dims.size() / k;
        assert_eq!(out.ty.dims.size(), m_total * n_dim);

        let DataType::Float(act_float_ty) = act.ty.elem_type else {
            panic!("DequantMatMul activation must be float");
        };
        let DataType::Float(out_float_ty) = out.ty.elem_type else {
            panic!("DequantMatMul output must be float");
        };
        assert!(matches!(wq.ty.elem_type, DataType::SInt(SIntType::I8)));
        assert_eq!(scale.ty.elem_type, out.ty.elem_type);
        assert!(matches!(act_float_ty, FloatType::F32 | FloatType::BF16));
        assert!(matches!(out_float_ty, FloatType::F32 | FloatType::BF16));
        assert_eq!(
            workspace.ty.elem_type,
            DataType::Float(FloatType::F32),
            "DequantMatMul workspace must be f32",
        );

        let m = m_total as u64;
        let n = n_dim as u64;
        let k_u = k as u64;
        let i64_ty = self.context.i64_type();
        let f32_ty = self.context.f32_type();
        let i8_ty = self.context.i8_type();
        let i32_ty = self.context.i32_type();

        self.builder.position_at_end(entry);
        let act_gep = self.build_gep(act)?;
        let wq_gep = self.build_gep(wq)?;
        let out_gep = self.build_gep(out)?;
        let ws_a = self.workspace_offset(workspace, 0, "dqmm.ws_a")?;
        let ws_b = self.workspace_offset(workspace, m * k_u, "dqmm.ws_b")?;
        let ws_c = self.workspace_offset(workspace, m * k_u + n * k_u, "dqmm.ws_c")?;

        // ws_a <- act (no-op when act is already f32; just point at act_gep)
        let a_for_gemm = match act_float_ty {
            FloatType::BF16 => {
                self.bf16_buf_to_f32(act_gep, ws_a, m * k_u)?;
                ws_a
            }
            FloatType::F32 => act_gep,
            FloatType::F64 => unreachable!(),
        };

        // ws_b[n*k + k_idx] = scale[n] * (f32)wq[n*k + k_idx]
        {
            let preheader = self.builder.get_insert_block().unwrap();
            let header = self.context.append_basic_block(*self.func, "dqmm.dq.h");
            let after = self.context.append_basic_block(*self.func, "dqmm.dq.x");
            self.builder.build_unconditional_branch(header)?;
            let (phi, idx) = self.init_counted_loop(header)?;

            let n_idx = if n == 1 {
                i64_ty.const_zero()
            } else {
                self.builder.build_int_unsigned_div(
                    idx,
                    i64_ty.const_int(k_u, false),
                    "dqmm.dq.n",
                )?
            };
            let scale_off = self
                .builder
                .build_int_add(scale.offset, n_idx, "dqmm.dq.scale.off")?;
            let scale_v = self
                .build_load(&scale.clone().set_offset(scale_off))?
                .into_float_value();
            let scale_f32 = match out_float_ty {
                FloatType::F32 | FloatType::BF16 => scale_v,
                FloatType::F64 => unreachable!(),
            };

            let wq_idx_gep = unsafe {
                self.builder
                    .build_in_bounds_gep(i8_ty, wq_gep, &[idx], "dqmm.dq.wq.gep")?
            };
            let wq_i8 = self
                .builder
                .build_load(i8_ty, wq_idx_gep, "dqmm.dq.wq.i8")?
                .into_int_value();
            let wq_i32 = self
                .builder
                .build_int_s_extend(wq_i8, i32_ty, "dqmm.dq.wq.sext")?;
            let wq_f32 =
                self.builder
                    .build_signed_int_to_float(wq_i32, f32_ty, "dqmm.dq.wq.f32")?;
            let prod = self
                .builder
                .build_float_mul(scale_f32, wq_f32, "dqmm.dq.prod")?;
            let dst_gep = unsafe {
                self.builder
                    .build_in_bounds_gep(f32_ty, ws_b, &[idx], "dqmm.dq.dst.gep")?
            };
            self.builder.build_store(dst_gep, prod)?;

            self.finalize_counted_loop(
                phi,
                preheader,
                i64_ty.const_int(n * k_u, false),
                header,
                after,
                header,
            )?;
            self.builder.position_at_end(after);
        }

        // BLAS sgemm: C = A @ B^T  (B is [N, K], so trans_b=true)
        let c_for_gemm = match out_float_ty {
            FloatType::BF16 => ws_c,
            FloatType::F32 => out_gep,
            FloatType::F64 => unreachable!(),
        };
        let gemm_args = GemmArgs {
            a: (a_for_gemm, false),
            b: (ws_b, true),
            c: c_for_gemm,
            alpha: 1.0,
            beta: 0.0,
            m,
            n,
            k: k_u,
        };
        self.blas
            .call_gemm(FloatType::F32, &gemm_args, self.builder)?;

        if matches!(out_float_ty, FloatType::BF16) {
            self.f32_buf_to_bf16(ws_c, out_gep, m * n)?;
        }

        Ok(self.builder.get_insert_block().unwrap())
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
                    FloatType::F32 | FloatType::BF16 => f32::MIN as f64,
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

    // Quantize new[B, H, NEW_SEQ, D] into the i8 cache slot starting at `offset`,
    // recording the per-token scale alongside.
    //
    //   for each (b, h, t):
    //       max_abs = max_d |new[b,h,t,d]|
    //       s       = max_abs / 127
    //       cache[b, h, offset+t, d] = round(new[..] / s) clamped to [-128, 127]
    //       scale[b, h, offset+t]    = s
    //
    // round() is round-half-away-from-zero (matches CUDA roundf). Implemented via
    // a 0.5 sign-bias plus truncation-toward-zero, since we already need to clamp
    // into i8 afterwards.
    pub fn build_quantizing_kv_cache_update(
        &self,
        cache: &TensorPtr<'ctx>,
        scale: &TensorPtr<'ctx>,
        new_kv: &TensorPtr<'ctx>,
        offset_ptr: &TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert_eq!(cache.ty.dims.ndim(), 4);
        assert_eq!(new_kv.ty.dims.ndim(), 4);
        assert_eq!(scale.ty.dims.ndim(), 3);
        assert!(matches!(cache.ty.elem_type, DataType::SInt(SIntType::I8)));
        let DataType::Float(scale_float_ty) = scale.ty.elem_type else {
            panic!("QuantizingKVCacheUpdate scale must be float");
        };
        let DataType::Float(new_float_ty) = new_kv.ty.elem_type else {
            panic!("QuantizingKVCacheUpdate new must be float");
        };
        assert_eq!(scale_float_ty, new_float_ty);
        assert!(matches!(scale_float_ty, FloatType::F32 | FloatType::BF16));
        let b_dim = cache.ty.dims[0];
        let h_dim = cache.ty.dims[1];
        let max_seq = cache.ty.dims[2];
        let head_dim = cache.ty.dims[3];
        assert_eq!(new_kv.ty.dims[0], b_dim);
        assert_eq!(new_kv.ty.dims[1], h_dim);
        let new_seq = new_kv.ty.dims[2];
        assert_eq!(new_kv.ty.dims[3], head_dim);
        assert_eq!(scale.ty.dims[0], b_dim);
        assert_eq!(scale.ty.dims[1], h_dim);
        assert_eq!(scale.ty.dims[2], max_seq);
        let total_outer = b_dim * h_dim * new_seq;

        let i64_ty = self.context.i64_type();
        let i32_ty = self.context.i32_type();
        let i8_ty = self.context.i8_type();
        let f32_ty = self.context.f32_type();
        let fmax_fn = self.intrinsics.fmax.get(FloatType::F32);

        self.builder.position_at_end(entry);
        let offset_gep = unsafe {
            self.builder.build_in_bounds_gep(
                offset_ptr.ty.elem_type.llvm_type(self.context),
                offset_ptr.ptr,
                &[offset_ptr.offset],
                "qkvcu.offset.gep",
            )?
        };
        let offset_val = self
            .builder
            .build_load(i64_ty, offset_gep, "qkvcu.offset")?
            .into_int_value();

        let outer_header = self.context.append_basic_block(*self.func, "qkvcu.outer.h");
        let after_max_loop = self
            .context
            .append_basic_block(*self.func, "qkvcu.maxloop.x");
        let after_quant_loop = self
            .context
            .append_basic_block(*self.func, "qkvcu.quantloop.x");
        let outer_latch = self
            .context
            .append_basic_block(*self.func, "qkvcu.outer.latch");
        let exit = self.context.append_basic_block(*self.func, "qkvcu.exit");

        self.builder.build_unconditional_branch(outer_header)?;
        let (outer_phi, outer_idx) = self.init_counted_loop(outer_header)?;
        self.builder.position_at_end(outer_header);

        // bh = outer / new_seq, t = outer % new_seq
        let new_seq_const = i64_ty.const_int(new_seq as u64, false);
        let max_seq_const = i64_ty.const_int(max_seq as u64, false);
        let head_dim_const = i64_ty.const_int(head_dim as u64, false);
        let bh = self
            .builder
            .build_int_unsigned_div(outer_idx, new_seq_const, "qkvcu.bh")?;
        let t = self
            .builder
            .build_int_unsigned_rem(outer_idx, new_seq_const, "qkvcu.t")?;
        let dst_token = self
            .builder
            .build_int_add(offset_val, t, "qkvcu.dst_token")?;

        // src_base = (bh * new_seq + t) * head_dim
        let src_token_idx = self
            .builder
            .build_int_mul(bh, new_seq_const, "qkvcu.src.bh_x_ns")?;
        let src_token_idx = self
            .builder
            .build_int_add(src_token_idx, t, "qkvcu.src.bh_t")?;
        let src_base =
            self.builder
                .build_int_mul(src_token_idx, head_dim_const, "qkvcu.src.base_local")?;
        let src_base = self
            .builder
            .build_int_add(new_kv.offset, src_base, "qkvcu.src.base")?;

        // dst_base = (bh * max_seq + dst_token) * head_dim (cache row)
        let dst_token_idx = self
            .builder
            .build_int_mul(bh, max_seq_const, "qkvcu.dst.bh_x_ms")?;
        let dst_token_idx =
            self.builder
                .build_int_add(dst_token_idx, dst_token, "qkvcu.dst.bh_t")?;
        let dst_base =
            self.builder
                .build_int_mul(dst_token_idx, head_dim_const, "qkvcu.dst.base_local")?;
        let dst_base = self
            .builder
            .build_int_add(cache.offset, dst_base, "qkvcu.dst.base")?;

        // Pass 1: max_abs reduction.
        let max_loop_h = self
            .context
            .append_basic_block(*self.func, "qkvcu.maxloop.h");
        self.builder.build_unconditional_branch(max_loop_h)?;
        self.builder.position_at_end(max_loop_h);
        let d_phi = self.builder.build_phi(i64_ty, "qkvcu.d")?;
        let max_phi = self.builder.build_phi(f32_ty, "qkvcu.max")?;
        let d = d_phi.as_basic_value().into_int_value();
        let cur_max = max_phi.as_basic_value().into_float_value();

        let src_idx = self.builder.build_int_add(src_base, d, "qkvcu.src.idx")?;
        let v_f32 = self
            .build_load(&new_kv.clone().set_offset(src_idx))?
            .into_float_value();
        let neg_v = self.builder.build_float_neg(v_f32, "qkvcu.neg")?;
        let abs_v = self
            .build_tail_call(fmax_fn, &[v_f32.into(), neg_v.into()], "qkvcu.abs")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let new_max = self
            .build_tail_call(fmax_fn, &[cur_max.into(), abs_v.into()], "qkvcu.max.new")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let d_next = self
            .builder
            .build_int_add(d, i64_ty.const_int(1, false), "qkvcu.d.next")?;
        let done = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            d_next,
            head_dim_const,
            "qkvcu.maxloop.done",
        )?;
        self.builder
            .build_conditional_branch(done, after_max_loop, max_loop_h)?;
        d_phi.add_incoming(&[(&i64_ty.const_zero(), outer_header), (&d_next, max_loop_h)]);
        max_phi.add_incoming(&[(&f32_ty.const_zero(), outer_header), (&new_max, max_loop_h)]);

        self.builder.position_at_end(after_max_loop);
        let max_abs = new_max;
        let inv127 = f32_ty.const_float(1.0 / 127.0);
        let s_f32 = self.builder.build_float_mul(max_abs, inv127, "qkvcu.s")?;
        let is_zero = self.builder.build_float_compare(
            inkwell::FloatPredicate::OEQ,
            max_abs,
            f32_ty.const_zero(),
            "qkvcu.iszero",
        )?;
        let inv_s_raw =
            self.builder
                .build_float_div(f32_ty.const_float(127.0), max_abs, "qkvcu.inv_s_raw")?;
        let inv_s = self
            .builder
            .build_select(is_zero, f32_ty.const_zero(), inv_s_raw, "qkvcu.inv_s")?
            .into_float_value();

        // scale[bh, dst_token] = s
        let scale_token_idx =
            self.builder
                .build_int_mul(bh, max_seq_const, "qkvcu.scale.bh_x_ms")?;
        let scale_token_idx =
            self.builder
                .build_int_add(scale_token_idx, dst_token, "qkvcu.scale.idx_local")?;
        let scale_idx =
            self.builder
                .build_int_add(scale.offset, scale_token_idx, "qkvcu.scale.idx")?;
        self.build_store(&scale.clone().set_offset(scale_idx), s_f32)?;

        // Pass 2: quantize and write i8.
        let quant_loop_h = self
            .context
            .append_basic_block(*self.func, "qkvcu.quantloop.h");
        self.builder.build_unconditional_branch(quant_loop_h)?;
        self.builder.position_at_end(quant_loop_h);
        let qd_phi = self.builder.build_phi(i64_ty, "qkvcu.qd")?;
        let qd = qd_phi.as_basic_value().into_int_value();

        let src_idx2 = self.builder.build_int_add(src_base, qd, "qkvcu.src.idx2")?;
        let v2_f32 = self
            .build_load(&new_kv.clone().set_offset(src_idx2))?
            .into_float_value();
        let scaled = self
            .builder
            .build_float_mul(v2_f32, inv_s, "qkvcu.scaled")?;
        let pos = self.builder.build_float_compare(
            inkwell::FloatPredicate::OGE,
            scaled,
            f32_ty.const_zero(),
            "qkvcu.pos",
        )?;
        let bias = self
            .builder
            .build_select(
                pos,
                f32_ty.const_float(0.5),
                f32_ty.const_float(-0.5),
                "qkvcu.bias",
            )?
            .into_float_value();
        let biased = self.builder.build_float_add(scaled, bias, "qkvcu.biased")?;
        let q_i32 = self
            .builder
            .build_float_to_signed_int(biased, i32_ty, "qkvcu.q.i32")?;
        let q_clamped_lo = self
            .build_tail_call(
                self.intrinsics.smax_i32,
                &[
                    q_i32.into(),
                    i32_ty.const_int((-128i32) as u64, true).into(),
                ],
                "qkvcu.q.lo",
            )?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let q_clamped = self
            .build_tail_call(
                self.intrinsics.smin_i32,
                &[q_clamped_lo.into(), i32_ty.const_int(127, false).into()],
                "qkvcu.q.hi",
            )?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_int_value();
        let q_i8 = self
            .builder
            .build_int_truncate(q_clamped, i8_ty, "qkvcu.q.i8")?;

        let dst_idx = self.builder.build_int_add(dst_base, qd, "qkvcu.dst.idx")?;
        let dst_gep = unsafe {
            self.builder
                .build_in_bounds_gep(i8_ty, cache.ptr, &[dst_idx], "qkvcu.dst.gep")?
        };
        self.builder.build_store(dst_gep, q_i8)?;

        let qd_next =
            self.builder
                .build_int_add(qd, i64_ty.const_int(1, false), "qkvcu.qd.next")?;
        let q_done = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            qd_next,
            head_dim_const,
            "qkvcu.quantloop.done",
        )?;
        self.builder
            .build_conditional_branch(q_done, after_quant_loop, quant_loop_h)?;
        qd_phi.add_incoming(&[
            (&i64_ty.const_zero(), after_max_loop),
            (&qd_next, quant_loop_h),
        ]);

        self.builder.position_at_end(after_quant_loop);
        self.builder.build_unconditional_branch(outer_latch)?;

        self.finalize_counted_loop(
            outer_phi,
            entry,
            i64_ty.const_int(total_outer as u64, false),
            outer_header,
            exit,
            outer_latch,
        )?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_kv_cache_update(
        &self,
        cache: &TensorPtr<'ctx>,
        new_kv: &TensorPtr<'ctx>,
        offset_ptr: &TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert_eq!(cache.ty.dims.ndim(), 4);
        assert_eq!(new_kv.ty.dims.ndim(), 4);
        let head_dim = cache.ty.dims[3];

        self.builder.position_at_end(entry);
        let i64_ty = self.context.i64_type();
        let offset_gep = unsafe {
            self.builder.build_in_bounds_gep(
                offset_ptr.ty.elem_type.llvm_type(self.context),
                offset_ptr.ptr,
                &[offset_ptr.offset],
                "kvcu.offset.gep",
            )?
        };
        let offset_val = self
            .builder
            .build_load(i64_ty, offset_gep, "kvcu.offset")?
            .into_int_value();
        let offset_in_elems = self.builder.build_int_mul(
            offset_val,
            i64_ty.const_int(head_dim as u64, false),
            "kvcu.offset_elems",
        )?;
        let dst_offset =
            self.builder
                .build_int_add(cache.offset, offset_in_elems, "kvcu.dst_offset")?;

        let sliced_cache = TensorPtr {
            ptr: cache.ptr,
            ty: ResolvedTensorType::with_stride(
                cache.ty.elem_type,
                new_kv.ty.dims.clone(),
                cache.ty.strides().clone(),
            ),
            offset: dst_offset,
            name: format!("{}.kv_slice", cache.name),
        };
        let op = Operation {
            opcode: SingleOpcode::Transfer.into(),
            operands: smallvec![sliced_cache, new_kv.clone()],
        };
        self.build_flat_loop(op, entry, false)
    }

    // FlashAttention-style streaming softmax (no materialized [seq_q, seq_k] buffer).
    //
    //   for each (b, h, sq):                                         // [A] outer loop
    //       m = -inf;  s = 0;  o[D] = 0                              // [B] init
    //       k_bound = active_seq_kv (clamped by causal upper bound)  // [C] bound
    //       for k in 0..k_bound:                                     // [D] k loop
    //           qk = scale * <Q[b,h,sq,:], K[b,h,k,:]>               // [E] qk
    //           qk += mask[b, h, sq, k]                              // [F] mask
    //           m_new   = max(m, qk)                                 // [G] running max
    //           factor  = exp(m - m_new)                             // [G] rescale prev
    //           e       = exp(qk - m_new)                            // [G] new contribution
    //           s       = s * factor + e                             // [G] running sum
    //           o[d]    = o[d] * factor + e * V[b,h,k,d]   for d     // [H] running out
    //           m       = m_new
    //       out[b,h,sq,d] = o[d] / s                       for d     // [I] normalize
    pub fn build_attention(
        &self,
        out: &TensorPtr<'ctx>,
        q: &TensorPtr<'ctx>,
        k: &TensorPtr<'ctx>,
        v: &TensorPtr<'ctx>,
        mask: Option<&TensorPtr<'ctx>>,
        active_seq_kv: Option<&TensorPtr<'ctx>>,
        attn: &operator::Attention,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(q.ty.is_contiguous());
        assert!(k.ty.is_contiguous());
        assert!(v.ty.is_contiguous());
        assert!(out.ty.is_contiguous());
        assert_eq!(q.ty.dims.ndim(), 4);
        let b_dim = q.ty.dims[0];
        let hq = q.ty.dims[1];
        let seq_q = q.ty.dims[2];
        let d_dim = q.ty.dims[3];
        let hkv = k.ty.dims[1];
        let seq_k = k.ty.dims[2];
        assert!(
            hq % hkv == 0,
            "Q head count ({}) must be a multiple of KV head count ({})",
            hq,
            hkv,
        );
        let DataType::Float(float_ty) = q.ty.elem_type else {
            panic!("Attention requires float input");
        };
        let llvm_float = float_ty.llvm_type(self.context);
        let storage_ty = q.ty.elem_type.llvm_type(self.context);
        let is_bf16 = matches!(float_ty, FloatType::BF16);
        let i64_ty = self.context.i64_type();

        let qk_dims = ResolvedTensorDims::new(&[b_dim, hq, seq_q, seq_k]);
        let mask_bc = mask.map(|m| m.ty.broadcast(&qk_dims));

        let neg_inf = llvm_float.const_float(f64::NEG_INFINITY);
        let zero_f = llvm_float.const_zero();
        let scale_const = llvm_float.const_float(attn.scale as f64);

        let exp_fn = self.intrinsics.exp.get(float_ty);
        let fmax_fn = self.intrinsics.fmax.get(float_ty);

        self.builder.position_at_end(entry);
        let array_ty = llvm_float.array_type(d_dim as u32);
        let o_arr = self.builder.build_alloca(array_ty, "attn.o")?;

        let active_val = if let Some(aptr) = active_seq_kv {
            let gep = unsafe {
                self.builder.build_in_bounds_gep(
                    aptr.ty.elem_type.llvm_type(self.context),
                    aptr.ptr,
                    &[aptr.offset],
                    "attn.active.gep",
                )?
            };
            self.builder
                .build_load(i64_ty, gep, "attn.active")?
                .into_int_value()
        } else {
            i64_ty.const_int(seq_k as u64, false)
        };

        let past_len = if attn.is_causal {
            Some(self.builder.build_int_sub(
                active_val,
                i64_ty.const_int(seq_q as u64, false),
                "attn.past_len",
            )?)
        } else {
            None
        };

        let bh_total = (b_dim * hq) as u64;
        let outer_total = bh_total * (seq_q as u64);

        let q_stride_bh = (seq_q * d_dim) as u64;
        let q_stride_sq = d_dim as u64;
        let k_stride_bh = (seq_k * d_dim) as u64;
        let k_stride_sk = d_dim as u64;
        let v_stride_bh = (seq_k * d_dim) as u64;
        let v_stride_sk = d_dim as u64;
        let out_stride_bh = (seq_q * d_dim) as u64;
        let out_stride_sq = d_dim as u64;

        let bb = |name: &str| self.context.append_basic_block(*self.func, name);

        // [A]: outer loop over (b, h, sq), flattened.
        let outer_header = bb("attn.outer.header");
        let outer_body = bb("attn.outer.body");
        let outer_latch = bb("attn.outer.latch");
        let outer_exit = bb("attn.outer.exit");

        self.builder.build_unconditional_branch(outer_header)?;
        let (outer_phi, outer_idx) = self.init_counted_loop(outer_header)?;
        self.builder.build_unconditional_branch(outer_body)?;

        self.builder.position_at_end(outer_body);
        let bh_idx = self.builder.build_int_unsigned_div(
            outer_idx,
            i64_ty.const_int(seq_q as u64, false),
            "bh_idx",
        )?;
        let sq_idx = self.builder.build_int_unsigned_rem(
            outer_idx,
            i64_ty.const_int(seq_q as u64, false),
            "sq_idx",
        )?;
        let b_idx = self.builder.build_int_unsigned_div(
            bh_idx,
            i64_ty.const_int(hq as u64, false),
            "b_idx",
        )?;
        let h_idx = self.builder.build_int_unsigned_rem(
            bh_idx,
            i64_ty.const_int(hq as u64, false),
            "h_idx",
        )?;

        let q_base_off = self.builder.build_int_mul(
            bh_idx,
            i64_ty.const_int(q_stride_bh, false),
            "q.base.bh",
        )?;
        let q_sq_off = self.builder.build_int_mul(
            sq_idx,
            i64_ty.const_int(q_stride_sq, false),
            "q.base.sq",
        )?;
        let q_local = self
            .builder
            .build_int_add(q_base_off, q_sq_off, "q.local")?;
        let q_total = self.builder.build_int_add(q.offset, q_local, "q.total")?;
        let q_row_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(storage_ty, q.ptr, &[q_total], "q.row")?
        };

        // GQA mapping: bh_kv = b * Hkv + h_q * Hkv / Hq.
        let bh_kv_idx = if hq == hkv {
            bh_idx
        } else {
            let bq_factor = (hq / hkv) as u64;
            let kv_b =
                self.builder
                    .build_int_unsigned_div(b_idx, i64_ty.const_int(1, false), "kv.b")?;
            let kv_h = self.builder.build_int_unsigned_div(
                h_idx,
                i64_ty.const_int(bq_factor, false),
                "kv.h",
            )?;
            let scaled = self.builder.build_int_mul(
                kv_b,
                i64_ty.const_int(hkv as u64, false),
                "kv.b.scaled",
            )?;
            self.builder.build_int_add(scaled, kv_h, "kv.bh")?
        };

        let k_base_off = self.builder.build_int_mul(
            bh_kv_idx,
            i64_ty.const_int(k_stride_bh, false),
            "k.base.bh",
        )?;
        let k_total = self
            .builder
            .build_int_add(k.offset, k_base_off, "k.total")?;
        let k_head_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(storage_ty, k.ptr, &[k_total], "k.head")?
        };

        let v_base_off = self.builder.build_int_mul(
            bh_kv_idx,
            i64_ty.const_int(v_stride_bh, false),
            "v.base.bh",
        )?;
        let v_total = self
            .builder
            .build_int_add(v.offset, v_base_off, "v.total")?;
        let v_head_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(storage_ty, v.ptr, &[v_total], "v.head")?
        };

        let mask_bh_sq_off = if let (Some(mptr), Some(mty)) = (mask, mask_bc.as_ref()) {
            let mut total: IntValue<'ctx> = mptr.offset;
            let stride_b = mty.stride(0);
            if stride_b != 0 {
                let term = self.builder.build_int_mul(
                    b_idx,
                    i64_ty.const_int(stride_b as u64, false),
                    "m.b.term",
                )?;
                total = self.builder.build_int_add(total, term, "m.off.b")?;
            }
            let stride_h = mty.stride(1);
            if stride_h != 0 {
                let term = self.builder.build_int_mul(
                    h_idx,
                    i64_ty.const_int(stride_h as u64, false),
                    "m.h.term",
                )?;
                total = self.builder.build_int_add(total, term, "m.off.h")?;
            }
            let stride_sq = mty.stride(2);
            if stride_sq != 0 {
                let term = self.builder.build_int_mul(
                    sq_idx,
                    i64_ty.const_int(stride_sq as u64, false),
                    "m.sq.term",
                )?;
                total = self.builder.build_int_add(total, term, "m.off.sq")?;
            }
            Some(total)
        } else {
            None
        };

        let out_base_off = self.builder.build_int_mul(
            bh_idx,
            i64_ty.const_int(out_stride_bh, false),
            "out.base.bh",
        )?;
        let out_sq_off = self.builder.build_int_mul(
            sq_idx,
            i64_ty.const_int(out_stride_sq, false),
            "out.base.sq",
        )?;
        let out_local = self
            .builder
            .build_int_add(out_base_off, out_sq_off, "out.local")?;
        let out_total = self
            .builder
            .build_int_add(out.offset, out_local, "out.total")?;
        let out_row_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(storage_ty, out.ptr, &[out_total], "out.row")?
        };

        // [B]: init o[d] = 0. m and s are init via PHI at k_loop_header below.
        let o_d_ptrs: Vec<PointerValue<'ctx>> = (0..d_dim)
            .map(|d| -> Result<PointerValue<'ctx>, BuilderError> {
                Ok(unsafe {
                    self.builder.build_in_bounds_gep(
                        array_ty,
                        o_arr,
                        &[i64_ty.const_zero(), i64_ty.const_int(d as u64, false)],
                        &format!("o.d{}", d),
                    )?
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        for ptr in &o_d_ptrs {
            self.builder.build_store(*ptr, zero_f)?;
        }

        // [C]: k_bound = min(active_seq_kv, causal_cap, seq_k).
        let k_bound = if let Some(past_len) = past_len {
            let cap0 = self
                .builder
                .build_int_add(past_len, sq_idx, "k.cap.add_sq")?;
            let cap = self
                .builder
                .build_int_add(cap0, i64_ty.const_int(1, false), "k.cap")?;
            let cmp = self.builder.build_int_compare(
                inkwell::IntPredicate::SLT,
                cap,
                active_val,
                "k.bound.cmp",
            )?;
            self.builder
                .build_select(cmp, cap, active_val, "k.bound")?
                .into_int_value()
        } else {
            active_val
        };
        let seq_k_const = i64_ty.const_int(seq_k as u64, false);
        let cmp_max = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            k_bound,
            seq_k_const,
            "k.bound.cmp_max",
        )?;
        let k_bound = self
            .builder
            .build_select(cmp_max, k_bound, seq_k_const, "k.bound.clamped")?
            .into_int_value();

        // [D]: k loop. m and s are loop-carried via PHI; o[d] lives in alloca.
        let k_loop_header = bb("attn.k.header");
        let k_loop_body = bb("attn.k.body");
        let k_loop_latch = bb("attn.k.latch");
        let after_k = bb("attn.after_k");

        let k_guard = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            k_bound,
            i64_ty.const_zero(),
            "k.guard",
        )?;
        let pre_loop_bb = self.builder.get_insert_block().unwrap();
        self.builder
            .build_conditional_branch(k_guard, after_k, k_loop_header)?;

        self.builder.position_at_end(k_loop_header);
        let m_phi = self.builder.build_phi(llvm_float, "m")?;
        let s_phi = self.builder.build_phi(llvm_float, "s")?;
        let k_phi = self.builder.build_phi(i64_ty, "k.idx")?;
        let m_val = m_phi.as_basic_value().into_float_value();
        let s_val = s_phi.as_basic_value().into_float_value();
        let k_idx = k_phi.as_basic_value().into_int_value();
        self.builder.build_unconditional_branch(k_loop_body)?;

        self.builder.position_at_end(k_loop_body);
        let k_row_off =
            self.builder
                .build_int_mul(k_idx, i64_ty.const_int(k_stride_sk, false), "k.row.off")?;
        let k_row_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(storage_ty, k_head_ptr, &[k_row_off], "k.row.gep")?
        };
        let v_row_off =
            self.builder
                .build_int_mul(k_idx, i64_ty.const_int(v_stride_sk, false), "v.row.off")?;
        let v_row_ptr = unsafe {
            self.builder
                .build_in_bounds_gep(storage_ty, v_head_ptr, &[v_row_off], "v.row.gep")?
        };

        // [E]: qk = scale * <Q[b,h,sq,:], K[b,h,k,:]>. D is unrolled.
        let mut sum = zero_f;
        for d in 0..d_dim {
            let d_const = i64_ty.const_int(d as u64, false);
            let q_d_ptr = unsafe {
                self.builder.build_in_bounds_gep(
                    storage_ty,
                    q_row_ptr,
                    &[d_const],
                    &format!("q.d{}.gep", d),
                )?
            };
            let k_d_ptr = unsafe {
                self.builder.build_in_bounds_gep(
                    storage_ty,
                    k_row_ptr,
                    &[d_const],
                    &format!("k.d{}.gep", d),
                )?
            };
            let q_v = self.load_tensor_f32(storage_ty, q_d_ptr, is_bf16, &format!("q.d{}", d))?;
            let k_v = self.load_tensor_f32(storage_ty, k_d_ptr, is_bf16, &format!("k.d{}", d))?;
            let prod = self
                .builder
                .build_float_mul(q_v, k_v, &format!("dot.prod{}", d))?;
            sum = self
                .builder
                .build_float_add(sum, prod, &format!("dot.sum{}", d))?;
        }
        let qk = self.builder.build_float_mul(sum, scale_const, "qk")?;

        // [F]: qk += mask[b,h,sq,k] using broadcast strides on the mask tensor.
        let qk = if let (Some(mptr), Some(mty), Some(base_off)) =
            (mask, mask_bc.as_ref(), mask_bh_sq_off)
        {
            let stride_k = mty.stride(3);
            let m_off = if stride_k == 0 {
                base_off
            } else {
                let k_term = self.builder.build_int_mul(
                    k_idx,
                    i64_ty.const_int(stride_k as u64, false),
                    "m.k.term",
                )?;
                self.builder.build_int_add(base_off, k_term, "m.off.k")?
            };
            let m_gep = unsafe {
                self.builder.build_in_bounds_gep(
                    mptr.ty.elem_type.llvm_type(self.context),
                    mptr.ptr,
                    &[m_off],
                    "m.gep",
                )?
            };
            let m_v = self
                .builder
                .build_load(llvm_float, m_gep, "m.load")?
                .into_float_value();
            self.builder.build_float_add(qk, m_v, "qk.masked")?
        } else {
            qk
        };

        // [G]: streaming softmax update of (m, s).
        //   m_new  = max(m, qk)
        //   factor = exp(m - m_new)              (rescale prior contributions)
        //   e      = exp(qk - m_new)             (current contribution)
        //   s      = s * factor + e
        let m_new = self
            .builder
            .build_call(fmax_fn, &[m_val.into(), qk.into()], "m.new")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let m_diff = self.builder.build_float_sub(m_val, m_new, "m.diff")?;
        let factor = self
            .builder
            .build_call(exp_fn, &[m_diff.into()], "factor")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let qk_diff = self.builder.build_float_sub(qk, m_new, "qk.diff")?;
        let e = self
            .builder
            .build_call(exp_fn, &[qk_diff.into()], "e")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let s_scaled = self.builder.build_float_mul(s_val, factor, "s.scaled")?;
        let s_new = self.builder.build_float_add(s_scaled, e, "s.new")?;

        // [H]: o[d] = o[d] * factor + e * V[b,h,k,d]. D is unrolled.
        for d in 0..d_dim {
            let d_const = i64_ty.const_int(d as u64, false);
            let v_d_ptr = unsafe {
                self.builder.build_in_bounds_gep(
                    storage_ty,
                    v_row_ptr,
                    &[d_const],
                    &format!("v.d{}.gep", d),
                )?
            };
            let v_d_val =
                self.load_tensor_f32(storage_ty, v_d_ptr, is_bf16, &format!("v.d{}", d))?;
            let o_d_val = self
                .builder
                .build_load(llvm_float, o_d_ptrs[d], &format!("o.d{}.load", d))?
                .into_float_value();
            let o_scaled =
                self.builder
                    .build_float_mul(o_d_val, factor, &format!("o.d{}.scaled", d))?;
            let e_v = self
                .builder
                .build_float_mul(e, v_d_val, &format!("o.d{}.ev", d))?;
            let o_new = self
                .builder
                .build_float_add(o_scaled, e_v, &format!("o.d{}.new", d))?;
            self.builder.build_store(o_d_ptrs[d], o_new)?;
        }

        self.builder.build_unconditional_branch(k_loop_latch)?;

        self.builder.position_at_end(k_loop_latch);
        let k_next = self
            .builder
            .build_int_add(k_idx, i64_ty.const_int(1, false), "k.next")?;
        let k_done =
            self.builder
                .build_int_compare(inkwell::IntPredicate::EQ, k_next, k_bound, "k.done")?;
        self.builder
            .build_conditional_branch(k_done, after_k, k_loop_header)?;

        m_phi.add_incoming(&[(&neg_inf, pre_loop_bb), (&m_new, k_loop_latch)]);
        s_phi.add_incoming(&[(&zero_f, pre_loop_bb), (&s_new, k_loop_latch)]);
        k_phi.add_incoming(&[(&i64_ty.const_zero(), pre_loop_bb), (&k_next, k_loop_latch)]);

        self.builder.position_at_end(after_k);
        let s_final_phi = self.builder.build_phi(llvm_float, "s.final")?;
        s_final_phi.add_incoming(&[(&zero_f, pre_loop_bb), (&s_new, k_loop_latch)]);
        let s_final = s_final_phi.as_basic_value().into_float_value();

        for d in 0..d_dim {
            let d_const = i64_ty.const_int(d as u64, false);
            let o_d_val = self
                .builder
                .build_load(llvm_float, o_d_ptrs[d], &format!("o.final.d{}", d))?
                .into_float_value();
            let normalized =
                self.builder
                    .build_float_div(o_d_val, s_final, &format!("o.norm.d{}", d))?;
            let out_d_ptr = unsafe {
                self.builder.build_in_bounds_gep(
                    storage_ty,
                    out_row_ptr,
                    &[d_const],
                    &format!("out.d{}.gep", d),
                )?
            };
            self.store_tensor_f32(out_d_ptr, normalized, is_bf16)?;
        }

        self.builder.build_unconditional_branch(outer_latch)?;

        self.finalize_counted_loop(
            outer_phi,
            entry,
            i64_ty.const_int(outer_total, false),
            outer_header,
            outer_exit,
            outer_latch,
        )?;
        self.builder.position_at_end(outer_exit);
        Ok(outer_exit)
    }

    fn scalar_to_llvm_value(
        &self,
        value: &ScalarData,
    ) -> Result<BasicValueEnum<'ctx>, BuilderError> {
        let res = match value {
            ScalarData::Bool(v) => self.context.i8_type().const_int(*v as u64, false).into(),
            ScalarData::SInt(ty, v) => {
                let ty = match ty {
                    SIntType::I8 => self.context.i8_type(),
                    SIntType::I32 => self.context.i32_type(),
                    SIntType::I64 => self.context.i64_type(),
                };
                ty.const_int(*v as u64, true).into()
            }
            ScalarData::UInt(ty, v) => {
                let ty = match ty {
                    UIntType::U8 => self.context.i8_type(),
                    UIntType::U64 => self.context.i64_type(),
                };
                ty.const_int(*v, false).into()
            }
            ScalarData::Float(ty, v) => {
                let ty = match ty {
                    FloatType::F32 | FloatType::BF16 => self.context.f32_type(),
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

        let (outer_i, outer_i_val) = self.init_counted_loop(outer_header)?;
        let src = src.set_offset(outer_i_val);
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

        let (inner_i, inner_i_val) = self.init_counted_loop(inner)?;
        let is_on = self.builder.build_int_compare(
            inkwell::IntPredicate::EQ,
            inner_i_val,
            index,
            "is_on",
        )?;
        let val = self
            .builder
            .build_select(is_on, on_value, off_value, "val")?;
        let offset = self.builder.build_int_mul(outer_i_val, depth, "offset")?;
        let offset = self.builder.build_int_add(offset, inner_i_val, "offset")?;
        let dst = dst.set_offset(offset);
        self.build_store(&dst, val)?;

        self.finalize_counted_loop(inner_i, outer_header, depth, inner, outer_latch, inner)?;
        self.finalize_counted_loop(
            outer_i,
            entry,
            self.context
                .i64_type()
                .const_int(src.ty.dims.size() as u64, false),
            outer_header,
            exit,
            outer_latch,
        )?;
        self.builder.position_at_end(exit);

        Ok(exit)
    }

    pub fn build_where(
        &self,
        dst: &TensorPtr<'ctx>,
        cond: &TensorPtr<'ctx>,
        x: &TensorPtr<'ctx>,
        y: &TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let i64_type = self.context.i64_type();
        let output_size = dst.ty.dims.size() as u64;
        let output_dims: Vec<u64> = dst.ty.dims.iter().map(|d| *d as u64).collect();

        let cond_bc = cond.ty.broadcast(&dst.ty.dims);
        let x_bc = x.ty.broadcast(&dst.ty.dims);
        let y_bc = y.ty.broadcast(&dst.ty.dims);

        let cond_info: Vec<(u64, u64)> = (0..cond_bc.dims.ndim())
            .map(|d| (cond_bc.dims[d] as u64, cond_bc.stride(d) as u64))
            .collect();
        let x_info: Vec<(u64, u64)> = (0..x_bc.dims.ndim())
            .map(|d| (x_bc.dims[d] as u64, x_bc.stride(d) as u64))
            .collect();
        let y_info: Vec<(u64, u64)> = (0..y_bc.dims.ndim())
            .map(|d| (y_bc.dims[d] as u64, y_bc.stride(d) as u64))
            .collect();

        let body = self.context.append_basic_block(*self.func, "where.body");
        let exit = self.context.append_basic_block(*self.func, "where.exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let (ind, ind_val) = self.init_counted_loop(body)?;
        let compute_offset =
            |info: &[(u64, u64)], base: IntValue<'ctx>| -> Result<IntValue<'ctx>, BuilderError> {
                let mut offset = base;
                let mut remaining = ind_val;
                for (dim_idx, out_dim) in output_dims.iter().enumerate().rev() {
                    let dim_const = i64_type.const_int(*out_dim, false);
                    let idx = self.builder.build_int_unsigned_rem(
                        remaining,
                        dim_const,
                        &format!("idx.{}", dim_idx),
                    )?;
                    remaining = self.builder.build_int_unsigned_div(
                        remaining,
                        dim_const,
                        &format!("rem.{}", dim_idx),
                    )?;
                    let (_, stride) = info[dim_idx];
                    if stride != 0 {
                        let stride_const = i64_type.const_int(stride, false);
                        let contrib = self.builder.build_int_mul(idx, stride_const, "contrib")?;
                        offset = self.builder.build_int_add(offset, contrib, "off")?;
                    }
                }
                Ok(offset)
            };
        let cond_offset = compute_offset(&cond_info, cond.offset)?;
        let x_offset = compute_offset(&x_info, x.offset)?;
        let y_offset = compute_offset(&y_info, y.offset)?;
        let cond_ptr = cond.clone().set_offset(cond_offset);
        let x_ptr = x.clone().set_offset(x_offset);
        let y_ptr = y.clone().set_offset(y_offset);
        let cond_val = self.build_load(&cond_ptr)?;
        let x_val = self.build_load(&x_ptr)?;
        let y_val = self.build_load(&y_ptr)?;
        let cond_int = cond_val.into_int_value();
        let zero = cond_int.get_type().const_zero();
        let is_true =
            self.builder
                .build_int_compare(inkwell::IntPredicate::NE, cond_int, zero, "is_true")?;
        let selected = self
            .builder
            .build_select(is_true, x_val, y_val, "selected")?;
        let dst_ptr = dst.clone().set_offset(ind_val);
        self.build_store(&dst_ptr, selected)?;
        self.finalize_counted_loop(
            ind,
            entry,
            i64_type.const_int(output_size, false),
            body,
            exit,
            body,
        )?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_slice(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        slices: &[operator::Slice],
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let i64_type = self.context.i64_type();
        let output_size = dst.ty.dims.size() as u64;
        let output_dims: Vec<u64> = dst.ty.dims.iter().map(|d| *d as u64).collect();
        let ndim = output_dims.len();

        // Per-axis start offset (0 if axis not sliced). step=1 only.
        let mut starts: Vec<u64> = vec![0; ndim];
        for s in slices.iter() {
            assert!(s.step == 1, "Slice codegen only supports step=1");
            assert!(s.axis < ndim);
            starts[s.axis] = s.start as u64;
        }

        let src_strides: Vec<u64> = (0..ndim).map(|d| src.stride(d) as u64).collect();
        let dst_strides: Vec<u64> = (0..ndim).map(|d| dst.stride(d) as u64).collect();

        let body = self.context.append_basic_block(*self.func, "slice.body");
        let exit = self.context.append_basic_block(*self.func, "slice.exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let (ind, ind_val) = self.init_counted_loop(body)?;
        let mut src_offset = src.offset;
        let mut dst_offset = dst.offset;
        let mut remaining = ind_val;
        for (dim_idx, out_dim) in output_dims.iter().enumerate().rev() {
            let dim_const = i64_type.const_int(*out_dim, false);
            let out_idx = self.builder.build_int_unsigned_rem(
                remaining,
                dim_const,
                &format!("o_idx.{}", dim_idx),
            )?;
            remaining = self.builder.build_int_unsigned_div(
                remaining,
                dim_const,
                &format!("o_rem.{}", dim_idx),
            )?;
            let in_idx = if starts[dim_idx] != 0 {
                let start_const = i64_type.const_int(starts[dim_idx], false);
                self.builder
                    .build_int_add(out_idx, start_const, &format!("i_idx.{}", dim_idx))?
            } else {
                out_idx
            };
            let src_stride = src_strides[dim_idx];
            if src_stride != 0 {
                let stride_const = i64_type.const_int(src_stride, false);
                let contrib = self.builder.build_int_mul(
                    in_idx,
                    stride_const,
                    &format!("src_contrib.{}", dim_idx),
                )?;
                src_offset = self.builder.build_int_add(
                    src_offset,
                    contrib,
                    &format!("src_off.{}", dim_idx),
                )?;
            }
            let dst_stride = dst_strides[dim_idx];
            if dst_stride != 0 {
                let stride_const = i64_type.const_int(dst_stride, false);
                let contrib = self.builder.build_int_mul(
                    out_idx,
                    stride_const,
                    &format!("dst_contrib.{}", dim_idx),
                )?;
                dst_offset = self.builder.build_int_add(
                    dst_offset,
                    contrib,
                    &format!("dst_off.{}", dim_idx),
                )?;
            }
        }
        let src_ptr = src.clone().set_offset(src_offset);
        let val = self.build_load(&src_ptr)?;
        let dst_ptr = dst.clone().set_offset(dst_offset);
        self.build_store(&dst_ptr, val)?;
        self.finalize_counted_loop(
            ind,
            entry,
            i64_type.const_int(output_size, false),
            body,
            exit,
            body,
        )?;

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    pub fn build_expand(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let i64_type = self.context.i64_type();
        let output_size = dst.ty.dims.size() as u64;
        let output_dims: Vec<u64> = dst.ty.dims.iter().map(|d| *d as u64).collect();

        let src_bc = src.ty.broadcast(&dst.ty.dims);

        let src_info: Vec<(u64, u64)> = (0..src_bc.dims.ndim())
            .map(|d| (src_bc.dims[d] as u64, src_bc.stride(d) as u64))
            .collect();

        let body = self.context.append_basic_block(*self.func, "expand.body");
        let exit = self.context.append_basic_block(*self.func, "expand.exit");

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let (ind, ind_val) = self.init_counted_loop(body)?;
        let mut offset = src.offset;
        let mut remaining = ind_val;
        for (dim_idx, out_dim) in output_dims.iter().enumerate().rev() {
            let dim_const = i64_type.const_int(*out_dim, false);
            let idx = self.builder.build_int_unsigned_rem(
                remaining,
                dim_const,
                &format!("idx.{}", dim_idx),
            )?;
            remaining = self.builder.build_int_unsigned_div(
                remaining,
                dim_const,
                &format!("rem.{}", dim_idx),
            )?;
            let (_, stride) = src_info[dim_idx];
            if stride != 0 {
                let stride_const = i64_type.const_int(stride, false);
                let contrib = self.builder.build_int_mul(idx, stride_const, "contrib")?;
                offset = self.builder.build_int_add(offset, contrib, "off")?;
            }
        }
        let src_ptr = src.clone().set_offset(offset);
        let val = self.build_load(&src_ptr)?;
        let dst_ptr = dst.clone().set_offset(ind_val);
        self.build_store(&dst_ptr, val)?;
        self.finalize_counted_loop(
            ind,
            entry,
            i64_type.const_int(output_size, false),
            body,
            exit,
            body,
        )?;

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
            FloatType::F32 | FloatType::BF16 => f32::MIN as f64,
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

        self.finalize_counted_loop(
            middle_i,
            outer_header,
            i64_type.const_int(middle_bound, false),
            middle_header,
            outer_latch,
            middle_latch,
        )?;

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

    pub fn build_rms_norm(
        &self,
        dst: TensorPtr<'ctx>,
        src: TensorPtr<'ctx>,
        scale: TensorPtr<'ctx>,
        entry: BasicBlock<'ctx>,
        rn: &operator::RMSNormalization,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        assert!(src.ty.is_contiguous() && dst.ty.is_contiguous());

        let axis = rn.axis.index(src.ty.dims.ndim());
        assert_eq!(axis, src.ty.dims.ndim() - 1);

        let val_ty = match src.ty.elem_type {
            DataType::Float(t) => t,
            _ => unimplemented!(),
        };
        let sqrt = self.intrinsics.sqrt.get(val_ty);
        let val_ty = val_ty.llvm_type(self.context);

        let inner_bound = src.ty.dims[axis] as u64;
        let outer_bound = src.ty.dims.size() as u64 / inner_bound;
        let i64_ty = self.context.i64_type();

        let outer_hdr = self.context.append_basic_block(*self.func, "outer.hdr");
        let sumsq_hdr = self.context.append_basic_block(*self.func, "sumsq.hdr");
        let sumsq_done = self.context.append_basic_block(*self.func, "sumsq.done");
        let norm_hdr = self.context.append_basic_block(*self.func, "norm.hdr");
        let outer_latch = self.context.append_basic_block(*self.func, "outer.latch");
        let exit = self.context.append_basic_block(*self.func, "exit");

        let epsilon = val_ty.const_float(rn.epsilon);
        let inner_bound_f = val_ty.const_float(inner_bound as f64);

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(outer_hdr)?;

        let (outer_phi, outer_i) = self.init_counted_loop(outer_hdr)?;
        let outer_offset = self.builder.build_int_mul(
            outer_i,
            i64_ty.const_int(inner_bound, false),
            "outer.offset",
        )?;
        self.builder.build_unconditional_branch(sumsq_hdr)?;

        // Pass 1: sum of x^2
        let (sumsq_idx_phi, sumsq_idx) = self.init_counted_loop(sumsq_hdr)?;
        let sumsq_acc = self.builder.build_phi(val_ty, "sumsq.acc")?;
        let offset = self
            .builder
            .build_int_add(outer_offset, sumsq_idx, "offset")?;
        let src_val = self
            .build_load(&src.clone().set_offset(offset))?
            .into_float_value();
        let sq = self.builder.build_float_mul(src_val, src_val, "sq")?;
        let sumsq_next = self.builder.build_float_add(
            sumsq_acc.as_basic_value().into_float_value(),
            sq,
            "sumsq.next",
        )?;
        self.finalize_counted_loop(
            sumsq_idx_phi,
            outer_hdr,
            i64_ty.const_int(inner_bound, false),
            sumsq_hdr,
            sumsq_done,
            sumsq_hdr,
        )?;
        sumsq_acc.add_incoming(&[
            (&val_ty.const_float(0.0), outer_hdr),
            (&sumsq_next, sumsq_hdr),
        ]);

        // sumsq_done: inv_rms = 1 / sqrt(mean_sq + epsilon)
        self.builder.position_at_end(sumsq_done);
        let mean_sq = self
            .builder
            .build_float_div(sumsq_next, inner_bound_f, "mean.sq")?;
        let mean_sq_eps = self
            .builder
            .build_float_add(mean_sq, epsilon, "mean.sq.eps")?;
        let rms = self
            .build_tail_call(sqrt, &[mean_sq_eps.into()], "rms")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let inv_rms = self
            .builder
            .build_float_div(val_ty.const_float(1.0), rms, "inv.rms")?;
        self.builder.build_unconditional_branch(norm_hdr)?;

        // Pass 2: dst[i] = src[i] * inv_rms * scale[i]
        let (norm_idx_phi, norm_idx) = self.init_counted_loop(norm_hdr)?;
        let offset = self
            .builder
            .build_int_add(outer_offset, norm_idx, "offset")?;
        let src_val = self
            .build_load(&src.clone().set_offset(offset))?
            .into_float_value();
        let scale_val = self
            .build_load(&scale.clone().set_offset(norm_idx))?
            .into_float_value();
        let scaled = self.builder.build_float_mul(src_val, inv_rms, "scaled")?;
        let result = self.builder.build_float_mul(scaled, scale_val, "result")?;
        self.build_store(
            &dst.clone().set_offset(offset),
            result.as_basic_value_enum(),
        )?;
        self.finalize_counted_loop(
            norm_idx_phi,
            sumsq_done,
            i64_ty.const_int(inner_bound, false),
            norm_hdr,
            outer_latch,
            norm_hdr,
        )?;

        self.finalize_counted_loop(
            outer_phi,
            entry,
            i64_ty.const_int(outer_bound, false),
            outer_hdr,
            exit,
            outer_latch,
        )?;

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

        let (ind, ind_val) = self.init_counted_loop(header)?;
        let bound = self
            .context
            .i64_type()
            .const_int(src.ty.dims[depth as usize] as u64, false);
        let src_offset = self.builder.build_int_mul(
            ind_val,
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
            ind_val,
            self.context
                .i64_type()
                .const_int(dst.ty.stride(depth as usize).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        self.finalize_counted_loop(ind, entry, bound, header, exit, latch)?;

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
        let bound = self
            .context
            .i64_type()
            .const_int(indicies.ty.dims[indices_idx] as u64, false);

        self.builder.position_at_end(entry);
        self.builder.build_unconditional_branch(header)?;

        let (ind, ind_val) = self.init_counted_loop(header)?;
        let offset = self.builder.build_int_mul(
            ind_val,
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
            ind_val,
            self.context
                .i64_type()
                .const_int(dst.ty.stride(depth as usize).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        self.finalize_counted_loop(ind, entry, bound, header, exit, latch)?;

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
        assert!(src.ty.dims[src_idx] == dst.ty.dims[depth as usize]);
        let bound = self
            .context
            .i64_type()
            .const_int(dst.ty.dims[depth as usize] as u64, false);
        self.builder.build_unconditional_branch(header)?;

        let (ind, ind_val) = self.init_counted_loop(header)?;
        let src_offset = self.builder.build_int_mul(
            ind_val,
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
            ind_val,
            self.context
                .i64_type()
                .const_int(dst.ty.stride(depth as usize).try_into().unwrap(), false),
            "dst.offset",
        )?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, dst_offset, "dst.offset")?;
        let dst = dst.set_offset(dst_offset);

        self.finalize_counted_loop(ind, entry, bound, header, exit, latch)?;

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

        let (ind, ind_val) = self.init_counted_loop(body)?;
        let mut offset = {
            let mut indexes = Vec::new();
            for (d, s) in izip!(dst.ty.dims.iter(), dst.ty.strides().iter()) {
                let idx = self.builder.build_int_unsigned_div(
                    ind_val,
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

        let src_offset = self
            .builder
            .build_int_add(src.offset, offset, "src.offset")?;
        let src = src.set_offset(src_offset);
        let src_val = self.build_load(&src)?;
        let dst_offset = self
            .builder
            .build_int_add(dst.offset, ind_val, "dst.offset")?;
        let dst = dst.clone().set_offset(dst_offset);
        self.build_store(&dst, src_val)?;

        self.finalize_counted_loop(
            ind,
            entry,
            self.context
                .i64_type()
                .const_int(dst.ty.dims.size().max(1) as u64, false),
            body,
            exit,
            body,
        )?;

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
