use inkwell::basic_block::BasicBlock;
use inkwell::builder::Builder;
use inkwell::builder::BuilderError;
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::*;
use inkwell::values::*;
use smallvec::smallvec;

use crate::codegen::cpu::blas::*;
use crate::codegen::cpu::llvm::*;
use crate::codegen::cpu::omp::*;
use crate::codegen::cpu::op::*;
use crate::onnx::operator;
use crate::schedule::ElementwiseOpArg;
use crate::tensor::types::DataType;
use crate::tensor::types::FloatType;

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

    fn build_im2col_by_channel_inner(
        &self,
        inner_loops: Im2ColsInnerLoop<'_, 'ctx>,
        im2col: &operator::Im2Col,
    ) -> Result<IntValue<'ctx>, BuilderError> {
        if inner_loops.nest as usize == inner_loops.outer_offsets.len() {
            // dbg!(&inner_loops);
            let prolog = self.context.append_basic_block(*self.func, "inner.prolog");
            let normal = self.context.append_basic_block(*self.func, "inner.normal");
            let pad = self.context.append_basic_block(*self.func, "inner.pad");
            let epilog = self.context.append_basic_block(*self.func, "inner.epilog");

            self.builder.build_unconditional_branch(prolog)?;
            self.builder.position_at_end(prolog);
            self.builder
                .build_conditional_branch(inner_loops.is_pad, pad, normal)?;

            self.builder.position_at_end(pad);
            self.builder.build_unconditional_branch(epilog)?;

            self.builder.position_at_end(normal);
            let load_v = self.build_load(&inner_loops.src_ptr)?;
            self.builder.build_unconditional_branch(epilog)?;

            self.builder.position_at_end(epilog);
            let ty = inner_loops.src_ptr.ty.elem_type;
            let (ty, id_v) = match (ty, im2col.pad_val) {
                (ty, operator::PadVal::Zero) => {
                    let ty = ty.llvm_type(self.context);
                    (ty, ty.const_zero())
                }
                (DataType::Float(ty), operator::PadVal::NInf) => {
                    let id_v = match ty {
                        FloatType::F32 => f32::NEG_INFINITY as f64,
                        FloatType::F64 => f64::NEG_INFINITY,
                    };
                    let ty = ty.llvm_type(self.context);
                    let id_v = ty.const_float(id_v).as_basic_value_enum();
                    (ty.as_basic_type_enum(), id_v)
                }
                _ => unreachable!(),
            };
            let store_v = self.builder.build_phi(ty, "store.v")?;
            store_v.add_incoming(&[(&load_v, normal), (&id_v, pad)]);
            self.build_raw_store(
                ty,
                inner_loops.dst_ptr,
                inner_loops.dst_offset,
                store_v.as_basic_value(),
            )?;
            let next_dst_offset = self.builder.build_int_add(
                inner_loops.dst_offset,
                self.context.i64_type().const_int(1, false),
                "next.dst.offset",
            )?;
            self.builder.build_unconditional_branch(inner_loops.exit)?;
            return Ok(next_dst_offset);
        }

        let head = self.context.append_basic_block(*self.func, "inner.head");
        let exit = self.context.append_basic_block(*self.func, "inner.exit");
        let nest = inner_loops.nest;

        self.builder.position_at_end(inner_loops.preheader);
        self.builder.build_unconditional_branch(head)?;

        self.builder.position_at_end(head);
        let ind = self.builder.build_phi(self.context.i64_type(), "ind")?;
        let dst_offset = self
            .builder
            .build_phi(self.context.i64_type(), "dst.offset")?;
        let src_inner_offset = self
            .builder
            .build_phi(self.context.i64_type(), "src.inner.offset")?;
        let dst_offset_int = dst_offset.as_basic_value().into_int_value();
        let src_inner_offset_int = src_inner_offset.as_basic_value().into_int_value();
        let src_offset = self.builder.build_int_add(
            inner_loops.outer_offsets[nest as usize],
            src_inner_offset_int,
            "src.offset",
        )?;
        let is_pad_left = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            src_offset,
            self.context
                .i64_type()
                .const_int(inner_loops.pads[nest as usize], false),
            "is.pad.left",
        )?;
        let is_pad_right = self.builder.build_int_compare(
            inkwell::IntPredicate::SLE,
            self.context.i64_type().const_int(
                u64::try_from(inner_loops.src_ptr.ty.dims[nest as usize + 2]).unwrap() +
                    inner_loops.pads[nest as usize],
                false,
            ),
            src_offset,
            "is.pad.right",
        )?;
        let is_pad_i = self
            .builder
            .build_or(is_pad_left, is_pad_right, "is.pad.i")?;
        let is_pad = self
            .builder
            .build_or(inner_loops.is_pad, is_pad_i, "is.pad")?;

        let src_offset = self.builder.build_int_sub(
            src_offset,
            self.context
                .i64_type()
                .const_int(inner_loops.pads[nest as usize], false),
            "src.offset",
        )?;
        let src_offset = self.builder.build_int_mul(
            src_offset,
            self.context.i64_type().const_int(
                inner_loops
                    .src_ptr
                    .stride(nest as usize + 2)
                    .try_into()
                    .unwrap(),
                false,
            ),
            "src.inner.offset.mul",
        )?;
        let src_offset =
            self.builder
                .build_int_add(inner_loops.src_ptr.offset, src_offset, "src.offset")?;
        let src_ptr = inner_loops
            .src_ptr
            .with_offset(src_offset)
            .with_name(format!("src.{}", nest));
        let next_inner_loops = Im2ColsInnerLoop {
            preheader: head,
            exit,
            dst_ptr: inner_loops.dst_ptr,
            dst_offset: dst_offset_int,
            src_ptr,
            is_pad,
            pads: inner_loops.pads,
            outer_offsets: inner_loops.outer_offsets,
            nest: nest + 1,
        };

        let next_dst_offset = self.build_im2col_by_channel_inner(next_inner_loops, im2col)?;

        self.builder.position_at_end(exit);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.next",
        )?;
        let dilation = self
            .context
            .i64_type()
            .const_int(im2col.dilations[nest as usize].try_into().unwrap(), false);
        let src_inner_offset_next = self.builder.build_int_add(
            src_inner_offset.as_basic_value().into_int_value(),
            dilation,
            "src.inner.offset.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind_next,
            self.context.i64_type().const_int(
                im2col.one_kernel_shape[nest as usize].try_into().unwrap(),
                false,
            ),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, head, inner_loops.exit)?;
        ind.add_incoming(&[
            (&ind_next, exit),
            (&self.context.i64_type().const_zero(), inner_loops.preheader),
        ]);
        src_inner_offset.add_incoming(&[
            (&src_inner_offset_next, exit),
            (&self.context.i64_type().const_zero(), inner_loops.preheader),
        ]);
        dst_offset.add_incoming(&[
            (&next_dst_offset, exit),
            (&inner_loops.dst_offset, inner_loops.preheader),
        ]);
        Ok(next_dst_offset)
    }

    fn build_im2col_by_channel_outer(
        &self,
        dst_info: (PointerValue<'ctx>, IntValue<'ctx>),
        src_ptr: TensorPtr<'ctx>,
        offsets: Vec<IntValue<'ctx>>,
        nest: usize,
        blocks: (BasicBlock<'ctx>, BasicBlock<'ctx>),
        im2col: &operator::Im2Col,
    ) -> Result<IntValue<'ctx>, BuilderError> {
        let (dst_ptr, dst_offset) = dst_info;
        let (preheader, exit) = blocks;
        let max_nest = im2col.one_fm_shape.ndim();
        if nest == max_nest {
            let is_pad = self.context.bool_type().const_int(0, false);
            let pads = (0..max_nest)
                .map(|i| match &im2col.pad {
                    operator::ConvPad::NotSet(pad) => pad[i].0,
                    operator::ConvPad::Valid => 0,
                    operator::ConvPad::SameLower | operator::ConvPad::SameUpper => {
                        let padded_len = im2col.padded_len(i);
                        let pad_len = padded_len - src_ptr.ty.dims[i + 2];
                        let pad_left = pad_len / 2;
                        let add_left =
                            pad_len % 2 == 1 && matches!(im2col.pad, operator::ConvPad::SameLower);
                        pad_left + add_left as usize
                    }
                })
                .map(|x| x.try_into().unwrap())
                .collect::<Vec<u64>>();

            let inner_loops = Im2ColsInnerLoop {
                preheader,
                exit,

                dst_ptr,
                dst_offset,

                src_ptr,
                is_pad,
                pads: &pads,
                outer_offsets: &offsets,

                nest: 0,
            };

            return self.build_im2col_by_channel_inner(inner_loops, im2col);
        }

        let head = self.context.append_basic_block(*self.func, "outer.head");
        let exiting = self.context.append_basic_block(*self.func, "outer.exit");

        self.builder.build_unconditional_branch(head)?;

        self.builder.position_at_end(head);
        let dst_offset_init = dst_offset;
        let src_offset = self
            .builder
            .build_phi(self.context.i64_type(), "src.offset")?;
        let dst_offset = self
            .builder
            .build_phi(self.context.i64_type(), "dst.offset")?;
        let src_offset_int = src_offset.as_basic_value().into_int_value();
        let dst_offset_int = dst_offset.as_basic_value().into_int_value();
        let mut offsets = offsets;
        offsets.push(src_offset_int);
        let next_dst_offset = self.build_im2col_by_channel_outer(
            (dst_ptr, dst_offset_int),
            src_ptr,
            offsets,
            nest + 1,
            (head, exiting),
            im2col,
        )?;

        self.builder.position_at_end(exiting);
        let next_dst_offset = if nest + 1 == max_nest {
            match im2col.channel {
                operator::Channel::Meld(channel) => self.builder.build_int_add(
                    next_dst_offset,
                    self.context.i64_type().const_int(
                        ((channel - 1) * im2col.one_kernel_shape.size())
                            .try_into()
                            .unwrap(),
                        false,
                    ),
                    "next.dst.offset",
                )?,
                operator::Channel::Split(_) => next_dst_offset,
            }
        } else {
            next_dst_offset
        };
        let padded_img_size: i64 = im2col.padded_len(nest).try_into().unwrap();
        let kernel_size: i64 = im2col.one_kernel_shape[nest].try_into().unwrap();
        let dilation: i64 = im2col.dilations[nest].try_into().unwrap();
        let next_src_offset = self.builder.build_int_add(
            src_offset_int,
            self.context
                .i64_type()
                .const_int(im2col.strides[nest].try_into().unwrap(), false),
            "src.offset.next",
        )?;
        let bound = padded_img_size - (kernel_size - 1) * dilation;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            next_src_offset,
            self.context
                .i64_type()
                .const_int(bound.try_into().unwrap(), false),
            "cond",
        )?;
        self.builder.build_conditional_branch(cond, head, exit)?;
        dst_offset.add_incoming(&[(&next_dst_offset, exiting), (&dst_offset_init, preheader)]);
        src_offset.add_incoming(&[
            (&next_src_offset, exiting),
            (&self.context.i64_type().const_zero(), preheader),
        ]);
        Ok(next_dst_offset)
    }

    // We assume that `dst` is contiguous.
    pub fn build_im2col(
        &self,
        dst: &TensorPtr<'ctx>,
        src: &TensorPtr<'ctx>,
        im2col: &operator::Im2Col,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let header_nbatch = self
            .context
            .append_basic_block(*self.func, "im2col.header.nbatch");
        let exiting_nbatch = self
            .context
            .append_basic_block(*self.func, "im2col.exit.nbatch");
        let header_channel = self
            .context
            .append_basic_block(*self.func, "im2col.header.channel");
        let exiting_channel = self
            .context
            .append_basic_block(*self.func, "im2col.exit.channel");
        let exit = self.context.append_basic_block(*self.func, "im2col.exit");

        self.builder.build_unconditional_branch(header_nbatch)?;

        self.builder.position_at_end(header_nbatch);
        let ind_nbatch = self
            .builder
            .build_phi(self.context.i64_type(), "ind.nbatch")?;
        let offset_dst_nbatch = self
            .builder
            .build_phi(self.context.i64_type(), "offset.dst.nbatch")?;
        self.builder.build_unconditional_branch(header_channel)?;

        self.builder.position_at_end(header_channel);
        let ind_channel = self
            .builder
            .build_phi(self.context.i64_type(), "ind.channel")?;
        let offset_dst_channel = self
            .builder
            .build_phi(self.context.i64_type(), "offset.dst.channel")?;
        let offset_dst = self.builder.build_int_add(
            offset_dst_nbatch.as_basic_value().into_int_value(),
            offset_dst_channel.as_basic_value().into_int_value(),
            "offset.dst",
        )?;
        let offset_src_nbatch = self.builder.build_int_mul(
            ind_nbatch.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(src.ty.stride(0).try_into().unwrap(), false),
            "offset.src.nbatch",
        )?;
        let offset_src_channel = self.builder.build_int_mul(
            ind_channel.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(src.ty.stride(1).try_into().unwrap(), false),
            "offset.src.channel",
        )?;
        let offset_src =
            self.builder
                .build_int_add(offset_src_nbatch, offset_src_channel, "offset.src")?;

        let src = src.clone().with_offset(offset_src);

        self.build_im2col_by_channel_outer(
            (dst.ptr, offset_dst),
            src,
            vec![],
            0,
            (header_channel, exiting_channel),
            im2col,
        )?;

        self.builder.position_at_end(exiting_channel);
        let next_ind_channel = self.builder.build_int_add(
            ind_channel.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.channel.next",
        )?;
        let next_offset_dst_channel = match im2col.channel {
            operator::Channel::Meld(_) => im2col.one_kernel_shape.size(),
            operator::Channel::Split(_) => {
                im2col.one_fm_shape.size() * im2col.one_kernel_shape.size()
            }
        };
        let next_offset_dst_channel = self.builder.build_int_add(
            offset_dst_channel.as_basic_value().into_int_value(),
            self.context
                .i64_type()
                .const_int(next_offset_dst_channel.try_into().unwrap(), false),
            "offset.dst.channel.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            next_ind_channel,
            self.context
                .i64_type()
                .const_int(im2col.channel.inner().try_into().unwrap(), false),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, header_channel, exiting_nbatch)?;
        ind_channel.add_incoming(&[
            (&next_ind_channel, exiting_channel),
            (&self.context.i64_type().const_int(0, false), header_nbatch),
        ]);
        offset_dst_channel.add_incoming(&[
            (&next_offset_dst_channel, exiting_channel),
            (&self.context.i64_type().const_zero(), header_nbatch),
        ]);

        self.builder.position_at_end(exiting_nbatch);
        let next_ind_nbatch = self.builder.build_int_add(
            ind_nbatch.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.nbatch.next",
        )?;
        let next_offset_dst_nbatch = self.builder.build_int_add(
            offset_dst_nbatch.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(
                (dst.ty.dims.size() / im2col.nbatch).try_into().unwrap(),
                false,
            ),
            "offset.dst.nbatch.next",
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            next_ind_nbatch,
            self.context
                .i64_type()
                .const_int(im2col.nbatch.try_into().unwrap(), false),
            "cond",
        )?;
        self.builder
            .build_conditional_branch(cond, header_nbatch, exit)?;
        ind_nbatch.add_incoming(&[
            (&next_ind_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_zero(), entry),
        ]);
        offset_dst_nbatch.add_incoming(&[
            (&next_offset_dst_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_zero(), entry),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
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
            opcode @ (SingleOpcode::Add | SingleOpcode::Mul | SingleOpcode::Sub) => {
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
                    (SingleOpcode::Mul, true) => {
                        body!(into_float_value, build_float_mul)
                    }
                    (SingleOpcode::Sub, true) => {
                        body!(into_float_value, build_float_sub)
                    }
                    (SingleOpcode::Add, false) => {
                        body!(into_int_value, build_int_add)
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
                let ty = ty.float_type().unwrap().llvm_type(self.context);
                let src = unary_op!(operands).into_float_value();
                let src = self
                    .builder
                    .build_float_ext(src, self.context.f64_type(), "ext")?;
                let res = self
                    .build_tail_call(self.intrinsics.tanh, &[src.into()], "res")?
                    .try_as_basic_value()
                    .left()
                    .unwrap();
                self.builder
                    .build_float_trunc(res.into_float_value(), ty, "res")?
                    .as_basic_value_enum()
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

    pub fn build_gemm(
        &self,
        dst: &TensorPtr<'ctx>,
        a: &TensorPtr<'ctx>,
        b: &TensorPtr<'ctx>,
        c: Option<&TensorPtr<'ctx>>,
        entry: BasicBlock<'ctx>,
        gemm: &operator::Gemm,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        // TODO?: Omit if c.ptr == dst.ptr.
        let entry = if let Some(c) = c {
            let op = Operation {
                opcode: SingleOpcode::Transfer.into(),
                operands: smallvec![dst.clone(), c.clone()],
            };
            let op = OperationContext {
                operation: op,
                omp_ctx: None,
                omp_parallel: None,
                omp_for: None,
            };
            self.build_nested_loop(op, entry, dst.ty.dims.ndim())?
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

    fn init_outlined(&self) -> Result<(OMPContext<'ctx>, LoopBB<'ctx>), BuilderError> {
        let entry = self.context.append_basic_block(*self.func, "entry");
        let body = self.context.append_basic_block(*self.func, "body");
        let exit = self.context.append_basic_block(*self.func, "exit");

        let i32_type = self.context.i32_type();

        self.builder.position_at_end(entry);
        let is_last = self.builder.build_alloca(i32_type, "is.last")?;
        let lb = self.builder.build_alloca(i32_type, "lb")?;
        let ub = self.builder.build_alloca(i32_type, "ub")?;
        let stride = self.builder.build_alloca(i32_type, "stride")?;
        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(exit);
        self.builder.build_return(None)?;

        self.builder.position_at_end(body);

        let omp_ctx = OMPContext {
            global_tid: self.func.get_nth_param(0).unwrap().into_pointer_value(),
            is_last,
            lb,
            ub,
            stride,
        };
        let loop_bb = LoopBB {
            preheader: entry,
            header: body,
            exit,
        };
        Ok((omp_ctx, loop_bb))
    }

    fn build_omp_outlined(
        &self,
        op_ctx: OperationContext<'ctx>,
        nest: usize,
        max_nest: usize,
    ) -> Result<FunctionValue<'ctx>, BuilderError> {
        let fn_type = op_ctx.operation.outlined_type(self.context);
        let fn_name = format!("{}.outlined", self.func.get_name().to_str().unwrap());
        let outlined_fn = self.module.add_function(&fn_name, fn_type, None);

        let new_builder = self.context.create_builder();
        let translator = {
            let mut translator = self.clone();
            translator.builder = &new_builder;
            translator.func = &outlined_fn;
            translator
        };
        let (omp_ctx, loop_bb) = translator.init_outlined()?;
        let op_ctx = op_ctx.to_outlined(&translator, omp_ctx)?;
        translator.build_nested_loop_rec(op_ctx, loop_bb, nest, max_nest, None)?;

        Ok(outlined_fn)
    }

    fn build_omp_for(
        &self,
        op_ctx: OperationContext<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
        max_nest: usize,
    ) -> Result<(), BuilderError> {
        let new_header = self
            .context
            .append_basic_block(*self.func, "omp.for.header");
        let prolog_bb = self
            .context
            .append_basic_block(*self.func, "omp.for.prolog");
        let epilog_bb = self
            .context
            .append_basic_block(*self.func, "omp.for.epilog");

        let i32_type = self.context.i32_type();
        let len: u64 = op_ctx.operation.result_dims()[nest].try_into().unwrap();
        let len = i32_type.const_int(len - 1, false);
        let OMPContext {
            global_tid,
            is_last,
            lb,
            ub,
            stride,
        } = op_ctx.omp_ctx.clone().unwrap();

        let tid = self
            .builder
            .build_load(i32_type, global_tid, "global.tid")?
            .as_basic_value_enum()
            .into_int_value();
        let one = i32_type.const_int(1, false);
        for (ptr, val) in [
            (is_last, i32_type.const_zero()),
            (lb, i32_type.const_zero()),
            (ub, len),
            (stride, one),
        ] {
            // self.builder.build_call(self.intrinsics.lifetime_start, &[i64_type.const_int(4, false).into(), ptr.into()], "")?;
            self.builder.build_store(ptr, val)?;
        }

        let args = StaticInitArgs {
            tid,
            sched: ScheduleType::UnorderedStatic,
            is_last,
            lb,
            ub,
            stride,
            incr: one,
        };
        self.omp.static_init(self.builder, &args)?;
        let lb = self
            .builder
            .build_load(i32_type, lb, "lb")?
            .into_int_value();
        let ub = self
            .builder
            .build_load(i32_type, ub, "ub_omp")?
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
            .build_conditional_branch(cond, prolog_bb, epilog_bb)?;

        self.builder.position_at_end(epilog_bb);
        self.omp
            .static_fini(self.builder, &StaticFiniArgs { tid })?;
        self.builder.build_unconditional_branch(loop_bb.exit)?;

        self.builder.position_at_end(prolog_bb);
        let i64_type = self.context.i64_type();
        let lb = self.builder.build_int_s_extend(lb, i64_type, "lb")?;
        let ub = self.builder.build_int_s_extend(ub, i64_type, "ub")?;

        let mut op_ctx = op_ctx;
        for op in op_ctx.operation.operands.iter_mut() {
            let add = self.builder.build_int_mul(
                lb,
                i64_type.const_int(op.stride(nest).try_into().unwrap(), false),
                "add",
            )?;
            op.offset = self.builder.build_int_add(op.offset, add, "offset")?;
        }
        self.builder.build_unconditional_branch(new_header)?;

        op_ctx.omp_for = None;
        let new_loop_bb = LoopBB {
            preheader: prolog_bb,
            header: new_header,
            exit: epilog_bb,
        };
        let loop_range = Some((lb, ub));
        self.builder.position_at_end(new_header);
        self.build_nested_loop_rec(op_ctx, new_loop_bb, nest, max_nest, loop_range)
    }

    fn build_nested_loop_rec(
        &self,
        op_ctx: OperationContext<'ctx>,
        loop_bb: LoopBB<'ctx>,
        nest: usize,
        max_nest: usize,
        loop_range: Option<(IntValue<'ctx>, IntValue<'ctx>)>,
    ) -> Result<(), BuilderError> {
        if op_ctx.to_paralleize(nest) {
            let tensors = op_ctx.operation.operands.to_vec();
            let mut args = Vec::with_capacity(tensors.len() * 2);
            for (i, tensor) in tensors.iter().enumerate() {
                let ptr_ptr = self
                    .builder
                    .build_alloca(tensor.ptr.get_type(), format!("tensor_{i}_ptr").as_str())?;
                let offset_ptr = self.builder.build_alloca(
                    tensor.offset.get_type(),
                    format!("tensor_{i}_offset").as_str(),
                )?;
                // TODO: Stop using hard-coded size value
                //self.builder.build_call(self.intrinsics.lifetime_start, &[
                //    self.context.i64_type().const_int(8, false).into(),
                //    ptr_ptr.into(),
                //], "")?;
                self.builder.build_store(ptr_ptr, tensor.ptr)?;
                // self.builder.build_call(self.intrinsics.lifetime_start, &[
                //     self.context.i64_type().const_int(8, false).into(),
                //     offset_ptr.into(),
                // ], "")?;
                self.builder.build_store(offset_ptr, tensor.offset)?;
                args.push(ptr_ptr);
                args.push(offset_ptr);
            }

            let outlined = self.build_omp_outlined(op_ctx, nest, max_nest)?;
            let args = ForkCallArgs { outlined, args };
            let call = self.omp.fork_call(self.builder, &args)?;
            call.set_tail_call(true);
            self.builder.build_unconditional_branch(loop_bb.exit)?;

            return Ok(());
        }

        if op_ctx.to_for(nest) {
            return self.build_omp_for(op_ctx, loop_bb, nest, max_nest);
        }

        if nest == max_nest {
            self.build_operation(&op_ctx.operation)?;
            self.builder.build_unconditional_branch(loop_bb.exit)?;
            return Ok(());
        }
        let ind = self
            .builder
            .build_phi(self.context.i64_type(), format!("ind.{}", nest).as_str())?;
        let exiting_bb = self
            .context
            .append_basic_block(*self.func, format!("exit.{}", nest).as_str());
        let bound = match loop_range {
            Some((_, ub)) => ub,
            None => {
                let bound: u64 = op_ctx.operation.result_dims()[nest].try_into().unwrap();
                self.context.i64_type().const_int(bound, false)
            }
        };

        // Update op
        let mut next_op_ctx = op_ctx;
        let phis = next_op_ctx
            .operation
            .operands
            .iter()
            .map(|op| {
                self.builder.build_phi(
                    op.offset.get_type(),
                    format!("offset.{}.{}", op.name, nest).as_str(),
                )
            })
            .collect::<Result<Vec<_>, BuilderError>>()?;
        for (op, offset_phi) in next_op_ctx.operation.operands.iter_mut().zip(phis) {
            let offset_int = offset_phi.as_basic_value().into_int_value();
            self.builder.position_at_end(exiting_bb);
            let stride = self
                .context
                .i64_type()
                .const_int(op.stride(nest).try_into().unwrap(), false);
            let offset_next = self.builder.build_int_add(
                offset_int,
                stride,
                format!("offset.{}.{}.next", op.name, nest).as_str(),
            )?;
            self.builder.position_at_end(loop_bb.header);
            offset_phi.add_incoming(&[
                (
                    &self.context.i64_type().const_int(0, false),
                    loop_bb.preheader,
                ),
                (&offset_next, exiting_bb),
            ]);
            op.offset = self.builder.build_int_add(
                op.offset,
                offset_int,
                format!("offset.sum.{}.{}", op.name, nest).as_str(),
            )?;
        }

        // Comp and branch
        let next_bb = self
            .context
            .append_basic_block(*self.func, format!("loop.{}", nest).as_str());
        self.builder.build_unconditional_branch(next_bb)?;
        self.builder.position_at_end(exiting_bb);
        let ind_next = self.builder.build_int_add(
            ind.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            format!("ind.{}.next", nest).as_str(),
        )?;
        let cond = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind_next,
            bound,
            format!("cond.{}", nest).as_str(),
        )?;
        self.builder
            .build_conditional_branch(cond, loop_bb.header, loop_bb.exit)?;

        let ind_init = match loop_range {
            Some((lb, _)) => lb,
            None => self.context.i64_type().const_zero(),
        };
        ind.add_incoming(&[(&ind_init, loop_bb.preheader), (&ind_next, exiting_bb)]);
        self.builder.position_at_end(next_bb);
        let next_loop_bb = LoopBB {
            preheader: loop_bb.header,
            header: next_bb,
            exit: exiting_bb,
        };
        self.build_nested_loop_rec(next_op_ctx, next_loop_bb, nest + 1, max_nest, None)
    }

    pub fn build_nested_loop(
        &self,
        op_ctx: OperationContext<'ctx>,
        preheader: BasicBlock<'ctx>,
        max_nest: usize,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let header = self.context.append_basic_block(*self.func, "header");
        let exit = self.context.append_basic_block(*self.func, "exit");
        self.builder.build_unconditional_branch(header)?;
        self.builder.position_at_end(header);
        let loop_bb = LoopBB {
            preheader,
            header,
            exit,
        };
        self.build_nested_loop_rec(op_ctx, loop_bb, 0, max_nest, None)?;
        self.builder.position_at_end(exit);
        Ok(exit)
    }

    fn build_resize_rec(&self, param: ResizeParam<'_, 'ctx>) -> Result<(), BuilderError> {
        let ResizeParam {
            mut dst,
            mut src,
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
        dst.offset = self
            .builder
            .build_int_add(dst.offset, dst_offset_add, "dst.offset")?;

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
        src.offset = self
            .builder
            .build_int_add(src.offset, src_offset_add, "src.offset")?;
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
        let max_nest = dst.ty.dims.ndim();
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
                dst.clone().with_offset(offset).with_type(new_ty)
            };
            let op = Operation {
                opcode: SingleOpcode::Transfer.into(),
                operands: smallvec![dst, src.clone()],
            };
            let op = OperationContext {
                operation: op,
                omp_ctx: None,
                omp_for: None,
                omp_parallel: None,
            };
            entry = self.build_nested_loop(op, entry, max_nest)?;
            acc += src.ty.dims[axis] * stride;
        }
        Ok(entry)
    }
}

#[derive(Debug)]
struct LoopBB<'ctx> {
    preheader: BasicBlock<'ctx>,
    header: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,
}

#[derive(Debug)]
struct Im2ColsInnerLoop<'a, 'ctx: 'a> {
    preheader: BasicBlock<'ctx>,
    exit: BasicBlock<'ctx>,

    dst_ptr: PointerValue<'ctx>,
    dst_offset: IntValue<'ctx>,

    src_ptr: TensorPtr<'ctx>,
    is_pad: IntValue<'ctx>,
    pads: &'a [u64],
    outer_offsets: &'a [IntValue<'ctx>],

    nest: u64,
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
