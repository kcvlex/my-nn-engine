use crate::codegen::blas::BLAS;
use crate::codegen::blas::*;
use crate::codegen::llvm::*;
use crate::codegen::omp::*;
use crate::codegen::op::*;
use crate::onnx::operator;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::{DataType, FloatType, ResolvedTensorType};
use inkwell::basic_block::BasicBlock;
use inkwell::builder::{Builder, BuilderError};
use inkwell::context::Context;
use inkwell::module::Module;
use inkwell::types::*;
use inkwell::values::*;

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
            let store_v = self.builder.build_phi(inner_loops.elem_ty, "store.v")?;
            store_v.add_incoming(&[(&load_v, normal), (&inner_loops.elem_ty.const_zero(), pad)]);
            self.build_raw_store(
                inner_loops.elem_ty,
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
        let src_ptr = TensorPtr {
            ptr: inner_loops.src_ptr.ptr,
            ty: inner_loops.src_ptr.ty.clone(),
            offset: src_offset,
            name: format!("src.{}", nest),
        };
        let next_inner_loops = Im2ColsInnerLoop {
            preheader: head,
            exit,
            dst_ptr: inner_loops.dst_ptr,
            dst_offset: dst_offset_int,
            src_ptr,
            is_pad,
            pads: inner_loops.pads,
            outer_offsets: inner_loops.outer_offsets,
            elem_ty: inner_loops.elem_ty,
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
            let elem_ty = src_ptr.ty.elem_type.llvm_type(self.context);

            let inner_loops = Im2ColsInnerLoop {
                preheader,
                exit,

                dst_ptr,
                dst_offset,

                src_ptr,
                is_pad,
                pads: &pads,
                outer_offsets: &offsets,

                elem_ty,
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

    pub fn build_im2col(
        &self,
        dst_info: (PointerValue<'ctx>, &ResolvedTensorDims),
        src_info: (PointerValue<'ctx>, &ResolvedTensorDims),
        elem_type: DataType,
        im2col: &operator::Im2Col,
        entry: BasicBlock<'ctx>,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let (dst_ptr, dst_shape) = dst_info;
        let (src_ptr, src_shape) = src_info;

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
        let offset_src_nbatch = self
            .builder
            .build_phi(self.context.i64_type(), "offset.src.nbatch")?;
        let offset_dst_nbatch_int = offset_dst_nbatch.as_basic_value().into_int_value();
        let offset_src_nbatch_int = offset_src_nbatch.as_basic_value().into_int_value();
        self.builder.build_unconditional_branch(header_channel)?;

        self.builder.position_at_end(header_channel);
        let ind_channel = self
            .builder
            .build_phi(self.context.i64_type(), "ind.channel")?;
        let offset_dst_channel = self
            .builder
            .build_phi(self.context.i64_type(), "offset.dst.channel")?;
        let offset_src_channel = self
            .builder
            .build_phi(self.context.i64_type(), "offset.src.channel")?;
        let offset_dst_channel_int = offset_dst_channel.as_basic_value().into_int_value();
        let offset_src_channel_int = offset_src_channel.as_basic_value().into_int_value();
        let offset_dst = self.builder.build_int_add(
            offset_dst_nbatch_int,
            offset_dst_channel_int,
            "offset.dst",
        )?;
        let offset_src = self.builder.build_int_add(
            offset_src_nbatch_int,
            offset_src_channel_int,
            "offset.src",
        )?;

        let src_ptr = TensorPtr {
            ptr: src_ptr,
            ty: ResolvedTensorType::new(elem_type, src_shape.clone()),
            offset: offset_src,
            name: "src".to_string(),
        };

        self.build_im2col_by_channel_outer(
            (dst_ptr, offset_dst),
            src_ptr,
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
            offset_dst_channel_int,
            self.context
                .i64_type()
                .const_int(next_offset_dst_channel.try_into().unwrap(), false),
            "offset.dst.channel.next",
        )?;
        let next_offset_src_channel = self.builder.build_int_add(
            offset_src_channel_int,
            self.context.i64_type().const_int(
                (src_shape.size() / im2col.nbatch / im2col.channel.inner())
                    .try_into()
                    .unwrap(),
                false,
            ),
            "offset.src.channel.next",
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
            (&offset_dst_nbatch_int, header_nbatch),
        ]);
        offset_src_channel.add_incoming(&[
            (&next_offset_src_channel, exiting_channel),
            (&self.context.i64_type().const_zero(), header_nbatch),
        ]);

        self.builder.position_at_end(exiting_nbatch);
        let next_ind_nbatch = self.builder.build_int_add(
            ind_nbatch.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind.nbatch.next",
        )?;
        let next_offset_dst_nbatch = self.builder.build_int_add(
            offset_dst_nbatch_int,
            self.context.i64_type().const_int(
                (dst_shape.size() / im2col.nbatch).try_into().unwrap(),
                false,
            ),
            "offset.dst.nbatch.next",
        )?;
        let next_offset_src_nbatch = self.builder.build_int_add(
            offset_src_nbatch_int,
            self.context.i64_type().const_int(
                (src_shape.size() / im2col.nbatch).try_into().unwrap(),
                false,
            ),
            "offset.src.nbatch.next",
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
            (&self.context.i64_type().const_int(0, false), entry),
        ]);
        offset_dst_nbatch.add_incoming(&[
            (&next_offset_dst_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_int(0, false), entry),
        ]);
        offset_src_nbatch.add_incoming(&[
            (&next_offset_src_nbatch, exiting_nbatch),
            (&self.context.i64_type().const_int(0, false), entry),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }

    fn build_operation(&self, op: &Operation<'ctx>) -> Result<(), BuilderError> {
        match op {
            Operation::UnaryOp(op, opcode) => {
                let res = match opcode {
                    UnaryOpcode::Exp => {
                        let ty = op.dst.ty.elem_type.float_type().unwrap();
                        let exp = self.intrinsics.exp(ty);
                        let src = self.build_load(&op.src)?.into_float_value();
                        self.build_tail_call(exp, &[src.into()], "res")?
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                    },
                    UnaryOpcode::LeakyReLU(operator::LeakyReLU { alpha }) => {
                        let ty = op.dst.ty.elem_type.float_type().unwrap().llvm_type(self.context);
                        let zero = ty.const_zero();
                        let src = self.build_load(&op.src)?.into_float_value();
                        let lt = self.builder.build_float_compare(
                            inkwell::FloatPredicate::OLT,
                            src,
                            zero,
                            "lt",
                        )?;
                        let lhs =
                            self.builder
                                .build_float_mul(src, ty.const_float(*alpha), "lhs")?;
                        let rhs = src;
                        self.builder.build_select(lt, lhs, rhs, "res")?
                    }
                    UnaryOpcode::ReLU => {
                        let (fmax, zero) = match op.dst.ty.elem_type {
                            DataType::Float(FloatType::F32) => (
                                self.intrinsics.fmax_f32,
                                self.context.f32_type().const_float(0.0),
                            ),
                            DataType::Float(FloatType::F64) => (
                                self.intrinsics.fmax_f64,
                                self.context.f64_type().const_float(0.0),
                            ),
                            _ => unreachable!(),
                        };
                        let src = self.build_load(&op.src)?.into_float_value();
                        self.build_tail_call(fmax, &[src.into(), zero.into()], "res")?
                            .try_as_basic_value()
                            .left()
                            .unwrap()
                    }
                    UnaryOpcode::Transfer => self.build_load(&op.src)?,
                };
                self.build_store(&op.dst, res)
            }
            Operation::BinaryOp(op, opcode) => match opcode {
                BinaryOpcode::BinaryArithmetic(BinaryArithmetic { opcode, is_float }) => {
                    macro_rules! body {
                        ($into: ident, $arith: ident) => {{
                            let lhs = self.build_load(&op.lhs)?.$into();
                            let rhs = self.build_load(&op.rhs)?.$into();
                            let res = self.builder.$arith(lhs, rhs, "res")?;
                            self.build_store(&op.dst, res)
                        }};
                    }

                    match (opcode, is_float) {
                        (BinaryArithmeticOpcode::Add, true) => {
                            body!(into_float_value, build_float_add)
                        }
                        (BinaryArithmeticOpcode::Mul, true) => {
                            body!(into_float_value, build_float_mul)
                        }
                        (BinaryArithmeticOpcode::Add, false) => {
                            body!(into_int_value, build_int_add)
                        }
                        (BinaryArithmeticOpcode::Mul, false) => {
                            body!(into_int_value, build_int_mul)
                        }
                    }
                }
                // BinaryOpcode::IntAdd => todo!(),
                BinaryOpcode::Gemm(ref gemm) => {
                    let prec = gemm.prec;
                    // TODO: Transpose
                    let mut trans_a = gemm.trans_a;
                    let mut trans_b = gemm.trans_b;
                    assert!(op.lhs.ty.dims.ndim() == 2);
                    assert!(op.rhs.ty.dims.ndim() == 2);
                    if op.lhs.ty.stride(0) < op.lhs.ty.stride(1) {
                        trans_a = !trans_a;
                    }
                    if op.rhs.ty.stride(0) < op.rhs.ty.stride(1) {
                        trans_b = !trans_b;
                    }
                    let gemm = GemmArgs {
                        a: (op.lhs.ptr, trans_a),
                        b: (op.rhs.ptr, trans_b),
                        c: (op.dst.ptr, gemm.trans_c),
                        alpha: gemm.alpha,
                        beta: gemm.beta,
                        m: gemm.m as u64,
                        n: gemm.n as u64,
                        k: gemm.k as u64,
                    };
                    self.blas.call_gemm(prec, &gemm, self.builder)?;
                    Ok(())
                }
            },
        }
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
                let (id_v, fmax) = match elem_ty {
                    DataType::Float(FloatType::F32) => (
                        self.context.f32_type().const_float(f32::MIN as f64),
                        self.intrinsics.fmax_f32,
                    ),
                    DataType::Float(FloatType::F64) => (
                        self.context.f64_type().const_float(f64::MIN),
                        self.intrinsics.fmax_f64,
                    ),
                    _ => todo!(),
                };
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

        macro_rules! update_offset {
            ($ptr: expr) => {{
                let add = self.builder.build_int_mul(
                    lb,
                    i64_type.const_int($ptr.stride(nest).try_into().unwrap(), false),
                    "add",
                )?;
                $ptr.offset = self.builder.build_int_add($ptr.offset, add, "offset")?;
            }};
        }
        let mut op_ctx = op_ctx;
        match op_ctx.operation {
            Operation::UnaryOp(ref mut op, _) => {
                update_offset!(op.dst);
                update_offset!(op.src);
            }
            Operation::BinaryOp(ref mut op, _) => {
                update_offset!(op.dst);
                update_offset!(op.lhs);
                update_offset!(op.rhs);
            }
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
        macro_rules! update_offset {
            ($ptr: expr, $offset_phi: expr, $exiting: expr) => {{
                let offset_int = $offset_phi.as_basic_value().into_int_value();
                self.builder.position_at_end($exiting);
                let stride = self
                    .context
                    .i64_type()
                    .const_int($ptr.stride(nest).try_into().unwrap(), false);
                let offset_next = self.builder.build_int_add(
                    offset_int,
                    stride,
                    format!("offset.{}.{}.next", $ptr.name, nest).as_str(),
                )?;
                self.builder.position_at_end(loop_bb.header);
                $offset_phi.add_incoming(&[
                    (
                        &self.context.i64_type().const_int(0, false),
                        loop_bb.preheader,
                    ),
                    (&offset_next, $exiting),
                ]);
                let offset_sum = self.builder.build_int_add(
                    $ptr.offset,
                    offset_int,
                    format!("offset.sum.{}.{}", $ptr.name, nest).as_str(),
                )?;
                TensorPtr {
                    ptr: $ptr.ptr,
                    ty: $ptr.ty,
                    offset: offset_sum,
                    name: $ptr.name,
                }
            }};
        }

        if op_ctx.to_paralleize(nest) {
            let tensors = op_ctx.operation.operands_as_vec();
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
            //call.set_tail_call(true);
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
        let mut next_op_ctx = op_ctx;
        next_op_ctx.operation = match next_op_ctx.operation {
            Operation::UnaryOp(op, opcode) => {
                let offset_phi_dst = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.dst.{}", nest).as_str(),
                )?;
                let offset_phi_src = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.src.{}", nest).as_str(),
                )?;
                let next_dst = update_offset!(op.dst, offset_phi_dst, exiting_bb);
                let next_src = update_offset!(op.src, offset_phi_src, exiting_bb);
                Operation::UnaryOp(
                    UnaryOps {
                        dst: next_dst,
                        src: next_src,
                    },
                    opcode,
                )
            }
            Operation::BinaryOp(op, opcode) => {
                let offset_phi_dst = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.dst.{}", nest).as_str(),
                )?;
                let offset_phi_lhs = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.lhs.{}", nest).as_str(),
                )?;
                let offset_phi_rhs = self.builder.build_phi(
                    self.context.i64_type(),
                    format!("offset.rhs.{}", nest).as_str(),
                )?;
                let next_dst = update_offset!(op.dst, offset_phi_dst, exiting_bb);
                let next_lhs = update_offset!(op.lhs, offset_phi_lhs, exiting_bb);
                let next_rhs = update_offset!(op.rhs, offset_phi_rhs, exiting_bb);
                Operation::BinaryOp(
                    BinaryOps {
                        dst: next_dst,
                        lhs: next_lhs,
                        rhs: next_rhs,
                    },
                    opcode,
                )
            }
        };
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

    pub fn build_batchnorm_by_channel(
        &self,
        dst: PointerValue<'ctx>,
        inputs: &[PointerValue<'ctx>],
        elem_ty: DataType,
        mn: (u64, u64),
        preheader: BasicBlock<'ctx>,
        batchnorm: &operator::BatchNormalization,
    ) -> Result<BasicBlock<'ctx>, BuilderError> {
        let src = inputs[operator::args::BATCHNORM_DATA];
        let scale_ptr = inputs[operator::args::BATCHNORM_SCALE];
        let bias_ptr = inputs[operator::args::BATCHNORM_BIAS];
        let mean_ptr = inputs[operator::args::BATCHNORM_MEAN];
        let variance_ptr = inputs[operator::args::BATCHNORM_VAR];
        let (m, n) = mn;
        let (fp_type, sqrt, fma) = match elem_ty {
            DataType::Float(FloatType::F32) => (
                self.context.f32_type(),
                self.intrinsics.sqrt_f32,
                self.intrinsics.fma_f32,
            ),
            DataType::Float(FloatType::F64) => (
                self.context.f64_type(),
                self.intrinsics.sqrt_f64,
                self.intrinsics.fma_f64,
            ),
            _ => unreachable!(),
        };
        let epsilon = fp_type.const_float(batchnorm.epsilon as f64);

        let header = self.context.append_basic_block(*self.func, "entry");
        let exit = self.context.append_basic_block(*self.func, "exit");
        let exiting = self.context.append_basic_block(*self.func, "exiting");
        let body = self.context.append_basic_block(*self.func, "body");

        self.builder.build_unconditional_branch(header)?;

        self.builder.position_at_end(header);

        let ind0 = self.builder.build_phi(self.context.i64_type(), "ind0")?;
        let offset0 = self.builder.build_phi(self.context.i64_type(), "offset0")?;
        let ind0_int = ind0.as_basic_value().into_int_value();
        let offset0_int = offset0.as_basic_value().into_int_value();
        let scale = self
            .build_raw_load(fp_type, scale_ptr, ind0_int)?
            .into_float_value();
        let bias = self
            .build_raw_load(fp_type, bias_ptr, ind0_int)?
            .into_float_value();
        let mean = self
            .build_raw_load(fp_type, mean_ptr, ind0_int)?
            .into_float_value();
        let variance = self
            .build_raw_load(fp_type, variance_ptr, ind0_int)?
            .into_float_value();
        let factor = self.builder.build_float_add(variance, epsilon, "factor")?;
        let factor = self
            .build_tail_call(sqrt, &[factor.into()], "factor")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        let factor = self.builder.build_float_div(scale, factor, "factor")?;

        self.builder.build_unconditional_branch(body)?;

        self.builder.position_at_end(body);
        let ind1 = self.builder.build_phi(self.context.i64_type(), "ind1")?;
        let offset1 = self.builder.build_int_add(
            offset0_int,
            ind1.as_basic_value().into_int_value(),
            "offset1",
        )?;
        let val = self
            .build_raw_load(fp_type, src, offset1)?
            .into_float_value();
        let val = self.builder.build_float_sub(val, mean, "val.sub.mean")?;
        let val = self
            .build_tail_call(fma, &[val.into(), factor.into(), bias.into()], "val")?
            .try_as_basic_value()
            .left()
            .unwrap()
            .into_float_value();
        self.build_raw_store(fp_type, dst, offset1, val)?;
        let ind1_next = self.builder.build_int_add(
            ind1.as_basic_value().into_int_value(),
            self.context.i64_type().const_int(1, false),
            "ind1.next",
        )?;
        let cond1 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind1_next,
            self.context.i64_type().const_int(n, false),
            "cond1",
        )?;
        self.builder
            .build_conditional_branch(cond1, body, exiting)?;
        ind1.add_incoming(&[
            (&ind1_next, body),
            (&self.context.i64_type().const_zero(), header),
        ]);

        self.builder.position_at_end(exiting);
        let ind0_next = self.builder.build_int_add(
            ind0_int,
            self.context.i64_type().const_int(1, false),
            "ind0.next",
        )?;
        let offset0_next = self.builder.build_int_add(
            offset0_int,
            self.context.i64_type().const_int(n, false),
            "offset0.next",
        )?;
        let cond0 = self.builder.build_int_compare(
            inkwell::IntPredicate::SLT,
            ind0_next,
            self.context.i64_type().const_int(m, false),
            "cond0",
        )?;
        self.builder.build_conditional_branch(cond0, header, exit)?;
        ind0.add_incoming(&[
            (&ind0_next, exiting),
            (&self.context.i64_type().const_zero(), preheader),
        ]);
        offset0.add_incoming(&[
            (&offset0_next, exiting),
            (&self.context.i64_type().const_zero(), preheader),
        ]);

        self.builder.position_at_end(exit);
        Ok(exit)
    }
}

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

    elem_ty: BasicTypeEnum<'ctx>,
    nest: u64,
}
