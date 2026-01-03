use std::fmt::Display;

use delegate::delegate;
use derive_more::From;
use itertools::izip;
use strum_macros::AsRefStr;

use crate::codegen::cuda::*;
use crate::onnx::operator;
use crate::onnx::operator::args;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::DataType;

#[derive(From)]
pub enum CUDAKernel {
    MaxPoolKernel(MaxPoolKernel),
    GeneratedKernel(GeneratedKernel),
    ReduceMatrixKernel(ReduceMatrixKernel),
    SoftmaxKernel(SoftmaxKernel),
}

pub struct GeneratedKernel {
    pub decl: KernelDecl,
    pub args: Vec<Expr>,
}

impl GeneratedKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = self.decl.name();
        let args = self
            .args
            .iter()
            .map(|arg| arg.to_string())
            .collect::<Vec<_>>();
        (id, args)
    }
}

pub struct MaxPoolKernel {
    pub ty: DataType,

    pub out: Expr,
    pub in_: Expr,
    pub nbatch: Expr,
    pub channels: Expr,
    pub height: Expr,
    pub width: Expr,
    pub o_height: Expr,
    pub o_width: Expr,
    pub kernel_h: Expr,
    pub kernel_w: Expr,
    pub stride_h: Expr,
    pub stride_w: Expr,
    pub pad_h: Expr,
    pub pad_w: Expr,
}

macro_rules! cast {
    ($ty:expr, $e:expr) => {
        format!("({} *)({})", $ty, $e)
    };
}

impl MaxPoolKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!("max_pool_kernel<{}>", self.ty);
        let args = vec![
            cast!(self.ty, self.out),
            cast!(self.ty, self.in_),
            format!("std::numeric_limits<{}>::min()", self.ty),
            self.nbatch.to_string(),
            self.channels.to_string(),
            self.height.to_string(),
            self.width.to_string(),
            self.o_height.to_string(),
            self.o_width.to_string(),
            self.kernel_h.to_string(),
            self.kernel_w.to_string(),
            self.stride_h.to_string(),
            self.stride_w.to_string(),
            self.pad_h.to_string(),
            self.pad_w.to_string(),
        ];
        (id, args)
    }
}

#[derive(Clone, Copy, AsRefStr)]
pub enum ReduceType {
    #[strum(serialize = "ReduceType::Max")]
    Max,

    #[strum(serialize = "ReduceType::Mean")]
    Mean,
}

impl From<ReduceOp> for ReduceType {
    fn from(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Max => ReduceType::Max,
            ReduceOp::Mean => ReduceType::Mean,
            _ => unimplemented!(),
        }
    }
}

pub struct ReduceMatrixKernel {
    pub data_ty: DataType,
    pub reduce_ty: ReduceType,
    pub block_size: usize,

    pub out: Expr,
    pub in_: Expr,
    pub row: usize,
    pub col: usize,
}

impl ReduceMatrixKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "reduce2d<{}, {}, {}>",
            self.data_ty,
            self.reduce_ty.as_ref(),
            self.block_size
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            self.row.to_string(),
            self.col.to_string(),
        ];
        (id, args)
    }
}

pub struct SoftmaxKernel {
    pub data_ty: DataType,
    pub block_size: usize,
    pub axis_dim: usize,
    pub axis_stride: usize,

    pub out: Expr,
    pub in_: Expr,
    pub size: Expr,
}

impl SoftmaxKernel {
    pub fn fragment(&self) -> (String, Vec<String>) {
        let id = format!(
            "softmax<{}, {}, {}, {}>",
            self.data_ty, self.block_size, self.axis_dim, self.axis_stride
        );
        let args = vec![
            cast!(self.data_ty, self.out),
            cast!(self.data_ty, self.in_),
            self.size.to_string(),
        ];
        (id, args)
    }
}

pub struct LaunchKernel {
    pub cuda_kernel: CUDAKernel,
    pub grid_size: Expr,
    pub block_size: Expr,
    pub shared_mem_bytes: Option<String>,
    pub stream_id: StreamId,
}

impl LaunchKernel {
    delegate! {
        to match &self.cuda_kernel {
            CUDAKernel::MaxPoolKernel(m) => m,
            CUDAKernel::GeneratedKernel(g) => g,
            CUDAKernel::ReduceMatrixKernel(r) => r,
            CUDAKernel::SoftmaxKernel(s) => s,
        } {
            #[call(fragment)]
            fn kernel_fragment(&self) -> (String, Vec<String>);
        }
    }
}

impl std::fmt::Display for LaunchKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (id, args) = self.kernel_fragment();
        write!(
            f,
            "{id}<<<{}, {}, {}, {}>>>({args})",
            self.grid_size,
            self.block_size,
            self.shared_mem_bytes.clone().unwrap_or("0".to_string()),
            self.stream_id,
            args = args.join(", ")
        )
    }
}

#[derive(Clone)]
pub enum TypeSymbol {
    Primitive(DataType),
    Pointer(Box<TypeSymbol>),
}

impl TypeSymbol {
    pub fn to_pointer(&self) -> Self {
        TypeSymbol::Pointer(Box::new(self.clone()))
    }
}

impl Display for TypeSymbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Primitive(dt) => dt.to_string(),
                Self::Pointer(inner) => format!("{}*", inner),
            }
        )
    }
}

impl From<DataType> for TypeSymbol {
    fn from(dt: DataType) -> Self {
        TypeSymbol::Primitive(dt)
    }
}

enum KernelStmt {
    DefineVar {
        ty: TypeSymbol,
        var: KernelVar,
        init: KernelExpr,
    },
    Assign {
        lhs: KernelExpr,
        rhs: KernelExpr,
    },
}

#[derive(Clone)]
pub enum KernelExpr {
    KernelVar(KernelVar),
    ArrayAccess {
        array: Box<KernelExpr>,
        index: Box<KernelExpr>,
    },
    CallFunction {
        name: String,
        args: Vec<KernelExpr>,
    },
    Raw(String),
}

#[derive(Clone, Copy)]
pub enum KernelVar {
    Gid,
    Value(ValueId),
    Local(usize),
}

impl From<KernelVar> for KernelExpr {
    fn from(val: KernelVar) -> Self {
        KernelExpr::KernelVar(val)
    }
}

impl Display for KernelStmt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                KernelStmt::DefineVar { ty, var, init } => {
                    format!("{} {} = {};", ty, var, init)
                }
                KernelStmt::Assign { lhs, rhs } => {
                    format!("{} = {};", lhs, rhs)
                }
            }
        )
    }
}

impl Display for KernelExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                KernelExpr::KernelVar(var) => var.to_string(),
                KernelExpr::ArrayAccess { array, index } => {
                    format!("{}[{}]", array, index)
                }
                KernelExpr::CallFunction { name, args } => {
                    let args_str: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                    format!("{}({})", name, args_str.join(", "))
                }
                KernelExpr::Raw(s) => s.clone(),
            }
        )
    }
}

impl Display for KernelVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                KernelVar::Gid => "gid".to_string(),
                KernelVar::Value(value_id) => format!("value_{}", value_id.index()),
                KernelVar::Local(idx) => format!("local_{}", idx),
            }
        )
    }
}

#[derive(Clone)]
pub struct KernelDecl {
    pub kernel_id: KernelId,
    pub params: Vec<(KernelVar, TypeSymbol)>,
}

impl KernelDecl {
    pub fn name(&self) -> String {
        format!("kernel_{}", self.kernel_id.index())
    }

    pub fn decl(&self) -> String {
        let args: Vec<String> = self
            .params
            .iter()
            .map(|(var, ty)| format!("{} {}", ty, var))
            .collect();
        format!(
            "extern \"C\" __global__ void {}({})",
            self.name(),
            args.join(", ")
        )
    }
}

struct BuilderContext<'sched> {
    schedule: &'sched Schedule,
    decl: KernelDecl,
    local_slot: usize,
}

impl<'sched> BuilderContext<'sched> {
    fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            schedule,
            decl,
            local_slot: 0,
        }
    }

    fn get_resolved_tensor_type(
        &self,
        value_id: ValueId,
    ) -> Result<&ResolvedTensorType, BuildError> {
        self.schedule
            .get_resolved_tensor_type(value_id)
            .ok_or(BuildError::UnresolvedType(value_id))
    }

    fn new_local_var(&mut self) -> KernelVar {
        let var = KernelVar::Local(self.local_slot);
        self.local_slot += 1;
        var
    }

    fn tensor_idx(
        &self,
        value_id: ValueId,
        target_dims: Option<&ResolvedTensorDims>,
    ) -> Result<KernelExpr, BuildError> {
        let ty = self.get_resolved_tensor_type(value_id)?;
        let orig_size = ty.dims.size();
        let (ty, broadcasted) = if let Some(target_dims) = target_dims {
            let res = ty.broadcast(target_dims);
            let broadcasted = res.dims != ty.dims;
            (res, broadcasted)
        } else {
            (ty.clone(), false)
        };
        match ty.dims.ndim() {
            1 => {
                if broadcasted {
                    assert!(orig_size == 1);
                    Ok(KernelExpr::Raw("0".to_string()))
                } else {
                    Ok(KernelVar::Gid.into())
                }
            }
            d @ (2..=5) => {
                let name = format!("to_tensor_idx{}d", d);
                let mut args = vec![KernelVar::Gid.into()];
                for cnst in ty.dims[1..].iter().chain(ty.strides().iter()) {
                    args.push(KernelExpr::Raw(cnst.to_string()));
                }
                Ok(KernelExpr::CallFunction { name, args })
            }
            d => Err(BuildError::UnsupportedTensorDim(value_id, d)),
        }
    }
}

pub struct ElementwiseKernelBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> ElementwiseKernelBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    fn unary_op(&self, op: &Operator, x: KernelVar) -> KernelExpr {
        KernelExpr::Raw(match op {
            Operator::Cast(Cast { to }) => format!("({})({})", to, x),
            Operator::Exp => format!("exp({})", x),
            Operator::Identity => format!("{}", x),
            Operator::LeakyReLU(LeakyReLU { alpha }) => {
                format!("((0 <= {}) ? {} : {} * {})", x, x, alpha, x)
            }
            Operator::Log => format!("log({})", x),
            Operator::Reciprocal => format!("(1.0 / {})", x),
            Operator::ReLU => format!("((0 <= {}) ? {} : 0)", x, x),
            Operator::Sigmoid => format!("(1.0 / (1.0 + exp(-{})))", x),
            Operator::Sqrt => format!("sqrt({})", x),
            Operator::Tanh => format!("tanh({})", x),
            _ => unreachable!(),
        })
    }

    fn binary_op(&self, op: &Operator, lhs: KernelVar, rhs: KernelVar) -> KernelExpr {
        KernelExpr::Raw(match op {
            Operator::Add => format!("({} + {})", lhs, rhs),
            Operator::Div => format!("({} / {})", lhs, rhs),
            Operator::Mul => format!("({} * {})", lhs, rhs),
            // TODO: Support integer types.
            Operator::Pow => format!("pow({}, {})", lhs, rhs),
            Operator::Sub => format!("({} - {})", lhs, rhs),
            _ => unreachable!(),
        })
    }

    fn single_op(&self, op: &Operator, inputs: &[KernelVar]) -> KernelExpr {
        match op {
            Operator::BatchNormalization(BatchNormalization { epsilon, .. }) => {
                let x = inputs[args::BATCHNORM_DATA];
                let scale = inputs[args::BATCHNORM_SCALE];
                let bias = inputs[args::BATCHNORM_BIAS];
                let mean = inputs[args::BATCHNORM_MEAN];
                let var = inputs[args::BATCHNORM_VAR];
                KernelExpr::Raw(format!(
                    "(({x} - {mean}) / sqrt({var} + {epsilon})) * {scale} + {bias}"
                ))
            }
            uop @ (Operator::Cast(_) |
            Operator::Exp |
            Operator::Identity |
            Operator::LeakyReLU(_) |
            Operator::Log |
            Operator::Reciprocal |
            Operator::ReLU |
            Operator::Sigmoid |
            Operator::Sqrt |
            Operator::Tanh) => {
                let [a] = inputs else {
                    panic!("Expected 1 input for unary operator")
                };
                self.unary_op(uop, *a)
            }
            binop @ (Operator::Add |
            Operator::Div |
            Operator::Mul |
            Operator::Pow |
            Operator::Sub) => {
                let [a, b] = inputs else {
                    panic!("Expected 2 inputs for binary operator")
                };
                self.binary_op(binop, *a, *b)
            }
            _ => unimplemented!(),
        }
    }

    fn build_body(&mut self) -> Result<Vec<KernelStmt>, BuildError> {
        let mut stmts = Vec::new();
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];

        macro_rules! handle_input_value {
            ($value_id: expr, $target_dims: expr) => {{
                let idx = self.ctx.tensor_idx($value_id, $target_dims)?;
                let array = KernelVar::Value($value_id);
                let var = self.ctx.new_local_var();
                stmts.push(KernelStmt::DefineVar {
                    ty: TypeSymbol::Primitive(
                        self.ctx.get_resolved_tensor_type($value_id)?.elem_type,
                    ),
                    var,
                    init: KernelExpr::ArrayAccess {
                        array: Box::new(array.into()),
                        index: Box::new(idx),
                    },
                });
                var
            }};
        }

        let output_dims = self
            .ctx
            .get_resolved_tensor_type(kernel.outputs[0])?
            .dims
            .clone();
        let output = match &kernel.body {
            KernelBody::SingleKernel(SingleKernel { op }) => {
                let mut inputs = Vec::with_capacity(kernel.inputs.len());
                for input in kernel.inputs.iter() {
                    let var = handle_input_value!(*input, Some(&output_dims));
                    inputs.push(var);
                }
                self.single_op(op, &inputs)
            }
            KernelBody::FusedElementWises(FusedElementWises { ops }) => {
                let mut outputs = Vec::new();
                for (op, args) in ops.iter() {
                    let mut inputs = Vec::with_capacity(args.len());
                    for input in args.iter() {
                        match input {
                            ElementwiseOpArg::Input(i) => {
                                let var =
                                    handle_input_value!(kernel.inputs[*i], Some(&output_dims));
                                inputs.push(var);
                            }
                            ElementwiseOpArg::NthResult(i) => inputs.push(outputs[*i]),
                        }
                    }
                    let var = self.ctx.new_local_var();
                    stmts.push(KernelStmt::DefineVar {
                        ty: TypeSymbol::Primitive(
                            self.ctx
                                .get_resolved_tensor_type(kernel.outputs[0])?
                                .elem_type,
                        ),
                        var,
                        init: self.single_op(op, &inputs),
                    });
                    outputs.push(var)
                }
                outputs.pop().unwrap().into()
            }
        };

        assert!(kernel.outputs.len() == 1);
        let output_array = kernel.outputs[0];
        stmts.push(KernelStmt::Assign {
            lhs: KernelExpr::ArrayAccess {
                array: Box::new(KernelVar::Value(output_array).into()),
                index: Box::new(self.ctx.tensor_idx(output_array, None)?),
            },
            rhs: output,
        });
        Ok(stmts)
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let body = self
            .build_body()?
            .iter()
            .map(|stmt| stmt.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        let size = {
            let output = self.ctx.schedule.kernels[self.ctx.decl.kernel_id].outputs[0];
            self.ctx.get_resolved_tensor_type(output)?.dims.size()
        };
        let gid = KernelVar::Gid;
        let decl = self.ctx.decl.decl();
        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {body}
}}
"
        ))
    }
}

pub struct SplitBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

fn select_rec<T: Display>(sizes: &[usize], pivot: &T) -> String {
    fn select_rec_impl<T: Display>(
        sizes: &[usize],
        pivot: &T,
        acc_sz: usize,
        depth: usize,
    ) -> String {
        if sizes.len() == 1 {
            return depth.to_string();
        }

        let size = sizes[0];
        let sizes = &sizes[1..];
        let next_acc_sz = acc_sz + size;
        let next_select = select_rec_impl(sizes, pivot, next_acc_sz, depth + 1);
        format!("({pivot} < {next_acc_sz} ? {depth} : {next_select})")
    }

    select_rec_impl(sizes, pivot, 0, 0)
}

fn acc_sizes(sizes: &[usize]) -> Vec<usize> {
    let mut vec = Vec::with_capacity(sizes.len());
    let mut acc = 0;
    for s in sizes.iter() {
        vec.push(acc);
        acc += *s;
    }
    vec
}

impl<'sched> SplitBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::SingleKernel(SingleKernel {
            op: Operator::Split(split),
        }) = &kernel.body
        else {
            panic!("Expected Split operator");
        };

        let axis_idx_var = self.ctx.new_local_var();

        let input_id = kernel.inputs[0];
        let input_ty = self.ctx.get_resolved_tensor_type(input_id)?;
        // if !input_ty.is_contiguous() {
        //     return Err(BuildError::NonContiguousTensor(input_id));
        // }
        let value_ty: TypeSymbol = input_ty.elem_type.into();
        let ptr_ty = value_ty.to_pointer();
        let gid = KernelVar::Gid;
        let size = input_ty.dims.size();
        let axis = split.axis.index(input_ty.dims.ndim());
        let axis_stride = input_ty.strides()[axis];
        let axis_dim = input_ty.dims[axis];
        let (outs, sizes): (Vec<_>, Vec<_>) = kernel
            .outputs
            .iter()
            .map(|id| {
                let output_ty = self.ctx.get_resolved_tensor_type(*id)?;
                let sz = output_ty.dims[axis];
                Ok((*id, sz))
            })
            .collect::<Result<Vec<_>, BuildError>>()?
            .into_iter()
            .unzip();
        let in_ = KernelVar::Value(input_id);
        let select = select_rec(&sizes[..], &axis_idx_var);
        let sizes_acc = acc_sizes(&sizes[..]);
        let outs = outs
            .into_iter()
            .map(|id| format!("{}", KernelVar::Value(id)))
            .collect::<Vec<_>>()
            .join(", ");
        let sizes = sizes
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let sizes_acc = sizes_acc
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let indexes = izip!(input_ty.dims.iter(), input_ty.strides().iter())
            .map(|(dim, stride)| {
                let stride = stride.max(&1);
                format!("({gid} / {stride}) % {dim}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let output_dims_init = input_ty
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let ndim = input_ty.dims.ndim();
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {value_ty} load = {in_}[{gid}];
    int {axis_idx_var} = ({gid} / {axis_stride}) % {axis_dim};
    int select = {select};
    int indexes[] = {{{indexes}}};
    int sizes[] = {{{sizes}}};
    int sizes_acc[] = {{{sizes_acc}}};
    indexes[{axis}] = {axis_idx_var} - sizes_acc[select];
    int output_dims[] = {{{output_dims_init}}};
    output_dims[{axis}] = sizes[select];
    int out_offset = 0;
    for (int i = 0; i < {ndim}; i++) {{
        out_offset *= output_dims[i];
        out_offset += indexes[i];
    }}
    {ptr_ty} outs[] = {{{outs}}};
    outs[select][out_offset] = load;\n
}}
"
        ))
    }
}

pub struct ConcatBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> ConcatBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::SingleKernel(SingleKernel {
            op: Operator::Concat(concat),
        }) = &kernel.body
        else {
            panic!("Expected Concat operator");
        };

        let output_id = kernel.outputs[0];
        let output_ty = self.ctx.get_resolved_tensor_type(output_id)?;
        if !output_ty.is_contiguous() {
            return Err(BuildError::NonContiguousTensor(output_id));
        }
        let size = output_ty.dims.size();
        let value_ty: TypeSymbol = output_ty.elem_type.into();
        let ptr_ty = value_ty.to_pointer();
        let gid = KernelVar::Gid;
        let axis = concat.axis.index(output_ty.dims.ndim());
        let input_tensor_sizes = kernel
            .inputs
            .iter()
            .map(|id| {
                self.ctx
                    .get_resolved_tensor_type(*id)
                    .map(|ty| ty.dims.size())
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let input_select = select_rec(&input_tensor_sizes[..], &gid);
        let input_tensor_sizes_acc = acc_sizes(&input_tensor_sizes[..]);
        let input_axis_sizes = kernel
            .inputs
            .iter()
            .map(|id| {
                let input_ty = self.ctx.get_resolved_tensor_type(*id)?;
                Ok(input_ty.dims[axis])
            })
            .collect::<Result<Vec<_>, BuildError>>()?;
        let input_axis_sizes_acc = acc_sizes(&input_axis_sizes[..]);
        let input_strides = kernel
            .inputs
            .iter()
            .map(|id| {
                let input_ty = self.ctx.get_resolved_tensor_type(*id)?;
                Ok(format!("{{{}}}", input_ty.strides().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")))
            })
            .collect::<Result<Vec<_>, BuildError>>()?
            .join(", ");
        let input_dims = kernel
            .inputs
            .iter()
            .map(|id| {
                let input_ty = self.ctx.get_resolved_tensor_type(*id)?;
                Ok(format!("{{{}}}", input_ty.dims.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")))
            })
            .collect::<Result<Vec<_>, BuildError>>()?
            .join(", ");

        let output_dims = output_ty
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let out = KernelVar::Value(output_id);
        let ndim = output_ty.dims.ndim();
        let ins = kernel
            .inputs
            .iter()
            .map(|id| format!("{}", KernelVar::Value(*id)))
            .collect::<Vec<_>>()
            .join(", ");
        let input_tensor_sizes_acc = input_tensor_sizes_acc
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let input_axis_sizes_acc = input_axis_sizes_acc
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {ptr_ty} ins[] = {{{ins}}};
    int in_sizes_acc[] = {{{input_tensor_sizes_acc}}};
    int select = {input_select};
    int in_offset = {gid} - in_sizes_acc[select];
    {value_ty} load = ins[select][in_offset];
    int input_dims[][{ndim}] = {{{input_dims}}};
    int input_strides[][{ndim}] = {{{input_strides}}};
    int indexes[{ndim}];
    for (int i = 0; i < {ndim}; i++) {{
        indexes[i] = in_offset / input_strides[select][i] % input_dims[select][i];
    }}
    int input_axis_sizes_acc[] = {{{input_axis_sizes_acc}}};
    indexes[{axis}] += input_axis_sizes_acc[select];
    int output_dims[] = {{{output_dims}}};
    int out_offset = 0;
    for (int i = 0; i < {ndim}; i++) {{
        out_offset *= output_dims[i];
        out_offset += indexes[i];
    }}
    {out}[out_offset] = load;
}}
"
        ))
    }
}

pub struct ContiguousBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> ContiguousBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        assert!(matches!(
            kernel.body,
            KernelBody::SingleKernel(SingleKernel {
                op: Operator::Contiguous,
            })
        ));

        let gid = KernelVar::Gid;
        let input = kernel.inputs[0];
        let output = kernel.outputs[0];
        let size = self.ctx.get_resolved_tensor_type(input)?.dims.size();
        let input_idx = self.ctx.tensor_idx(input, None)?;
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {out}[{gid}] = {in_}[{input_idx}];
}}
"
        ))
    }
}

pub struct ResizeBuilder<'sched> {
    ctx: BuilderContext<'sched>,
    stmts: Vec<String>,
}

impl<'sched> ResizeBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
            stmts: Vec::new(),
        }
    }

    fn output_dims(&self) -> Result<Vec<String>, BuildError> {
        let output_ty = self.ctx.schedule.kernels[self.ctx.decl.kernel_id].outputs[0];
        let output_ty = self.ctx.get_resolved_tensor_type(output_ty)?;

        let mut res = Vec::with_capacity(output_ty.dims.ndim());
        let mut acc = 1;
        let gid = KernelVar::Gid;
        for dim in output_ty.dims.iter().rev() {
            res.push(format!("{gid} / {acc} % {dim}"));
            acc *= *dim;
        }

        res.reverse();
        Ok(res)
    }

    fn resize_axis(
        &mut self,
        axis: usize,
        i_dim: usize,
        x_resized: KernelVar,
        resize: &operator::Resize,
    ) -> Result<KernelVar, BuildError> {
        let x_original_var = self.ctx.new_local_var();
        let scale = match resize.scale {
            Some(ref scale) => match scale {
                operator::ResizeScale::Scales(ref scales) => scales[axis] as f32,
                operator::ResizeScale::Sizes(ref sizes) => sizes[axis] as f32 / i_dim as f32,
            },
            None => unreachable!(),
        };
        let x_original = format!("({x_resized} + 0.5) / {scale} - 0.5");
        self.stmts
            .push(format!("float {x_original_var} = {x_original};"));

        let x_original = match resize.mode {
            operator::ResizeMode::Nearest(nearest) => match nearest {
                operator::ResizeNearestMode::RoundPreferFloor => {
                    format!("ceil({x_original_var} - 0.5)")
                }
                operator::ResizeNearestMode::RoundPreferCeil => {
                    format!("floor({x_original_var} + 0.5)")
                }
                operator::ResizeNearestMode::Floor => format!("floor({x_original_var})"),
                operator::ResizeNearestMode::Ceil => format!("ceil({x_original_var})"),
            },
        };
        let x_original_var = self.ctx.new_local_var();
        self.stmts.push(format!(
            "int {x_original_var} = max(0, min((int){x_original}, {v}));",
            v = i_dim - 1
        ));
        Ok(x_original_var)
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let output = kernel.outputs[0];
        let input = kernel.inputs[0];
        let KernelBody::SingleKernel(SingleKernel {
            op: Operator::Resize(resize),
        }) = &kernel.body
        else {
            panic!("Expected Resize operator");
        };

        let x_resized_vec_var = self.ctx.new_local_var();
        let output_dims = self.output_dims()?.join(", ");
        self.stmts
            .push(format!("int {x_resized_vec_var}[] = {{{output_dims}}};"));

        let output_ty = self.ctx.get_resolved_tensor_type(output)?.clone();
        let input_ty = self.ctx.get_resolved_tensor_type(input)?.clone();
        let axes = match &resize.axes {
            Some(axes) => axes
                .iter()
                .map(|a| a.index(output_ty.dims.ndim()))
                .collect::<Vec<_>>(),
            None => (0..output_ty.dims.ndim()).collect(),
        };
        for (i, axis) in axes.iter().copied().enumerate() {
            let x_resized_var = self.ctx.new_local_var();
            self.stmts.push(format!(
                "float {x_resized_var} = (float){x_resized_vec_var}[{axis}];"
            ));
            let x_original_var = self.resize_axis(i, input_ty.dims[axis], x_resized_var, resize)?;
            self.stmts
                .push(format!("{x_resized_vec_var}[{axis}] = {x_original_var};"));
        }

        let in_offset_var = self.ctx.new_local_var();
        let in_offset = input_ty
            .strides()
            .iter()
            .enumerate()
            .map(|(i, stride)| format!("{x_resized_vec_var}[{i}] * {stride}"))
            .collect::<Vec<_>>()
            .join(" + ");
        self.stmts
            .push(format!("int {in_offset_var} = {in_offset};"));

        let gid = KernelVar::Gid;
        let size = output_ty.dims.size();
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let body = self.stmts.join("\n");
        let decl = self.ctx.decl.decl();
        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {body}
    {out}[{gid}] = {in_}[{in_offset_var}];
}}
"
        ))
    }
}

// Only supports:
//   - on_value and off_value are constants.
//   - axis is the last dimension.
pub struct OneHotBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> OneHotBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::SingleKernel(SingleKernel {
            op: Operator::OneHot(one_hot),
        }) = &kernel.body
        else {
            panic!("Expected Resize operator");
        };

        let Some(depth) = one_hot.depth else {
            unimplemented!()
        };
        let Some(on_value) = one_hot.on_value else {
            unimplemented!()
        };
        let Some(off_value) = one_hot.off_value else {
            unimplemented!()
        };

        let indexes = kernel.inputs[args::ONEHOT_INDICES];
        let size = self.ctx.get_resolved_tensor_type(indexes)?.dims.size();
        let decl = self.ctx.decl.decl();
        let gid = KernelVar::Gid;
        let out = KernelVar::Value(kernel.outputs[0]);
        let indexes = KernelVar::Value(indexes);

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int index = {indexes}[{gid}];
    if (index < 0) index += {depth};
    for (int i = 0; i < {depth}; i++) {{
        {out}[{gid} * {depth} + i] = (i == index) ? {on_value} : {off_value};
    }}
}}
"
        ))
    }
}

pub struct GatherBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> GatherBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        let KernelBody::SingleKernel(SingleKernel {
            op: Operator::Gather(gather),
        }) = &kernel.body
        else {
            panic!("Expected Gather operator");
        };

        if gather.axis.raw() != 0 {
            unimplemented!()
        }

        let input_ty = self.ctx.get_resolved_tensor_type(kernel.inputs[0])?;
        let indices_ty = self.ctx.get_resolved_tensor_type(kernel.inputs[1])?;
        let axis_dim = input_ty.dims[0];
        let repeat = input_ty.dims.size() / axis_dim;

        let in_ = KernelVar::Value(kernel.inputs[0]);
        let indices = KernelVar::Value(kernel.inputs[1]);
        let out = KernelVar::Value(kernel.outputs[0]);
        let size = indices_ty.dims.size();
        let gid = KernelVar::Gid;
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;

    int index = {indices}[{gid}];
    if (index < 0) index += {axis_dim};
    for (int i = 0; i < {repeat}; i++) {{
        {out}[{gid} * {repeat} + i] = {in_}[index * {repeat} + i];
    }}
}}
"
        ))
    }
}

pub struct CopyBuilder<'sched> {
    ctx: BuilderContext<'sched>,
}

impl<'sched> CopyBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
        Self {
            ctx: BuilderContext::new(schedule, decl),
        }
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let kernel = &self.ctx.schedule.kernels[self.ctx.decl.kernel_id];
        assert!(matches!(
            kernel.body,
            KernelBody::SingleKernel(SingleKernel {
                op: Operator::Identity,
            })
        ));

        let gid = KernelVar::Gid;
        let input = kernel.inputs[0];
        let output = kernel.outputs[0];
        let size = self.ctx.get_resolved_tensor_type(input)?.dims.size();
        let in_ = KernelVar::Value(input);
        let out = KernelVar::Value(output);
        let out_idx = self.ctx.tensor_idx(output, None)?;
        let decl = self.ctx.decl.decl();

        Ok(format!(
            "
{decl} {{
    int {gid} = blockIdx.x * blockDim.x + threadIdx.x;
    if ({size} <= {gid}) return;
    {out}[{out_idx}] = {in_}[{gid}];
}}
"
        ))
    }
}
