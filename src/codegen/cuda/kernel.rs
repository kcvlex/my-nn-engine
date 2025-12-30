use std::fmt::Display;

use delegate::delegate;
use derive_more::From;
use strum_macros::AsRefStr;

use crate::codegen::cuda::*;
use crate::onnx::operator::args;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::DataType;

#[derive(From)]
pub enum CUDAKernel {
    MaxPoolKernel(MaxPoolKernel),
    GeneratedKernel(GeneratedKernel),
    ReduceMatrixKernel(ReduceMatrixKernel),
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
    Size,
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
                KernelVar::Size => "N".to_string(),
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

pub struct KernelBuilder<'sched> {
    schedule: &'sched Schedule,
    decl: KernelDecl,
    local_slot: usize,
}

impl<'sched> KernelBuilder<'sched> {
    pub fn new(schedule: &'sched Schedule, decl: KernelDecl) -> Self {
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

    fn tensor_idx(
        &self,
        value_id: ValueId,
        target_dims: Option<&ResolvedTensorDims>,
    ) -> Result<KernelExpr, BuildError> {
        let ty = self.get_resolved_tensor_type(value_id)?;
        let ty = if let Some(target_dims) = target_dims {
            ty.broadcast(target_dims)
        } else {
            ty.clone()
        };
        match ty.dims.ndim() {
            1 => Ok(KernelVar::Gid.into()),
            d @ (2..=4) => {
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

    fn unary_op(&self, op: &Operator, x: KernelVar) -> KernelExpr {
        KernelExpr::Raw(match op {
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
            uop @ (Operator::Exp |
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
            binop @ (Operator::Add | Operator::Mul | Operator::Pow | Operator::Sub) => {
                let [a, b] = inputs else {
                    panic!("Expected 2 inputs for binary operator")
                };
                self.binary_op(binop, *a, *b)
            }
            _ => unimplemented!(),
        }
    }

    fn new_local_var(&mut self) -> KernelVar {
        let var = KernelVar::Local(self.local_slot);
        self.local_slot += 1;
        var
    }

    fn build_body(&mut self) -> Result<Vec<KernelStmt>, BuildError> {
        let mut stmts = Vec::new();
        let kernel = &self.schedule.kernels[self.decl.kernel_id];

        macro_rules! handle_input_value {
            ($value_id: expr, $target_dims: expr) => {{
                let idx = self.tensor_idx($value_id, $target_dims)?;
                let array = KernelVar::Value($value_id);
                let var = self.new_local_var();
                stmts.push(KernelStmt::DefineVar {
                    ty: TypeSymbol::Primitive(self.get_resolved_tensor_type($value_id)?.elem_type),
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
                    let var = self.new_local_var();
                    stmts.push(KernelStmt::DefineVar {
                        ty: TypeSymbol::Primitive(
                            self.get_resolved_tensor_type(kernel.outputs[0])?.elem_type,
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
                index: Box::new(self.tensor_idx(output_array, None)?),
            },
            rhs: output,
        });
        Ok(stmts)
    }

    pub fn build(&mut self) -> Result<String, BuildError> {
        let body = self.build_body()?;
        Ok(format!(
            "{decl} {{\n\
i64 {gid} = blockIdx.x * blockDim.x + threadIdx.x;\n\
if ({size} <= {gid}) return;\n\
{body}\n\
}}",
            decl = self.decl.decl(),
            gid = KernelVar::Gid.to_string(),
            size = KernelVar::Size.to_string(),
            body = body
                .iter()
                .map(|stmt| stmt.to_string())
                .collect::<Vec<_>>()
                .join("\n"),
        ))
    }
}
