mod early_broadcast;
pub mod infer;
pub mod verify;

use itertools::izip;
use itertools::Itertools;
use num::Zero;

use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::model::UnifyMode;
use crate::onnx::operator::*;
use crate::options::Options;
use crate::tensor::data::TensorData;
use crate::tensor::types::broadcast_shape;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::ResolvedTensorType;
use crate::tensor::types::SIntType;
use crate::tensor::types::TensorType;
use crate::tensor::types::TypeError;
use crate::transform::modify::SimpleGraphOp;
use crate::transform::shape::early_broadcast::EarlyBroadcast;
use crate::transform::shape::infer::ShapeInference;
use crate::transform::shape::verify::ShapeVerification;
use crate::transform::PassManager;
use crate::transform::SimplePassManager;
use crate::transform::Target;

fn reshape(
    a: &ResolvedTensorType,
    shape: &[i64],
    mode: UnifyMode,
) -> Result<ResolvedTensorType, TypeError> {
    let prod0 = a.dims.size();
    let prod1 = shape
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, x)| *x != -1)
        .map(|(i, x)| if x == 0 { a.dims[i] } else { x as usize })
        .product::<usize>();
    let shape = ResolvedTensorDims::new(
        shape
            .iter()
            .copied()
            .enumerate()
            .map(|(i, x)| {
                if x == -1 {
                    prod0 / prod1
                } else if x == 0 {
                    a.dims[i]
                } else {
                    x as usize
                }
            })
            .collect_vec()
            .as_slice(),
    );

    assert!(
        a.dims.size() == shape.size() ||
            (a.dims.compatible_with_scalar() && shape.compatible_with_scalar())
    );
    let reshaped = if matches!(mode, UnifyMode::CheckStrides) {
        a.try_reshape(&shape)
            .ok_or(TypeError::InferError("Unsupported reshape".to_string()))?
    } else {
        ResolvedTensorType::new(a.elem_type, shape.clone())
    };
    Ok(reshaped)
}

fn transpose(
    data: &ResolvedTensorType,
    transpose: &Transpose,
    mode: UnifyMode,
) -> Result<ResolvedTensorType, TypeError> {
    let perm = transpose
        .perm(data.dims.ndim())
        .ok_or(TypeError::InferError("Invalid permutation".to_string()))?;

    let ty = data.transpose(perm.as_slice());
    let ty = if matches!(mode, UnifyMode::CheckStrides) {
        ty
    } else {
        ty.contiguous()
    };
    Ok(ty)
}

// TODO: Remove _target.
pub fn infer_node_output(
    graph: &Graph,
    node_id: NodeId,
    mode: UnifyMode,
    _target: Target,
) -> Result<Vec<ResolvedTensorType>, TypeError> {
    let node = &graph.nodes[node_id];

    let inputs: Vec<&ResolvedTensorType> = node
        .inputs
        .iter()
        .filter_map(|id| *id)
        .flat_map(|id| {
            graph.values[id].ty.as_ref().map(|x| match x {
                TensorType::Resolved(x) => Some(x),
                TensorType::Unresolved(_) => None,
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(TypeError::UnresolvedInput)?;

    let mut res: Vec<ResolvedTensorType> = Vec::new();
    match &node.op {
        Operator::Add | Operator::Div | Operator::Sub | Operator::Mul => {
            let a = &inputs[0];
            let b = &inputs[1];

            assert_eq!(a.elem_type, b.elem_type);
            let dims = broadcast_shape(&a.dims, &b.dims)?;
            res.push(ResolvedTensorType::new(a.elem_type, dims));
        }
        Operator::BatchedGemm(gemm) => {
            let a = &inputs[0];
            let b = &inputs[1];
            assert!(a.dims.ndim() >= 3 && b.dims.ndim() >= 3);
            let ndim = a.dims.ndim();
            let (m, k_a) = if gemm.trans_a {
                (a.dims[ndim - 1], a.dims[ndim - 2])
            } else {
                (a.dims[ndim - 2], a.dims[ndim - 1])
            };
            let (k_b, n) = if gemm.trans_b {
                (b.dims[ndim - 1], b.dims[ndim - 2])
            } else {
                (b.dims[ndim - 2], b.dims[ndim - 1])
            };
            assert_eq!(k_a, k_b);
            let prefix = broadcast_shape(&a.dims.prefix(ndim - 2), &b.dims.prefix(ndim - 2))?;
            let mut dims = prefix;
            dims.push(m);
            dims.push(n);
            res.push(ResolvedTensorType::new(a.elem_type, dims));
        }
        Operator::Attention(_) => {
            let q = &inputs[args::ATTENTION_Q];
            let k = &inputs[args::ATTENTION_K];
            let v = &inputs[args::ATTENTION_V];
            assert!(q.dims.ndim() == k.dims.ndim() && k.dims.ndim() == v.dims.ndim());
            if q.dims.ndim() != 4 {
                unimplemented!()
            }
            let batch_size = q.dims[0];
            let q_num_heads = q.dims[1];
            let kv_num_heads = k.dims[1];

            assert!(q_num_heads == kv_num_heads);
            assert!(k.dims[0] == batch_size);
            assert!(v.dims[0] == batch_size);
            assert!(k.dims[1] == kv_num_heads);

            let q_sequence_length = q.dims[2];
            let head_size = q.dims[3];
            let kv_sequence_length = k.dims[2];
            let v_head_size = v.dims[3];

            assert!(k.dims[3] == head_size);
            assert!(v.dims[2] == kv_sequence_length);

            res.push(ResolvedTensorType::new(
                q.elem_type,
                ResolvedTensorDims::new(&[batch_size, q_num_heads, q_sequence_length, v_head_size]),
            ));
        }
        Operator::Pow => {
            let a = &inputs[0];
            let b = &inputs[1];
            let dims = broadcast_shape(&a.dims, &b.dims)?;
            res.push(ResolvedTensorType::new(a.elem_type, dims));
        }
        Operator::BatchNormalization(_) |
        Operator::Clip(_) |
        Operator::Cos |
        Operator::Exp |
        Operator::GeLU(_) |
        Operator::Identity |
        Operator::LayerNormalization(_) |
        Operator::LeakyReLU(_) |
        Operator::Log |
        Operator::Neg |
        Operator::Reciprocal |
        Operator::ReLU |
        Operator::Sigmoid |
        Operator::Sin |
        Operator::Softmax(_) |
        Operator::Sqrt |
        Operator::Tanh |
        Operator::Transfer(_) => {
            res.push(inputs[0].clone());
        }
        Operator::Cast(Cast { to }) => {
            // TODO: Check for element type
            let input = &inputs[0];
            res.push(ResolvedTensorType::new(*to, input.dims.clone()));
        }
        Operator::Transpose(ref t) => {
            let data = &inputs[args::TRANSPOSE_DATA];
            let ty = transpose(data, t, mode)?;
            res.push(ty);
        }
        Operator::Reshape => {
            let a = &inputs[args::RESHAPE_DATA];
            let shape = &graph
                .initializer
                .get(&node.inputs[args::RESHAPE_SHAPE].unwrap())
                .ok_or(TypeError::UnresolvedInput)?
                .data;
            let shape = match shape {
                TensorData::SInt(SIntType::I64, ref v) => v,
                _ => {
                    return Err(TypeError::InferError("Invalid shape".to_string()));
                }
            };
            res.push(reshape(a, shape, mode)?);
        }
        Operator::Resize(resize) => {
            let dims = resize
                .resized_shape(graph, node_id)
                .ok_or(TypeError::InferError("Invalid resized shape".to_string()))?;
            res.push(ResolvedTensorType::new(inputs[0].elem_type, dims));
        }
        Operator::Conv(ref conv) => {
            let x = &inputs[args::CONV_DATA];
            let w = &inputs[args::CONV_WEIGHT];
            if matches!(mode, UnifyMode::CheckStrides) {
                assert!(x.is_contiguous(), "Conv input must be contiguous");
                assert!(w.is_contiguous(), "Conv weight must be contiguous");
            }
            let dims = conv.output_shape(&x.dims, &w.dims);
            let ty = ResolvedTensorType::new(x.elem_type, dims);
            if matches!(mode, UnifyMode::CheckStrides) {
                assert!(ty.is_contiguous(), "Conv output must be contiguous");
            }
            res.push(ty);
        }
        Operator::MatMul => {
            let a = &inputs[args::MATMUL_LHS];
            let b = &inputs[args::MATMUL_RHS];

            assert_eq!(a.elem_type, b.elem_type);

            let (ldim, l_prepended) = if a.dims.ndim() == 1 {
                (ResolvedTensorDims::new(&[1, a.dims[0]]), true)
            } else {
                (a.dims.clone(), false)
            };
            let (rdim, r_prepended) = if b.dims.ndim() == 1 {
                (ResolvedTensorDims::new(&[b.dims[0], 1]), true)
            } else {
                (b.dims.clone(), false)
            };

            let l_prefix = ldim.prefix(ldim.ndim() - 2);
            let r_prefix = rdim.prefix(rdim.ndim() - 2);
            let prefix = broadcast_shape(&l_prefix, &r_prefix)?;

            let l_suffix = ldim.suffix(2);
            let r_suffix = rdim.suffix(2);
            assert_eq!(l_suffix[1], r_suffix[0]);

            let mut output = prefix;
            if !l_prepended {
                output.push(l_suffix[0]);
            }
            if !r_prepended {
                output.push(r_suffix[1]);
            }
            res.push(ResolvedTensorType::new(a.elem_type, output));
        }
        Operator::AveragePool(ref pooling) | Operator::MaxPool(ref pooling) => {
            let x = &inputs[args::MAXPOOL_DATA];
            let dims = pooling.output_shape(&x.dims);
            res.push(ResolvedTensorType::new(x.elem_type, dims));
        }
        Operator::GlobalAveragePool => {
            let x = &inputs[0];
            let nbatch = x.dims[0];
            let channel = x.dims[1];
            let mut dims = vec![1; x.dims.ndim()];
            dims[0] = nbatch;
            dims[1] = channel;
            res.push(ResolvedTensorType::new(
                x.elem_type,
                ResolvedTensorDims::new(&dims),
            ));
        }

        Operator::Gemm(Gemm {
            trans_a, trans_b, ..
        }) => {
            let a = &inputs[args::GEMM_A];
            let b = &inputs[args::GEMM_B];

            let m = a.dims[if !*trans_a { 0 } else { 1 }];
            let n = b.dims[if !*trans_b { 1 } else { 0 }];
            // let a_k = a.dims[1 - a_idx];
            // let b_k = b.dims[b_idx];
            // assert!(a_k == b_k);
            res.push(ResolvedTensorType::new(
                a.elem_type,
                ResolvedTensorDims::new(&[m, n]),
            ));
        }
        Operator::ReduceMax(ref reduce) |
        Operator::ReduceMean(ref reduce) |
        Operator::ReduceSum(ref reduce) => {
            let data = &inputs[0];
            let rank = data.dims.ndim();
            let axes = reduce
                .normalize_axes(rank)
                .ok_or(TypeError::InferError("Invalid axes".to_string()))?;

            let mut drop = vec![false; rank];
            for i in axes.iter() {
                drop[*i] = true;
            }
            let mut dims = Vec::with_capacity(rank - axes.len());
            for (i, &d) in data.dims.iter().enumerate() {
                if !reduce.keepdims && drop[i] {
                    continue;
                }
                let d = if drop[i] { 1 } else { d };
                dims.push(d);
            }

            res.push(ResolvedTensorType::new(
                data.elem_type,
                ResolvedTensorDims::new(&dims),
            ));
        }

        Operator::Concat(Concat { ref axis }) => {
            let mut dims = inputs[0].dims.clone();
            let axis = axis.index(dims.ndim());
            for input in inputs.iter().skip(1) {
                if dims.ndim() != input.dims.ndim() {
                    return Err(TypeError::InferError("Concat: Different ranks".to_string()));
                }
                for i in 0..dims.ndim() {
                    if i == axis {
                        dims[i] += input.dims[i];
                    } else if dims[i] != input.dims[i] {
                        return Err(TypeError::InferError("Concat: Different dim".to_string()));
                    }
                }
            }
            res.push(ResolvedTensorType::new(inputs[0].elem_type, dims));
        }

        Operator::Shape(Shape { start, end }) => {
            let ndim = inputs[0].dims.ndim();
            let start = start.index(ndim);
            let end = end.map(|x| x.index(ndim)).unwrap_or(ndim);
            res.push(ResolvedTensorType::new(
                SIntType::I64.into(),
                ResolvedTensorDims::new(&[end - start]),
            ));
        }

        Operator::Slice => {
            let slices = Slice::collect_slices(graph, node_id)
                .ok_or(TypeError::InferError("Invalid slices".to_string()))?;
            res.push(inputs[0].slices(&slices));
        }

        Operator::Split(ref split) => {
            let input = inputs[0];
            let axis = split.axis.index(input.dims.ndim());
            split
                .split(&input.dims)
                .ok_or(TypeError::InferError("Invalid split dims".to_string()))?
                .into_iter()
                .for_each(|x| {
                    let mut dims = input.dims.clone();
                    dims[axis] = x;
                    res.push(ResolvedTensorType::new(input.elem_type, dims))
                });
        }

        Operator::Squeeze(Squeeze { axes }) => {
            let input = &inputs[0];
            let input_dims = &input.dims;
            let rank = input_dims.ndim();
            let drop = if let Some(axes) = axes {
                let mut drop = vec![false; rank];
                for &axis in axes.iter() {
                    let axis = axis.index(rank);
                    assert!(axis < rank);
                    assert_eq!(input_dims[axis], 1);
                    drop[axis] = true;
                }
                drop
            } else {
                (0..rank).map(|i| input_dims[i] == 1).collect()
            };

            let mut dims = Vec::new();
            let mut strides = Vec::new();
            for (i, (d, s)) in izip!(input_dims.iter(), input.strides().iter()).enumerate() {
                if !drop[i] {
                    dims.push(*d);
                    strides.push(*s);
                }
            }

            let ty = if matches!(mode, UnifyMode::CheckStrides) {
                ResolvedTensorType::with_stride(
                    input.elem_type,
                    ResolvedTensorDims::new(&dims),
                    ResolvedTensorDims::new(&strides),
                )
            } else {
                ResolvedTensorType::new(input.elem_type, ResolvedTensorDims::new(&dims))
            };
            res.push(ty);
        }

        Operator::Gather(Gather { axis }) => {
            let input = &inputs[0].dims;
            let indices = &inputs[1].dims;
            let ndim = input.ndim();
            let axis = axis.index(ndim);
            let mut dims = Vec::with_capacity(ndim - 1 + indices.ndim());
            let mut indices = Some(indices);
            for (i, dim) in input.iter().copied().enumerate() {
                if i == axis {
                    dims.extend(&(indices.take().unwrap())[..]);
                } else {
                    dims.push(dim);
                }
            }
            res.push(ResolvedTensorType::new(
                inputs[0].elem_type,
                ResolvedTensorDims::new(&dims),
            ));
        }

        Operator::Unsqueeze(Unsqueeze { axes }) => {
            let input = &inputs[0];
            let input_dims = &input.dims;
            let expanded_rank = input_dims.ndim() + axes.len();
            let mut insert = vec![false; expanded_rank];
            for &axis in axes.iter() {
                let axis = axis.index(expanded_rank);
                assert!(axis < expanded_rank);
                insert[axis] = true;
            }
            let mut dims = Vec::with_capacity(expanded_rank);
            let mut strides = Vec::with_capacity(expanded_rank);
            let mut input_iter = input_dims.iter().copied();
            let mut stride_iter = input.strides().iter().copied();
            for i in 0..expanded_rank {
                if insert[i] {
                    dims.push(1);
                    strides.push(0);
                } else {
                    dims.push(input_iter.next().ok_or(TypeError::InferError(
                        "Unsqueeze: Not enough dimensions".to_string(),
                    ))?);
                    strides.push(stride_iter.next().ok_or(TypeError::InferError(
                        "Unsqueeze: Not enough strides".to_string(),
                    ))?);
                }
            }

            assert!(input_iter.next().is_none());

            let ty = if matches!(mode, UnifyMode::CheckStrides) {
                ResolvedTensorType::with_stride(
                    input.elem_type,
                    ResolvedTensorDims::new(&dims),
                    ResolvedTensorDims::new(&strides),
                )
            } else {
                ResolvedTensorType::new(input.elem_type, ResolvedTensorDims::new(&dims))
            };
            res.push(ty);
        }

        Operator::ConstantOfShape(ConstantOfShape { value }) => {
            let shape = graph
                .initializer
                .get(&node.inputs[0].unwrap())
                .ok_or(TypeError::UnresolvedInput)?;
            let ty = match shape.data {
                TensorData::SInt(SIntType::I64, ref v) => {
                    assert_eq!(shape.dims.ndim(), 1);
                    let dims = v
                        .iter()
                        .map(|&x| {
                            assert!(x != 0);
                            x as usize
                        })
                        .collect::<Vec<_>>();
                    ResolvedTensorType::new(value.elem_type(), ResolvedTensorDims::new(&dims))
                }
                _ => {
                    return Err(TypeError::InferError("Invalid shape".to_string()));
                }
            };
            res.push(ty);
        }

        Operator::OneHot(OneHot {
            axis,
            depth,
            on_value,
            off_value,
        }) => {
            let indices = &inputs[args::ONEHOT_INDICES];

            let axis = if *axis < 0 {
                axis + indices.dims.ndim() as isize + 1
            } else {
                *axis
            };
            let axis = axis as usize;

            // TODO: Support non-contiguous indices.
            assert!(indices.is_contiguous());
            // TODO: Support non-innermost axis.
            assert_eq!(axis, indices.dims.ndim());

            let depth = depth.ok_or(TypeError::InferError("Unresolved depth".to_string()))?;

            let on_value =
                on_value.ok_or(TypeError::InferError("Unresolved on_value".to_string()))?;
            let off_value =
                off_value.ok_or(TypeError::InferError("Unresolved off_value".to_string()))?;
            assert_eq!(on_value.elem_type(), off_value.elem_type());
            let elem_ty = on_value.elem_type();

            let mut dims = indices.dims.clone();
            dims.push(depth);
            res.push(ResolvedTensorType::new(elem_ty, dims));
        }

        Operator::Constant(Constant { ref value }) => {
            res.push(value.tensor_type());
        }

        Operator::NonZero => {
            let input = &graph
                .initializer
                .get(&node.inputs[0].unwrap())
                .ok_or(TypeError::UnresolvedInput)?
                .data;

            macro_rules! count_nonzero {
                ($data: expr) => {{
                    $data.iter().filter(|x| !x.is_zero()).count()
                }};
            }

            let count = match input {
                TensorData::SInt(_, ref v) => count_nonzero!(v),
                TensorData::UInt(_, ref v) => count_nonzero!(v),
                TensorData::Float(_, ref v) => count_nonzero!(v),
            };
            let dims = inputs[0].dims.ndim();
            let dims = vec![dims, count];
            res.push(ResolvedTensorType::new(
                SIntType::I64.into(),
                ResolvedTensorDims::new(&dims),
            ));
        }

        Operator::NHWC2NCHW => {
            let data = &inputs[0];
            let t = Transpose {
                perm: Some(vec![0, 3, 1, 2]),
            };
            let ty = transpose(data, &t, mode)?;
            res.push(ty.contiguous());
        }

        // Custom
        Operator::Contiguous(_) => {
            let input = &inputs[0];
            res.push(input.contiguous());
        }
        Operator::Reinterpret(Reinterpret { ref ops }) => {
            let mut ty = inputs[0].clone();
            for op in ops.iter() {
                ty = match op {
                    ReinterpretType::Reshape { ref after, .. } => {
                        let shape: Vec<i64> = after.iter().map(|d| *d as i64).collect();
                        reshape(&ty, &shape, mode)?
                    }
                    ReinterpretType::Transpose(ref perm) => transpose(&ty, perm, mode)?,
                    ReinterpretType::Broadcast { ref after, .. } => {
                        let dims = ResolvedTensorDims::new(after);
                        ResolvedTensorType::new(ty.elem_type, dims)
                    }
                };
            }
            res.push(ty);
        }
        Operator::Input(_) | Operator::Output(_) | Operator::ReduceMatrix(_) => {
            for output in node.outputs.iter() {
                let ty = graph.get_resolved_tensor_type(*output);
                assert!(ty.is_some());
                res.push(ty.unwrap().clone());
            }
        }
    }
    Ok(res)
}

pub fn create_infer_passes(opt: &Options) -> SimplePassManager<SimpleGraphOp> {
    let mut manager = SimplePassManager::new("Shape inference".to_string());
    let target = opt.target;
    manager.add_pass(Box::new(ShapeInference { target }));
    manager.add_pass(Box::new(ShapeVerification {
        target,
        check_strides: false,
    }));
    manager.add_pass(Box::new(EarlyBroadcast::default()));
    manager
}
