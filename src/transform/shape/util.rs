use crate::onnx::model::{Graph, NodeId, UnifyMode};
use crate::onnx::operator::*;
use crate::tensor::{
    data::TensorData,
    dimensions::{broadcast_shape, ResolvedTensorDims},
    types::{ResolvedTensorType, SIntType, TensorType, TypeError},
};

#[derive(Debug)]
struct ConvShape<'a> {
    kernel_shape: &'a [usize],
    input_shape: &'a [usize],
    pad: &'a OptionalVec<(usize, usize)>,
    dilations: &'a OptionalVec<usize>,
}

impl ConvShape<'_> {
    fn padded_input_size(&self, i: usize) -> usize {
        self.input_shape[i] + self.pad[i].0 + self.pad[i].1
    }

    // Number of elements between first and last element (inclusive)
    fn distance_per_conv(&self, i: usize) -> usize {
        self.dilations[i] * (self.kernel_shape[i] - 1) + 1
    }
}

pub fn infer_node_output(
    graph: &Graph,
    node_id: NodeId,
    mode: UnifyMode,
) -> Result<Vec<ResolvedTensorType>, TypeError> {
    macro_rules! cond_error {
        ($cond: expr) => {{
            if $cond {
                return Err(TypeError::InferError(stringify!($cond).to_string()));
            }
        }};
    }

    let node = &graph.nodes[node_id];

    // dbg!(node);

    let inputs: Vec<&ResolvedTensorType> = node
        .inputs
        .iter()
        .flat_map(|&id| {
            graph.values[id].ty.as_ref().map(|x| match x {
                TensorType::Resolved(x) => Some(x),
                TensorType::Unresolved(_) => None,
            })
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(TypeError::UnresolvedInput)?;

    let mut res: Vec<ResolvedTensorType> = Vec::new();
    match &node.op {
        Operator::Add | Operator::Sub | Operator::Mul => {
            let a = &inputs[0];
            let b = &inputs[1];

            cond_error!(a.elem_type != b.elem_type);
            let dims = broadcast_shape(&a.dims, &b.dims)?;
            res.push(ResolvedTensorType::new(a.elem_type, dims));
        }
        Operator::BatchNormalization(_) |
        Operator::Exp |
        Operator::Identity |
        Operator::LeakyReLU(_) |
        Operator::Log |
        Operator::ReLU |
        Operator::Sigmoid |
        Operator::Tanh => {
            res.push(inputs[0].clone());
        }
        Operator::Cast(Cast { to }) => {
            // TODO: Check for element type
            let input = &inputs[0];
            res.push(ResolvedTensorType::new(*to, input.dims.clone()));
        }
        Operator::Transpose(ref transpose) => {
            let data = &inputs[args::TRANSPOSE_DATA];
            let perm = transpose
                .perm(data.dims.ndim())
                .ok_or(TypeError::InferError("Invalid permutation".to_string()))?;
            res.push(data.transpose(perm.as_slice()));
        }
        Operator::Reshape => {
            let a = &inputs[args::RESHAPE_DATA];
            let shape = node.inputs[args::RESHAPE_SHAPE];
            let shape = &graph
                .initializer
                .get(&shape)
                .ok_or(TypeError::UnresolvedInput)?
                .data;
            let shape = match shape {
                TensorData::SInt(_, ref v) => {
                    let prod0 = a.dims.size();
                    let prod1 = v
                        .iter()
                        .copied()
                        .enumerate()
                        .filter(|(_, x)| *x != -1)
                        .map(|(i, x)| if x == 0 { a.dims[i] } else { x as usize })
                        .product::<usize>();
                    Ok(ResolvedTensorDims::new(
                        v.iter()
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
                            .collect(),
                    ))
                }
                _ => Err(TypeError::InferError("Invalid shape".to_string())),
            }?;

            cond_error!(a.dims.size() != shape.size());
            let reshaped = if matches!(mode, UnifyMode::CheckStrides) {
                a.try_reshape(&shape)
                    .ok_or(TypeError::InferError("Unsupported reshape".to_string()))?
            } else {
                ResolvedTensorType::new(a.elem_type, shape.clone())
            };
            res.push(reshaped);
        }
        Operator::Resize(resize) => {
            let dims = resize
                .resized_shape(graph, node_id)
                .ok_or(TypeError::InferError("Invalid resized shape".to_string()))?;
            res.push(ResolvedTensorType::new(inputs[0].elem_type, dims));
        }
        Operator::Conv(Conv {
            pad,
            dilations,
            groups,
            strides,
            ..
        }) => {
            let x = &inputs[args::CONV_DATA];
            let w = &inputs[args::CONV_WEIGHT];
            // TODO: Check bias
            // TODO: Check kernel_shape

            let batch_size = x.dims[0];
            let channels = x.dims[1];
            let ndim = x.dims.ndim() - 2;
            let input = &x.dims[2..];

            let kernel_shape = &w.dims;
            let feature_map_size = kernel_shape[0];

            cond_error!(feature_map_size % *groups != 0);
            cond_error!(channels != kernel_shape[1] * *groups);
            cond_error!(x.dims.ndim() != kernel_shape.ndim());
            let kernel_shape = &kernel_shape[2..];
            let default_pad = OptionalVec::new(None, (0, 0));
            let pad = match pad {
                ConvPad::NotSet(ref pad) => Some(pad),
                ConvPad::SameUpper | ConvPad::SameLower => None,
                ConvPad::Valid => Some(&default_pad),
            };

            let mut dims = Vec::with_capacity(ndim + 2);
            dims.push(batch_size);
            dims.push(feature_map_size);
            let conv_shape = pad.map(|pad| ConvShape {
                kernel_shape,
                input_shape: input,
                pad,
                dilations,
            });
            for i in 0..ndim {
                let stride = strides[i];
                let dim = if let Some(ref conv_shape) = &conv_shape {
                    let (q, rem) = num_integer::div_rem(
                        conv_shape.padded_input_size(i) - conv_shape.distance_per_conv(i),
                        stride,
                    );
                    // if rem != 0 {
                    //     println!("node.name={:?}", node.name);
                    //     return Err(TypeError::InferError("rem must be 0".to_string()));
                    // }
                    q + 1
                } else {
                    input[i].div_ceil(stride)
                };
                dims.push(dim);
            }
            res.push(ResolvedTensorType::new(
                x.elem_type,
                ResolvedTensorDims::new(dims),
            ));
        }
        Operator::MatMul => {
            let a = &inputs[args::MATMUL_LHS];
            let b = &inputs[args::MATMUL_RHS];

            cond_error!(a.elem_type != b.elem_type);

            let (ldim, l_prepended) = if a.dims.ndim() == 1 {
                (ResolvedTensorDims::new(vec![1, a.dims[0]]), true)
            } else {
                (a.dims.clone(), false)
            };
            let (rdim, r_prepended) = if b.dims.ndim() == 1 {
                (ResolvedTensorDims::new(vec![b.dims[0], 1]), true)
            } else {
                (b.dims.clone(), false)
            };

            let l_prefix = ldim.prefix(ldim.ndim() - 2);
            let r_prefix = rdim.prefix(rdim.ndim() - 2);
            let prefix = broadcast_shape(&l_prefix, &r_prefix)?;

            let l_suffix = ldim.suffix(2);
            let r_suffix = rdim.suffix(2);
            cond_error!(l_suffix[1] != r_suffix[0]);

            let mut output = prefix;
            if !l_prepended {
                output.push(l_suffix[0]);
            }
            if !r_prepended {
                output.push(r_suffix[1]);
            }
            res.push(ResolvedTensorType::new(a.elem_type, output));
        }
        Operator::MaxPool(Pooling {
            kernel_shape,
            pad,
            ceil_mode,
            dilations,
            strides,
        }) => {
            let x = &inputs[args::MAXPOOL_DATA];
            let mut dims = Vec::with_capacity(x.dims.ndim());
            dims.push(x.dims[0]);
            dims.push(x.dims[1]);
            let input = &x.dims[2..];
            let default_pad = OptionalVec::new(None, (0, 0));
            let pad = match pad {
                ConvPad::NotSet(ref pad) => pad,
                ConvPad::Valid => &default_pad,
                ConvPad::SameUpper | ConvPad::SameLower => {
                    return Err(TypeError::InferError(
                        "Deprecated attribution (not supported)".to_string(),
                    ))
                }
            };
            let conv_shape = ConvShape {
                kernel_shape: &kernel_shape[..],
                input_shape: input,
                pad,
                dilations,
            };
            for i in 0..input.len() {
                let stride = strides[i];
                let num = conv_shape.padded_input_size(i) - conv_shape.distance_per_conv(i);
                let dim = if !*ceil_mode {
                    // Floor div
                    num / stride + 1
                } else {
                    // TODO: Correct?
                    //
                    // https://onnx.ai/onnx/operators/onnx__MaxPool.html#summary
                    //  > Sliding windows that would start in the right padded region are ignored.
                    num.div_ceil(stride) + 1
                };
                dims.push(dim);
            }
            res.push(ResolvedTensorType::new(
                x.elem_type,
                ResolvedTensorDims::new(dims),
            ));
        }
        Operator::GlobalAveragePool => {
            let x = &inputs[0];
            let dims = x.dims[..2].to_vec();
            res.push(ResolvedTensorType::new(
                x.elem_type,
                ResolvedTensorDims::new(dims),
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
            let dims = [m, n];
            res.push(ResolvedTensorType::new(
                a.elem_type,
                ResolvedTensorDims::new(dims.to_vec()),
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
                ResolvedTensorDims::new(dims),
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
                ResolvedTensorDims::new(vec![end - start]),
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
                    // Use the same strides as the input tensor
                    let mut ty = input.clone();
                    ty.dims[axis] = x;
                    res.push(ty);
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
                    cond_error!(axis >= rank);
                    cond_error!(input_dims[axis] != 1);
                    drop[axis] = true;
                }
                drop
            } else {
                (0..rank).map(|i| input_dims[i] == 1).collect()
            };

            let mut dims = Vec::new();
            for (i, d) in input_dims.iter().copied().enumerate() {
                if !drop[i] {
                    dims.push(d);
                }
            }

            res.push(ResolvedTensorType::new(
                input.elem_type,
                ResolvedTensorDims::new(dims),
            ));
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
                ResolvedTensorDims::new(dims),
            ));
        }

        Operator::Unsqueeze(Unsqueeze { axes }) => {
            let input = &inputs[0];
            let input_dims = &input.dims;
            let expanded_rank = input_dims.ndim() + axes.len();
            let mut insert = vec![false; expanded_rank];
            for &axis in axes.iter() {
                let axis = axis.index(expanded_rank);
                cond_error!(axis >= expanded_rank);
                insert[axis] = true;
            }
            let mut dims = Vec::with_capacity(expanded_rank);
            let mut input_iter = input_dims.iter().copied();
            for i in 0..expanded_rank {
                if insert[i] {
                    dims.push(1);
                } else {
                    dims.push(input_iter.next().ok_or(TypeError::InferError(
                        "Unsqueeze: Not enough dimensions".to_string(),
                    ))?);
                }
            }

            cond_error!(input_iter.next().is_some());
            res.push(ResolvedTensorType::new(
                input.elem_type,
                ResolvedTensorDims::new(dims),
            ));
        }

        // Custom
        Operator::Contiguous => {
            let input = &inputs[0];
            res.push(input.contiguous());
        }
        Operator::Input(_) |
        Operator::Output(_) |
        Operator::Im2Col(_) |
        Operator::ReduceMatrix(_) |
        Operator::ElementwiseOps(_) |
        Operator::ForceReshape => {
            for output in node.outputs.iter() {
                let ty = graph.get_resolved_tensor_type(*output);
                cond_error!(ty.is_none());
                res.push(ty.unwrap().clone());
            }
        }
    }
    Ok(res)
}
