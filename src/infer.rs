use crate::model::{Graph, NodeId};
use crate::operator::*;
use crate::tensor::{
    resolved_dimensions::{broadcast_shape, ResolvedTensorDims},
    tensor::{ResolvedTensorType, TensorData, TensorType, TypeError},
};
use itertools::zip_eq;

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

impl Graph {
    fn infer_node_output(&mut self, node_id: NodeId) -> Result<Vec<ResolvedTensorType>, TypeError> {
        macro_rules! cond_error {
            ($cond: expr) => {{
                if $cond {
                    return Err(TypeError::InferError(stringify!($cond).to_string()));
                }
            }};
        }

        let node = &mut self.nodes[node_id];

        let inputs: Vec<&ResolvedTensorType> = node
            .inputs
            .iter()
            .flat_map(|&id| {
                self.values[id].ty.as_ref().map(|x| match x {
                    TensorType::Resolved(x) => Some(x),
                    TensorType::Unresolved(_) => None,
                })
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(TypeError::UnresolvedInput)?;

        let mut res: Vec<ResolvedTensorType> = Vec::new();
        match &mut node.op {
            Operator::Add => {
                let a = &inputs[args::ADD_LHS];
                let b = &inputs[args::ADD_RHS];

                cond_error!(a.elem_type != b.elem_type);
                let dims = broadcast_shape(&a.dims, &b.dims)?;
                res.push(ResolvedTensorType::new(a.elem_type, dims));
            }
            Operator::ReLU | Operator::Sigmoid => {
                res.push(inputs[0].clone());
            }
            Operator::Transpose(perms) => {
                let data = &inputs[args::TRANSPOSE_DATA];
                if perms.is_empty() {
                    *perms = (0..data.dims.ndim()).rev().collect();
                }
                let mut flags = vec![false; perms.len()];
                for p in perms.iter() {
                    if perms.len() <= *p || flags[*p] {
                        return Err(TypeError::InferError("Invalid permutation".to_string()));
                    }
                    flags[*p] = true;
                }
                res.push(data.transpose(perms.as_slice()));
            }
            Operator::Reshape => {
                let a = &inputs[args::RESHAPE_DATA];
                let shape = node.inputs[args::RESHAPE_SHAPE];
                let shape = self
                    .initializer
                    .get(&shape)
                    .ok_or(TypeError::UnresolvedInput)
                    .map(|x| match x.data {
                        TensorData::I64(ref v) => Ok(ResolvedTensorDims::new(
                            v.iter().map(|x| *x as usize).collect(),
                        )),
                        _ => Err(TypeError::InferError("Invalid shape".to_string())),
                    })??;

                cond_error!(a.dims.size() != shape.size());
                res.push(ResolvedTensorType::new(a.elem_type, shape));
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
                        if rem != 0 {
                            return Err(TypeError::InferError("rem must be 0".to_string()));
                        }
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

            Operator::Gemm(Gemm {
                trans_a,
                trans_b,
                trans_c,
                ..
            }) => {
                let a = &inputs[args::GEMM_A];
                let b = &inputs[args::GEMM_B];

                let m = a.dims[if !*trans_a { 0 } else { 1 }];
                let n = b.dims[if !*trans_b { 1 } else { 0 }];
                // let a_k = a.dims[1 - a_idx];
                // let b_k = b.dims[b_idx];
                // assert!(a_k == b_k);
                let mut dims = [m, n];
                if *trans_c {
                    dims.reverse();
                }
                res.push(ResolvedTensorType::new(
                    a.elem_type,
                    ResolvedTensorDims::new(dims.to_vec()),
                ));
            }

            // Custom
            Operator::Input(_)
            | Operator::Output(_)
            | Operator::Im2Col(_)
            | Operator::ReduceMatrix(_)
            | Operator::ForceReshape => {
                unreachable!()
            }
        }
        Ok(res)
    }

    pub fn infer(&mut self) -> Result<(), TypeError> {
        let ids = self
            .nodes
            .iter()
            .filter(|(_, node)| !node.is_dummy())
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        for id in ids {
            let types = self.infer_node_output(id)?;
            let node = &self.nodes[id];
            for (value_id, inferred) in zip_eq(node.outputs.iter(), types.into_iter()) {
                let cur_ty = &mut self.values[*value_id].ty;
                if let Some(TensorType::Resolved(cur_ty)) = cur_ty {
                    if *cur_ty != inferred {
                        return Err(TypeError::InferError(format!(
                            "Mismatched type:\n\tnode_name={:?}\n\texpected={:?}\n\tinferred={:?}",
                            node.name, cur_ty, inferred
                        )));
                    }
                } else {
                    *cur_ty = Some(inferred.into());
                }
            }
        }
        Ok(())
    }
}
