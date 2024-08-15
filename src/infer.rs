use crate::model::{Graph, Node};
use crate::operator::*;
use crate::tensor::{
    resolved_dimensions::{broadcast_shape, ResolvedTensorDims},
    tensor::{ResolvedTensorType, TensorData, TypeError},
};

impl Graph {
    fn infer_node_output(&self, node: &Node) -> Result<Vec<ResolvedTensorType>, TypeError> {
        macro_rules! cond_error {
            ($cond: expr) => {{
                if $cond {
                    return Err(TypeError::InferError(stringify!($cond).to_string()));
                }
            }};
        }

        let inputs = node
            .inputs
            .iter()
            .map(|&id| self.values[id].ty.clone().and_then(|x| x.to_resolved()))
            .collect::<Option<Vec<_>>>()
            .ok_or(TypeError::UnresolvedInput)?;

        // Number of elements between first and last element (inclusive)
        let elements_per_conv = |kernel: usize, dilation: usize| dilation * (kernel - 1) + 1;
        let padded_input = |input: usize, pad: (usize, usize)| input + pad.0 + pad.1;

        let mut res: Vec<ResolvedTensorType> = Vec::new();
        match &node.op {
            Operator::Add => {
                let a = &inputs[args::ADD_LHS];
                let b = &inputs[args::ADD_RHS];

                cond_error!(a.elem_type != b.elem_type);
                let dims = broadcast_shape(&a.dims, &b.dims)?;
                res.push(ResolvedTensorType::new(a.elem_type.clone(), dims));
            }
            Operator::ReLU => {
                let a = &inputs[args::RELU];
                res.push(a.clone());
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
                res.push(ResolvedTensorType::new(a.elem_type.clone(), shape));
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

                cond_error!(feature_map_size % groups != 0);
                cond_error!(channels != kernel_shape[1] * groups);
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
                for i in 0..ndim {
                    let stride = strides[i];
                    let dim = if let Some(pad) = &pad {
                        let dilation = dilations[i];
                        let pad = pad[i];

                        let (q, rem) = num_integer::div_rem(
                            padded_input(input[i], pad)
                                - elements_per_conv(kernel_shape[i], dilation),
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
                    x.elem_type.clone(),
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
                res.push(ResolvedTensorType::new(a.elem_type.clone(), output));
            }
            Operator::MaxPool(MaxPool {
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
                for i in 0..input.len() {
                    let stride = strides[i];
                    let dilation = dilations[i];
                    let kernel = kernel_shape[i];
                    let num = padded_input(input[i], pad[i]) - elements_per_conv(kernel, dilation);
                    let dim = if !ceil_mode {
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
                    x.elem_type.clone(),
                    ResolvedTensorDims::new(dims),
                ));
            }
        }
        Ok(res)
    }

    pub fn infer(&mut self) -> Result<(), TypeError> {
        for (_, node) in self.nodes.iter() {
            let types = self.infer_node_output(node)?;
            for i in 0..node.outputs.len() {
                let value_id = node.outputs[i];
                self.values[value_id].ty = Some(types[i].clone().into());
            }
        }
        Ok(())
    }
}
