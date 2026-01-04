use itertools::izip;
use itertools::Itertools;
use num::Zero;

use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::*;
use crate::tensor::data::TensorData;
use crate::tensor::dimensions::ResolvedTensorDims;
use crate::tensor::types::DataType;
use crate::tensor::types::SIntType;
use crate::tensor::Tensor;

fn all_slice_indices(dims: &ResolvedTensorDims) -> (Vec<isize>, Vec<isize>) {
    let starts = vec![0; dims.ndim()];
    let ends = dims[..].iter().map(|x| *x as isize).collect();
    (starts, ends)
}

pub fn fold_constant(graph: &Graph, node_id: NodeId) -> Option<Vec<Tensor>> {
    let node = &graph.nodes[node_id];
    match &node.op {
        op @ (Operator::Add | Operator::Mul | Operator::Div | Operator::Sub) => {
            let left = graph.initializer.get(&node.inputs[0])?;
            let right = graph.initializer.get(&node.inputs[1])?;

            // TODO: Broadcast.
            if left.tensor_type() != right.tensor_type() {
                return None;
            }

            macro_rules! calc {
                ($lhs: expr, $rhs: expr) => {{
                    match op {
                        Operator::Add => $lhs + $rhs,
                        Operator::Mul => $lhs * $rhs,
                        Operator::Div => $lhs / $rhs,
                        Operator::Sub => $lhs - $rhs,
                        _ => unreachable!(),
                    }
                }};
            }

            macro_rules! pattern {
                ($ctor: expr, $lhs: expr, $rhs: expr, $ty: expr) => {{
                    let data = izip!($lhs.iter(), $rhs.iter())
                        .map(|(l, r)| calc!(l, r))
                        .collect_vec();
                    Some($ctor(*$ty, data))
                }};
            }

            let tensor = match (&left.data, &right.data) {
                (TensorData::SInt(ty, lhs), TensorData::SInt(_, rhs)) => {
                    pattern!(TensorData::SInt, lhs, rhs, ty)
                }
                (TensorData::UInt(ty, lhs), TensorData::UInt(_, rhs)) => {
                    pattern!(TensorData::UInt, lhs, rhs, ty)
                }
                (TensorData::Float(ty, lhs), TensorData::Float(_, rhs)) => {
                    pattern!(TensorData::Float, lhs, rhs, ty)
                }
                _ => None,
            }?;
            let tensor = Tensor::new(left.dims.clone(), tensor).ok()?;
            Some(vec![tensor])
        }
        Operator::Cast(Cast { ref to }) => {
            let Tensor { data, dims } = &graph.initializer.get(&node.inputs[0])?;
            macro_rules! cast {
                ($data: expr, $to: expr) => {{
                    match $to {
                        DataType::SInt(to) => TensorData::SInt(
                            *to,
                            $data.to_vec().iter().map(|x| *x as i64).collect_vec(),
                        ),
                        DataType::UInt(to) => TensorData::UInt(
                            *to,
                            $data.to_vec().iter().map(|x| *x as u64).collect_vec(),
                        ),
                        DataType::Float(to) => TensorData::Float(
                            *to,
                            $data.to_vec().iter().map(|x| *x as f64).collect_vec(),
                        ),
                    }
                }};
            }
            let data = match data {
                TensorData::SInt(_, data) => cast!(data, to),
                TensorData::UInt(_, data) => cast!(data, to),
                TensorData::Float(_, data) => cast!(data, to),
            };
            let tensor = Tensor::new(dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Constant(Constant { ref value }) => Some(vec![value.clone()]),
        Operator::ConstantOfShape(ConstantOfShape { ref value }) => {
            let dims = graph.initializer.get(&node.inputs[0])?;
            let dims = match &dims.data {
                TensorData::SInt(SIntType::I64, data) => {
                    Some(data.iter().map(|x| *x as usize).collect_vec())
                }
                _ => None,
            }?;
            let dims = ResolvedTensorDims::from(dims);
            let data = value.to_tensor_data(dims.size());
            let tensor = Tensor::new(dims, data).ok()?;
            Some(vec![tensor])
        }
        Operator::Concat(Concat { ref axis }) => {
            let tensors = node
                .inputs
                .iter()
                .map(|x| graph.initializer.get(x))
                .collect::<Option<Vec<_>>>()?;
            let axis = axis.index(tensors[0].dims.ndim());
            Tensor::concat(&tensors, axis).map(|x| vec![x]).ok()
        }
        Operator::NonZero => {
            let input = node.inputs[0];
            let input = &graph.initializer.get(&input)?;
            let dims = &input.dims;

            fn calc<T: Zero>(data: &[T], dims: &ResolvedTensorDims) -> Vec<Vec<i64>> {
                let mut indices: Vec<Vec<i64>> = vec![Vec::new(); dims.ndim()];
                for (i, val) in data.iter().enumerate() {
                    if val.is_zero() {
                        continue;
                    }

                    let mut cur = i;
                    for (dim, index) in izip!(dims.iter(), indices.iter_mut()).rev() {
                        index.push((cur % *dim) as i64);
                        cur /= *dim;
                    }
                }
                indices
            }

            let indices = match &input.data {
                TensorData::SInt(_, data) => calc(data, dims),
                TensorData::UInt(_, data) => calc(data, dims),
                TensorData::Float(_, data) => calc(data, dims),
            };
            let shape = ResolvedTensorDims::from(vec![indices.len(), indices[0].len()]);
            let indices = indices.into_iter().flatten().collect_vec();
            let tensor = Tensor::new(shape, TensorData::SInt(SIntType::I64, indices)).ok()?;
            Some(vec![tensor])
        }
        Operator::Shape(Shape { ref start, ref end }) => {
            let input = node.inputs[0];
            let input = &graph.get_resolved_tensor_type(input)?.dims;
            let ndim = input.ndim();
            let start = start.index(ndim);
            let end = match end {
                Some(index) => index.index(ndim),
                None => ndim,
            };
            let shape = input[start..end].iter().map(|x| *x as i64).collect();
            let shape = TensorData::SInt(SIntType::I64, shape);
            let shape = shape.into_1d_tensor();
            Some(vec![shape])
        }
        Operator::Slice => {
            let input = node.inputs[0];
            let input = &graph.initializer.get(&input)?;
            let slices = Slice::collect_slices(graph, node_id)?;
            let (mut starts, mut ends) = all_slice_indices(&input.dims);
            for Slice {
                start,
                end,
                axis,
                step,
            } in slices.iter()
            {
                if *step != 1 {
                    unimplemented!();
                }
                starts[*axis] = *start;
                ends[*axis] = *end;
            }
            Some(vec![input.slices(&starts, &ends)])
        }
        Operator::Split(ref split) => {
            let input = node.inputs[0];
            let input = &graph.initializer.get(&input)?;
            let (mut starts, mut ends) = all_slice_indices(&input.dims);
            let axis = split.axis.index(input.dims.ndim());
            let mut res = Vec::new();
            let mut cur = 0;
            for sdim in split.split(&input.dims)?.into_iter() {
                let end = cur + sdim;
                starts[axis] = cur as isize;
                ends[axis] = end as isize;
                res.push(input.slices(&starts, &ends));
                cur = end;
            }
            Some(res)
        }
        Operator::Gather(Gather { ref axis }) => {
            let input = &graph.initializer.get(&node.inputs[0])?;
            let indices = &graph.initializer.get(&node.inputs[1])?;
            let axis = axis.index(input.dims.ndim());
            Some(vec![input.gather(indices, axis)])
        }
        Operator::Squeeze(_) | Operator::Unsqueeze(_) => {
            let input = graph.initializer.get(&node.inputs[0])?;
            let dims = &graph.get_resolved_tensor_type(node.outputs[0])?.dims;
            Some(vec![input.reshape(dims)])
        }
        _ => None,
    }
}
