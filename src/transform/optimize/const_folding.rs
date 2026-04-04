use itertools::izip;
use itertools::Itertools;
use num::Zero;

use crate::onnx::model::Graph;
use crate::onnx::model::NodeId;
use crate::onnx::operator::args;
use crate::onnx::operator::*;
use crate::onnx::utils::simple_topological_order;
use crate::tensor::data::ScalarData;
use crate::tensor::data::TensorData;
use crate::tensor::types::broadcast_shape;
use crate::tensor::types::DataType;
use crate::tensor::types::ResolvedTensorDims;
use crate::tensor::types::SIntType;
use crate::tensor::Tensor;
use crate::transform::modify::GraphOp;
use crate::transform::Pass;

fn all_slice_indices(dims: &ResolvedTensorDims) -> (Vec<isize>, Vec<isize>) {
    let starts = vec![0; dims.ndim()];
    let ends = dims[..].iter().map(|x| *x as isize).collect();
    (starts, ends)
}

pub fn fold_constant(graph: &Graph, node_id: NodeId) -> Option<Vec<Tensor>> {
    let node = &graph.nodes[node_id];
    match &node.op {
        op @ (Operator::Add | Operator::Mul | Operator::Div | Operator::Sub) => {
            let left = graph.initializer.get(&node.inputs[0].unwrap())?;
            let right = graph.initializer.get(&node.inputs[1].unwrap())?;

            let ty = broadcast_shape(&left.dims, &right.dims).ok()?;
            let left = left.broadcast(&ty);
            let right = right.broadcast(&ty);

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
            let Tensor { data, dims } = &graph.initializer.get(&node.inputs[0].unwrap())?;
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
            let dims = graph.initializer.get(&node.inputs[0].unwrap())?;
            let dims = match &dims.data {
                TensorData::SInt(SIntType::I64, data) => {
                    Some(data.iter().map(|x| *x as usize).collect_vec())
                }
                _ => None,
            }?;
            let dims = ResolvedTensorDims::from(&dims[..]);
            let data = value.to_tensor_data(dims.size());
            let tensor = Tensor::new(dims, data).ok()?;
            Some(vec![tensor])
        }
        Operator::Concat(Concat { ref axis }) => {
            let tensors = node
                .inputs
                .iter()
                .map(|x| graph.initializer.get(&x.unwrap()))
                .collect::<Option<Vec<_>>>()?;
            let axis = axis.index(tensors[0].dims.ndim());
            Tensor::concat(&tensors, axis).map(|x| vec![x]).ok()
        }
        Operator::NonZero => {
            let input = node.inputs[0].unwrap();
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
            let shape = ResolvedTensorDims::new(&[indices.len(), indices[0].len()]);
            let indices = indices.into_iter().flatten().collect_vec();
            let tensor = Tensor::new(shape, TensorData::SInt(SIntType::I64, indices)).ok()?;
            Some(vec![tensor])
        }
        Operator::Reciprocal => {
            let v = graph.initializer.get(&node.inputs[0].unwrap())?;
            let data = match &v.data {
                TensorData::Float(ty, data) => {
                    TensorData::Float(*ty, data.iter().map(|x| 1.0 / *x).collect_vec())
                }
                _ => return None,
            };
            let tensor = Tensor::new(v.dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Shape(Shape { ref start, ref end }) => {
            let input = node.inputs[0].unwrap();
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
            let input = node.inputs[0].unwrap();
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
            let input = node.inputs[0].unwrap();
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
            let input = &graph.initializer.get(&node.inputs[0].unwrap())?;
            let indices = &graph.initializer.get(&node.inputs[1].unwrap())?;
            let axis = axis.index(input.dims.ndim());
            Some(vec![input.gather(indices, axis)])
        }
        Operator::Reshape | Operator::Squeeze(_) | Operator::Unsqueeze(_) => {
            let input = graph.initializer.get(&node.inputs[0].unwrap())?;
            let dims = &graph.get_resolved_tensor_type(node.outputs[0])?.dims;
            Some(vec![input.reshape(dims)])
        }
        Operator::Transpose(Transpose { ref perm }) => {
            let input = graph.initializer.get(&node.inputs[0].unwrap())?;
            let perm = perm.as_ref()?;
            Some(vec![input.transpose(perm)])
        }
        Operator::Contiguous(Contiguous { ref ops }) => {
            let mut tensor = graph.initializer.get(&node.inputs[0].unwrap())?.clone();
            for op in ops {
                match op {
                    ReinterpretType::Reshape { after, .. } => {
                        tensor = tensor.reshape(&ResolvedTensorDims::new(after));
                    }
                    ReinterpretType::Transpose(Transpose { perm }) => {
                        let perm = perm.as_ref()?;
                        tensor = tensor.transpose(perm);
                    }
                    ReinterpretType::Broadcast { after, .. } => {
                        tensor = tensor.broadcast(&ResolvedTensorDims::new(after));
                    }
                }
            }
            Some(vec![tensor])
        }
        _ => None,
    }
}

// Propagate constant inputs "into" the given node.
pub fn prop_constant<T: GraphOp>(graph: &mut Graph, node_id: NodeId, modifier: &mut T) {
    match &graph.nodes[node_id].op {
        Operator::OneHot(_) => {
            let inputs = &graph.nodes[node_id].inputs;

            let depth = inputs
                .get(args::ONEHOT_DEPTH)
                .and_then(|id| id.as_ref())
                .and_then(|id| graph.initializer.get(id))
                .map(|tensor| {
                    let tensor = tensor
                        .data
                        .to_scalar_data()
                        .expect("OneHot 'depth' must be a scalar.");
                    match tensor {
                        ScalarData::SInt(_, v) => v as usize,
                        ScalarData::UInt(_, v) => v as usize,
                        ScalarData::Float(_, v) => v as usize,
                    }
                });

            let values = inputs
                .get(args::ONEHOT_VALUES)
                .and_then(|id| id.as_ref())
                .and_then(|id| graph.initializer.get(id))
                .map(|values| {
                    let [off_value, on_value] = values.data.to_scalars()[..] else {
                        panic!("OneHot 'values' input must contain exactly two scalar values.");
                    };
                    (off_value, on_value)
                });

            let Operator::OneHot(one_hot) = &mut graph.nodes[node_id].op else {
                unreachable!();
            };

            if one_hot.depth.is_none() {
                one_hot.depth = depth;
            }

            assert!(
                one_hot.off_value.is_none() && one_hot.on_value.is_none() ||
                    one_hot.off_value.is_some() && one_hot.on_value.is_some()
            );
            if one_hot.off_value.is_none() {
                one_hot.off_value = values.map(|(off, _)| off);
                one_hot.on_value = values.map(|(_, on)| on);
            }

            for (arg, drop) in [
                (args::ONEHOT_DEPTH, depth.is_some()),
                (args::ONEHOT_VALUES, values.is_some()),
            ] {
                if drop &&
                    graph.nodes[node_id]
                        .inputs
                        .get(arg)
                        .and_then(|x| x.as_ref())
                        .is_some()
                {
                    modifier.drop_node_input(graph, node_id, arg);
                }
            }
        }

        Operator::Resize(resize) => {
            if resize.scale.is_some() {
                return;
            }
            let scales = graph.nodes[node_id]
                .inputs
                .get(args::RESIZE_SCALES)
                .and_then(|id| id.as_ref())
                .and_then(|id| graph.initializer.get(id))
                .and_then(|tensor| tensor.to_1d_floats());

            let sizes = graph.nodes[node_id]
                .inputs
                .get(args::RESIZE_SIZES)
                .and_then(|id| id.as_ref())
                .and_then(|id| graph.initializer.get(id))
                .and_then(|tensor| tensor.to_1d_sints());

            let scale = match (scales, sizes) {
                (Some(scales), None) => ResizeScale::Scales(scales),
                (Some(scales), Some(sizes)) if scales.is_empty() => ResizeScale::Sizes(sizes),
                (None, Some(sizes)) => ResizeScale::Sizes(sizes),
                (Some(_), Some(_)) => unreachable!(),
                (None, None) => unimplemented!(),
            };

            let Operator::Resize(resize) = &mut graph.nodes[node_id].op else {
                unreachable!();
            };
            resize.scale = Some(scale);

            // TODO: Don't drop ROI.
            for arg in [args::RESIZE_ROI, args::RESIZE_SCALES, args::RESIZE_SIZES] {
                if graph.nodes[node_id]
                    .inputs
                    .get(arg)
                    .and_then(|x| x.as_ref())
                    .is_some()
                {
                    modifier.drop_node_input(graph, node_id, arg);
                }
            }
        }

        _ => (),
    }
}

#[derive(Default)]
pub struct ConstantFolding {
    pub check_strides: bool,
}

impl<T: GraphOp> Pass<T> for ConstantFolding {
    fn summary(&self) -> &'static str {
        "Constant Fold"
    }

    fn run(&self, graph: &mut Graph, modifier: &mut T) {
        let ids = simple_topological_order(graph);

        for id in ids {
            prop_constant(graph, id, modifier);

            if let Some(constants) = fold_constant(graph, id) {
                let outputs = graph.nodes[id].outputs.clone();
                for (old_value, tensor) in izip!(outputs.iter(), constants.into_iter()) {
                    let new_value = modifier.register_new_tensor(
                        graph,
                        tensor,
                        format!("folded_{}", graph.nodes[id].name),
                    );
                    if self.check_strides {
                        modifier.replace_input_value(graph, *old_value, new_value)
                    } else {
                        modifier.replace_input_value_if_without_typecheck(
                            graph,
                            *old_value,
                            new_value,
                            |_, _| true,
                        );
                    }
                }
            }
        }
    }
}
