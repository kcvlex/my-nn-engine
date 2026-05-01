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

pub fn fold_constant(graph: &mut Graph, node_id: NodeId) -> Option<Vec<Tensor>> {
    let op = graph.nodes[node_id].op.clone();
    let inputs = graph.nodes[node_id].inputs.clone();
    let outputs = graph.nodes[node_id].outputs.clone();
    match &op {
        op @ (Operator::Add | Operator::Mul | Operator::Div | Operator::Sub) => {
            let left = graph.get_inline_initializer(inputs[0].unwrap())?.clone();
            let right = graph.get_inline_initializer(inputs[1].unwrap())?.clone();

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
            let Tensor { data, dims } = graph.get_inline_initializer(inputs[0].unwrap())?.clone();
            let data = &data;
            let dims = &dims;
            macro_rules! cast {
                ($data: expr, $to: expr) => {{
                    match $to {
                        DataType::Bool => TensorData::Bool(
                            $data
                                .to_vec()
                                .iter()
                                .map(|x| if *x as i64 != 0 { 1u8 } else { 0u8 })
                                .collect_vec(),
                        ),
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
                TensorData::Bool(data) => {
                    let as_i64: Vec<i64> = data.iter().map(|&b| b as i64).collect();
                    cast!(as_i64, to)
                }
                TensorData::SInt(_, data) => cast!(data, to),
                TensorData::UInt(_, data) => cast!(data, to),
                TensorData::Float(_, data) => cast!(data, to),
            };
            let tensor = Tensor::new(dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Constant(Constant { ref value }) => Some(vec![value.clone()]),
        Operator::ConstantOfShape(ConstantOfShape { ref value }) => {
            let dims = graph.get_initializer(inputs[0].unwrap())?;
            let dims = match &dims.data {
                TensorData::SInt(SIntType::I64, data) => {
                    Some(data.iter().map(|x| *x as usize).collect_vec())
                }
                _ => None,
            }?;
            let dims = ResolvedTensorDims::from(&dims[..]);
            let data = value.to_tensor_data(dims.size().max(1));
            let tensor = Tensor::new(dims, data).ok()?;
            Some(vec![tensor])
        }
        Operator::Concat(Concat { ref axis }) => {
            let mut tensors: Vec<Tensor> = Vec::new();
            for input in inputs.iter() {
                tensors.push(graph.get_inline_initializer(input.unwrap())?.clone());
            }
            let axis = axis.index(tensors[0].dims.ndim());
            let refs: Vec<&Tensor> = tensors.iter().collect();
            Tensor::concat(&refs, axis).map(|x| vec![x]).ok()
        }
        Operator::NonZero => {
            let input = inputs[0].unwrap();
            let input = graph.get_inline_initializer(input)?;
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
                TensorData::Bool(data) => {
                    let as_i64: Vec<i64> = data.iter().map(|&b| b as i64).collect();
                    calc(&as_i64, dims)
                }
                TensorData::SInt(_, data) => calc(data, dims),
                TensorData::UInt(_, data) => calc(data, dims),
                TensorData::Float(_, data) => calc(data, dims),
            };
            let shape = ResolvedTensorDims::new(&[indices.len(), indices[0].len()]);
            let indices = indices.into_iter().flatten().collect_vec();
            let tensor = Tensor::new(shape, TensorData::SInt(SIntType::I64, indices)).ok()?;
            Some(vec![tensor])
        }
        Operator::Range => {
            let start_s = graph
                .get_initializer(inputs[0].unwrap())?
                .data
                .to_scalar_data()?;
            let limit_s = graph
                .get_initializer(inputs[1].unwrap())?
                .data
                .to_scalar_data()?;
            let delta_s = graph
                .get_initializer(inputs[2].unwrap())?
                .data
                .to_scalar_data()?;
            let (data, len) = match (&start_s, &limit_s, &delta_s) {
                (ScalarData::Float(ty, s), ScalarData::Float(_, l), ScalarData::Float(_, d)) => {
                    let n = ((*l - *s) / *d).ceil() as usize;
                    let v: Vec<f64> = (0..n).map(|i| *s + (i as f64) * *d).collect();
                    (TensorData::Float(*ty, v), n)
                }
                (ScalarData::SInt(ty, s), ScalarData::SInt(_, l), ScalarData::SInt(_, d)) => {
                    let diff = *l - *s;
                    let n = ((diff + *d - diff.signum()) / *d) as usize;
                    let v: Vec<i64> = (0..n).map(|i| *s + (i as i64) * *d).collect();
                    (TensorData::SInt(*ty, v), n)
                }
                _ => return None,
            };
            let dims = ResolvedTensorDims::new(&[len]);
            let tensor = Tensor::new(dims, data).ok()?;
            Some(vec![tensor])
        }
        Operator::Reciprocal => {
            let v = graph.get_initializer(inputs[0].unwrap())?;
            let data = match &v.data {
                TensorData::Float(ty, data) => {
                    TensorData::Float(*ty, data.iter().map(|x| 1.0 / *x).collect_vec())
                }
                _ => return None,
            };
            let tensor = Tensor::new(v.dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Neg => {
            let v = graph.get_initializer(inputs[0].unwrap())?;
            let data = match &v.data {
                TensorData::Float(ty, data) => {
                    TensorData::Float(*ty, data.iter().map(|x| -*x).collect_vec())
                }
                TensorData::SInt(ty, data) => {
                    TensorData::SInt(*ty, data.iter().map(|x| -*x).collect_vec())
                }
                _ => return None,
            };
            let tensor = Tensor::new(v.dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Sin => {
            let v = graph.get_initializer(inputs[0].unwrap())?;
            let data = match &v.data {
                TensorData::Float(ty, data) => {
                    TensorData::Float(*ty, data.iter().map(|x| x.sin()).collect_vec())
                }
                _ => return None,
            };
            let tensor = Tensor::new(v.dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Cos => {
            let v = graph.get_initializer(inputs[0].unwrap())?;
            let data = match &v.data {
                TensorData::Float(ty, data) => {
                    TensorData::Float(*ty, data.iter().map(|x| x.cos()).collect_vec())
                }
                _ => return None,
            };
            let tensor = Tensor::new(v.dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
        Operator::Expand => {
            let shape = graph.get_initializer(inputs[1].unwrap())?.clone();
            let target = match &shape.data {
                TensorData::SInt(SIntType::I64, v) => {
                    ResolvedTensorDims::new(&v.iter().map(|&x| x as usize).collect_vec())
                }
                _ => return None,
            };
            let input = graph.get_initializer(inputs[0].unwrap())?;
            let dims = broadcast_shape(&input.dims, &target).ok()?;
            Some(vec![input.broadcast(&dims)])
        }
        Operator::Flatten(Flatten { axis }) => {
            let input = graph.get_initializer(inputs[0].unwrap())?;
            let axis = axis.index(input.dims.ndim());
            let prefix: usize = input.dims.iter().take(axis).product();
            let suffix: usize = input.dims.iter().skip(axis).product();
            let dims = ResolvedTensorDims::new(&[prefix, suffix]);
            Some(vec![input.reshape(&dims)])
        }
        Operator::Where => {
            let cond = graph
                .get_initializer(inputs[args::WHERE_COND].unwrap())?
                .clone();
            let x = graph
                .get_initializer(inputs[args::WHERE_X].unwrap())?
                .clone();
            let y = graph
                .get_initializer(inputs[args::WHERE_Y].unwrap())?
                .clone();
            let ty = broadcast_shape(&cond.dims, &broadcast_shape(&x.dims, &y.dims).ok()?).ok()?;
            let cond = cond.broadcast(&ty);
            let x = x.broadcast(&ty);
            let y = y.broadcast(&ty);
            let TensorData::Bool(ref cond_data) = cond.data else {
                return None;
            };
            macro_rules! pattern {
                ($ctor: expr, $xv: expr, $yv: expr, $t: expr) => {{
                    let data = izip!(cond_data.iter(), $xv.iter(), $yv.iter())
                        .map(|(c, xv, yv)| if *c != 0 { *xv } else { *yv })
                        .collect_vec();
                    Some($ctor(*$t, data))
                }};
            }
            let data = match (&x.data, &y.data) {
                (TensorData::SInt(t, xv), TensorData::SInt(_, yv)) => {
                    pattern!(TensorData::SInt, xv, yv, t)
                }
                (TensorData::UInt(t, xv), TensorData::UInt(_, yv)) => {
                    pattern!(TensorData::UInt, xv, yv, t)
                }
                (TensorData::Float(t, xv), TensorData::Float(_, yv)) => {
                    pattern!(TensorData::Float, xv, yv, t)
                }
                _ => None,
            }?;
            let tensor = Tensor::new(ty, data).ok()?;
            Some(vec![tensor])
        }
        Operator::Equal => {
            let a = graph
                .get_initializer(inputs[args::EQUAL_A].unwrap())?
                .clone();
            let b = graph
                .get_initializer(inputs[args::EQUAL_B].unwrap())?
                .clone();
            let ty = broadcast_shape(&a.dims, &b.dims).ok()?;
            let a = a.broadcast(&ty);
            let b = b.broadcast(&ty);
            macro_rules! cmp {
                ($av: expr, $bv: expr) => {
                    izip!($av.iter(), $bv.iter())
                        .map(|(a, b)| if a == b { 1u8 } else { 0u8 })
                        .collect_vec()
                };
            }
            let data = match (&a.data, &b.data) {
                (TensorData::SInt(_, av), TensorData::SInt(_, bv)) => cmp!(av, bv),
                (TensorData::UInt(_, av), TensorData::UInt(_, bv)) => cmp!(av, bv),
                (TensorData::Float(_, av), TensorData::Float(_, bv)) => cmp!(av, bv),
                (TensorData::Bool(av), TensorData::Bool(bv)) => cmp!(av, bv),
                _ => return None,
            };
            let tensor = Tensor::new(ty, TensorData::Bool(data)).ok()?;
            Some(vec![tensor])
        }
        Operator::Shape(Shape { ref start, ref end }) => {
            let input = inputs[0].unwrap();
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
            let input_id = inputs[0].unwrap();
            let slices = Slice::collect_slices(graph, node_id)?;
            let input = graph.get_inline_initializer(input_id)?;
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
            let input = inputs[0].unwrap();
            let input = graph.get_inline_initializer(input)?;
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
            let indices = graph.get_inline_initializer(inputs[1].unwrap())?.clone();
            let input = graph.get_inline_initializer(inputs[0].unwrap())?;
            let axis = axis.index(input.dims.ndim());
            Some(vec![input.gather(&indices, axis)])
        }
        Operator::Reshape | Operator::Squeeze(_) | Operator::Unsqueeze(_) => {
            let dims = graph.get_resolved_tensor_type(outputs[0])?.dims.clone();
            let input = graph.get_inline_initializer(inputs[0].unwrap())?;
            Some(vec![input.reshape(&dims)])
        }
        Operator::Transpose(Transpose { ref perm }) => {
            let input = graph.get_inline_initializer(inputs[0].unwrap())?;
            let perm = perm.as_ref()?;
            Some(vec![input.transpose(perm)])
        }
        Operator::Contiguous(Contiguous { ref ops }) => {
            let mut tensor = graph.get_inline_initializer(inputs[0].unwrap())?.clone();
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
            let depth_id = graph.nodes[node_id]
                .inputs
                .get(args::ONEHOT_DEPTH)
                .and_then(|x| *x);
            let values_id = graph.nodes[node_id]
                .inputs
                .get(args::ONEHOT_VALUES)
                .and_then(|x| *x);

            let depth = depth_id
                .and_then(|id| graph.get_initializer(id).map(|t| t.clone()))
                .map(|tensor| {
                    let tensor = tensor
                        .data
                        .to_scalar_data()
                        .expect("OneHot 'depth' must be a scalar.");
                    match tensor {
                        ScalarData::Bool(v) => v as usize,
                        ScalarData::SInt(_, v) => v as usize,
                        ScalarData::UInt(_, v) => v as usize,
                        ScalarData::Float(_, v) => v as usize,
                    }
                });

            let values = values_id
                .and_then(|id| graph.get_initializer(id).map(|t| t.clone()))
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

        Operator::Clip(_) => {
            let extract_scalar_f64 = |graph: &mut Graph, idx: usize| -> Option<f64> {
                let id = graph.nodes[node_id].inputs.get(idx).and_then(|x| *x)?;
                let tensor = graph.get_initializer(id)?;
                tensor.data.to_scalar_data().map(|s| match s {
                    ScalarData::Bool(v) => v as f64,
                    ScalarData::SInt(_, v) => v as f64,
                    ScalarData::UInt(_, v) => v as f64,
                    ScalarData::Float(_, v) => v,
                })
            };

            let min_val = extract_scalar_f64(graph, args::CLIP_MIN);
            let max_val = extract_scalar_f64(graph, args::CLIP_MAX);

            let Operator::Clip(clip) = &mut graph.nodes[node_id].op else {
                unreachable!();
            };

            let mut drop_min = false;
            let mut drop_max = false;

            if clip.min.is_none() && min_val.is_some() {
                clip.min = min_val;
                drop_min = true;
            }
            if clip.max.is_none() && max_val.is_some() {
                clip.max = max_val;
                drop_max = true;
            }

            for (arg, drop) in [(args::CLIP_MIN, drop_min), (args::CLIP_MAX, drop_max)] {
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
            let scales_id = graph.nodes[node_id]
                .inputs
                .get(args::RESIZE_SCALES)
                .and_then(|x| *x);
            let sizes_id = graph.nodes[node_id]
                .inputs
                .get(args::RESIZE_SIZES)
                .and_then(|x| *x);
            let scales = scales_id
                .and_then(|id| graph.get_initializer(id).map(|t| t.clone()))
                .and_then(|tensor| tensor.to_1d_floats());
            let sizes = sizes_id
                .and_then(|id| graph.get_initializer(id).map(|t| t.clone()))
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
