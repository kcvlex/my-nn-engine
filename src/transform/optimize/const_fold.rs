use itertools::Itertools;

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
    match node.op {
        Operator::Cast(Cast { ref to }) => {
            let Tensor { data, dims } = &graph.initializer.get(&node.inputs[0])?;
            let data = match (data, to) {
                (TensorData::SInt(_, data), DataType::SInt(to)) => {
                    TensorData::SInt(*to, data.to_vec())
                }
                (TensorData::UInt(_, data), DataType::UInt(to)) => {
                    TensorData::UInt(*to, data.to_vec())
                }
                (TensorData::Float(_, data), DataType::Float(to)) => {
                    TensorData::Float(*to, data.to_vec())
                }
                _ => return None,
            };
            let tensor = Tensor::new(dims.clone(), data).ok()?;
            Some(vec![tensor])
        }
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
        _ => None,
    }
}
