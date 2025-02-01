use crate::onnx::model::{Graph, NodeId};
use crate::onnx::operator::*;
use crate::tensor::{data::TensorData, dimensions::ResolvedTensorDims, Tensor};

fn all_slice_indices(dims: &ResolvedTensorDims) -> (Vec<isize>, Vec<isize>) {
    let starts = vec![0; dims.ndim()];
    let ends = dims[..].iter().map(|x| *x as isize).collect();
    (starts, ends)
}

pub fn fold_constant(graph: &Graph, node_id: NodeId) -> Option<Vec<Tensor>> {
    let node = &graph.nodes[node_id];
    match node.op {
        Operator::Shape(Shape { ref start, ref end }) => {
            let input = node.inputs[0];
            let input = &graph.get_resolved_tensor_type(input)?.dims;
            let ndim = input.ndim();
            let start = start.index(ndim);
            let end = match end {
                Some(index) => index.index(ndim),
                None => ndim,
            };
            let shape = input[start..end].iter().map(|x| *x as u64).collect();
            let shape = TensorData::U64(shape);
            let shape = shape.into_1d_tensor();
            Some(vec![shape])
        }
        Operator::Slice(ref slice) => {
            let input = node.inputs[0];
            let input = &graph.initializer.get(&input)?;
            let (mut starts, mut ends) = all_slice_indices(&input.ty.dims);
            for Slice {
                start,
                end,
                axis,
                step,
            } in slice.iter()
            {
                let axis = axis.index(input.ty.dims.ndim());
                if *step != 1 {
                    unimplemented!();
                }
                starts[axis] = start.raw();
                ends[axis] = end.raw();
            }
            Some(vec![input.slices(&starts, &ends)])
        }
        Operator::Split(Split {
            ref axis,
            ref num_outputs,
        }) => {
            let input = node.inputs[0];
            let input = &graph.initializer.get(&input)?;
            let (mut starts, mut ends) = all_slice_indices(&input.ty.dims);
            let axis = axis.index(input.ty.dims.ndim());
            let dim = input.ty.dims[axis];
            let mut res = Vec::new();
            let mut cur = 0;
            let step = (dim / *num_outputs) as isize;
            let bound = ends[axis];
            while cur < bound {
                let start = cur;
                let end = (cur + step).min(bound);
                starts[axis] = start;
                ends[axis] = end;
                res.push(input.slices(&starts, &ends));
                cur = end;
            }
            Some(res)
        }
        Operator::Gather(Gather { ref axis }) => {
            let input = &graph.initializer.get(&node.inputs[0])?;
            let indices = &graph.initializer.get(&node.inputs[1])?;
            let axis = axis.index(input.ty.dims.ndim());
            Some(vec![input.gather(indices, axis)])
        }
        _ => None,
    }
}
