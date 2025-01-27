use crate::onnx::model::{Graph, NodeId};
use crate::onnx::operator::*;
use crate::optimize::optimizer::GraphModifier;
use crate::tensor::{
    resolved_dimensions::ResolvedTensorDims,
    tensor::{Tensor, TensorData},
};
use itertools::izip;

fn all_slice_indices(dims: &ResolvedTensorDims) -> (Vec<isize>, Vec<isize>) {
    let starts = vec![0; dims.ndim()];
    let ends = dims.as_slice().iter().map(|x| *x as isize).collect();
    (starts, ends)
}

pub fn constant_fold(graph: &Graph, node_id: NodeId) -> Option<Vec<Tensor>> {
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
            let shape = input[start..end].into_iter().map(|x| *x as u64).collect();
            let shape = TensorData::U64(shape);
            let shape = shape.into_1d_tensor();
            Some(vec![shape])
        }
        Operator::Slice(ref slice) => {
            let input = node.inputs[0];
            let input = &graph.initializer.get(&input)?;
            let (mut starts, mut ends) = all_slice_indices(&input.ty.dims);
            for (start, end, axis, step) in izip!(
                slice.starts.iter(),
                slice.ends.iter(),
                slice.axes.iter(),
                slice.steps.iter()
            ) {
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
        _ => None,
    }
}
