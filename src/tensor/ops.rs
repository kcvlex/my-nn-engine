use ndarray::concatenate;
use ndarray::stack;
use ndarray::Array;
use ndarray::ArrayView;
use ndarray::Axis;
use ndarray::IxDyn;
use ndarray::Slice;

use crate::onnx::operator::TensorIndex;

pub struct RawTensor<'a, T> {
    pub data: Vec<T>,
    pub dims: &'a [usize],
}

macro_rules! into_array_view {
    ($tensor: expr) => {{
        ArrayView::from_shape($tensor.dims, $tensor.data.as_slice())
    }};
}

pub fn ndarray_transpose<T: Clone>(tensor: RawTensor<'_, T>, perms: &[usize]) -> Array<T, IxDyn> {
    into_array_view!(tensor)
        .unwrap()
        .permuted_axes(perms)
        .into_dyn()
        .to_owned()
}

pub fn ndarray_slices<T: Clone>(
    tensor: RawTensor<'_, T>,
    starts: &[isize],
    ends: &[isize],
) -> Array<T, IxDyn> {
    into_array_view!(tensor)
        .unwrap()
        .slice_each_axis(|desc| Slice {
            start: starts[desc.axis.index()],
            end: Some(ends[desc.axis.index()]),
            step: 1,
        })
        .into_dyn()
        .to_owned()
}

pub fn ndarray_concat<T: Clone>(data: &[RawTensor<'_, T>], axis: usize) -> Array<T, IxDyn> {
    let arrays = data
        .iter()
        .map(|tensor| into_array_view!(tensor).unwrap())
        .collect::<Vec<_>>();
    concatenate(Axis(axis), &arrays[..]).unwrap().to_owned()
}

pub fn ndarray_gather<T: Clone, U: Clone + TryInto<isize>>(
    data: RawTensor<'_, T>,
    axis: usize,
    indices: RawTensor<'_, U>,
) -> Array<T, IxDyn> {
    if indices.dims.len() != 1 {
        unimplemented!("Indices must be 1D tensor");
    }

    let axis_dim = data.dims[axis];
    let indices = into_array_view!(indices)
        .unwrap()
        .flatten()
        .into_iter()
        .map(|x| x.try_into().map_err(|_| "convert").unwrap())
        .map(|x| TensorIndex::new(x).index(axis_dim))
        .collect::<Vec<_>>();
    into_array_view!(data)
        .unwrap()
        .select(Axis(axis), &indices[..])
        .into_dyn()
        .to_owned()
}

pub fn ndarray_broadcast<T: Clone>(
    tensor: RawTensor<'_, T>,
    target_dims: &[usize],
) -> Array<T, IxDyn> {
    into_array_view!(tensor)
        .unwrap()
        .broadcast(target_dims)
        .unwrap()
        .to_owned()
}

#[allow(unused)]
fn experimental_ndarray_gather<T: Clone, U: Clone + TryInto<isize>>(
    data: RawTensor<'_, T>,
    axis: usize,
    indices: RawTensor<'_, U>,
) -> Array<T, IxDyn> {
    let input = into_array_view!(data).unwrap();
    let dims = {
        let mut dims = data.dims.to_vec();
        let shape = input.shape();
        dims.extend_from_slice(&shape[..axis]);
        if axis + 1 < shape.len() {
            dims.extend_from_slice(&shape[axis + 1..]);
        }
        dims
    };
    let axis_dim = input.shape()[axis];
    let slices = into_array_view!(indices)
        .unwrap()
        .flatten()
        .into_iter()
        .map(|x| {
            input.index_axis(
                Axis(axis),
                TensorIndex::new(x.try_into().map_err(|_| "convert").unwrap()).index(axis_dim),
            )
        })
        .collect::<Vec<_>>();
    stack(Axis(axis), &slices[..])
        .unwrap()
        .to_shape(dims)
        .unwrap()
        .into_dyn()
        .to_owned()
}
