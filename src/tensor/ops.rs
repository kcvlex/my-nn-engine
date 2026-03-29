use ndarray::concatenate;
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
    let axis_dim = data.dims[axis];
    let flat_indices = into_array_view!(indices)
        .unwrap()
        .flatten()
        .into_iter()
        .map(|x| x.try_into().map_err(|_| "convert").unwrap())
        .map(|x| TensorIndex::new(x).index(axis_dim))
        .collect::<Vec<_>>();

    let selected = into_array_view!(data)
        .unwrap()
        .select(Axis(axis), &flat_indices[..]);

    let mut out_dims = Vec::new();
    out_dims.extend_from_slice(&data.dims[..axis]);
    out_dims.extend_from_slice(indices.dims);
    out_dims.extend_from_slice(&data.dims[axis + 1..]);

    selected
        .into_shape_with_order(&out_dims[..])
        .unwrap()
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
