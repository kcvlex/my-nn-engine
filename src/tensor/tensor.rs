use crate::tensor::dimensions::UnresolvedTensorDims;
use crate::tensor::resolved_dimensions::ResolvedTensorDims;
use itertools::izip;

#[derive(Debug, Clone)]
pub enum TypeError {
    InvalidShape(usize, ResolvedTensorDims),
    BroadcastError(ResolvedTensorDims, ResolvedTensorDims),
    ReshapeError(ResolvedTensorDims, ResolvedTensorDims),
    InferError(String),
    UnresolvedInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum DataType {
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    pub data: TensorData,
    pub ty: ResolvedTensorType,
}

// TODO: Complex
#[derive(Debug, Clone)]
pub enum TensorData {
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl Eq for TensorData {}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TensorData::I64(a), TensorData::I64(b)) => a == b,
            (TensorData::F32(a), TensorData::F32(b)) => a == b,
            (TensorData::F64(a), TensorData::F64(b)) => a == b,
            _ => false,
        }
    }
}

impl TensorData {
    pub fn size(&self) -> usize {
        match self {
            TensorData::I64(v) => v.len(),
            TensorData::F32(v) => v.len(),
            TensorData::F64(v) => v.len(),
        }
    }

    pub fn elem_type(&self) -> DataType {
        match self {
            TensorData::I64(_) => DataType::I64,
            TensorData::F32(_) => DataType::F32,
            TensorData::F64(_) => DataType::F64,
        }
    }

    pub fn raw_vec(&self) -> Vec<u8> {
        macro_rules! convert {
            ($v: expr) => {{
                $v.iter().map(|x| x.to_le_bytes()).flatten().collect()
            }};
        }

        match self {
            TensorData::I64(v) => convert!(v),
            TensorData::F32(v) => convert!(v),
            TensorData::F64(v) => convert!(v),
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        match self {
            TensorData::I64(v) => v.as_ptr() as *const u8,
            TensorData::F32(v) => v.as_ptr() as *const u8,
            TensorData::F64(v) => v.as_ptr() as *const u8,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorType {
    Unresolved(UnresolvedTensorType),
    Resolved(ResolvedTensorType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnresolvedTensorType {
    pub elem_type: DataType,
    pub dims: Option<UnresolvedTensorDims>,
}

impl TensorType {
    pub fn normalize(&mut self) {
        if let TensorType::Unresolved(ref ty) = self {
            if let Some(resolved) = ty.dims.as_ref().and_then(|x| x.to_resolved()) {
                *self =
                    TensorType::Resolved(ResolvedTensorType::new(ty.elem_type.clone(), resolved));
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedTensorType {
    pub elem_type: DataType,
    pub dims: ResolvedTensorDims,
    stride: ResolvedTensorDims,
}

impl ResolvedTensorType {
    pub fn new(elem_type: DataType, dims: ResolvedTensorDims) -> Self {
        let stride = calc_stride(&dims);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn stride(&self, i: usize) -> usize {
        self.stride[i]
    }

    pub fn broadcast(&self, target: &ResolvedTensorDims) -> Self {
        let elem_type = self.elem_type;
        let dims = target.clone();
        let stride = calc_stride_broadcast(&self.dims, target);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn is_fully_expanded(&self) -> bool {
        self.stride[0] == self.dims.size()
    }

    pub fn is_broadcast_required(&self, target: &ResolvedTensorDims) -> bool {
        !(self.is_fully_expanded() && self.dims == *target)
    }

    pub fn transpose(&self, perms: &[usize]) -> Self {
        let elem_type = self.elem_type;
        let dims = self.dims.transpose(perms);
        let stride = calc_stride(&dims);
        Self {
            elem_type,
            dims,
            stride,
        }
    }

    pub fn drop_head(&mut self) {
        self.dims = self.dims[1..].to_vec().into();
        self.stride = self.stride[1..].to_vec().into();
    }

    pub fn reshape(&self, dims: ResolvedTensorDims) -> Self {
        assert!(dims.size() == self.dims.size());
        Self::new(self.elem_type, dims)
    }

    pub fn pushed(&self, dim: usize) -> Self {
        let mut dims = self.dims.clone();
        dims.push(dim);
        Self::new(self.elem_type, dims)
    }
}

impl TensorType {
    pub fn as_resolved(&self) -> Option<&ResolvedTensorType> {
        match self {
            TensorType::Resolved(ty) => Some(ty),
            TensorType::Unresolved(_) => None,
        }
    }
}

impl From<ResolvedTensorType> for TensorType {
    fn from(ty: ResolvedTensorType) -> Self {
        TensorType::Resolved(ty)
    }
}

fn calc_stride_broadcast(
    orig: &ResolvedTensorDims,
    target: &ResolvedTensorDims,
) -> ResolvedTensorDims {
    let mut acc = 1;
    let mut stride = vec![0; target.ndim()];
    for (stride, orig, target) in izip!(
        stride.iter_mut().rev(),
        orig.iter().rev(),
        target.iter().rev()
    ) {
        if *orig == *target {
            *stride = acc;
            acc *= *orig;
        }
    }
    ResolvedTensorDims::new(stride)
}

fn calc_stride(dims: &ResolvedTensorDims) -> ResolvedTensorDims {
    let mut acc = 1;
    let mut stride = vec![0; dims.ndim()];
    for i in (0..dims.ndim()).rev() {
        stride[i] = acc;
        acc *= dims[i];
    }
    ResolvedTensorDims::new(stride)
}

impl Tensor {
    pub fn new(dims: ResolvedTensorDims, data: TensorData) -> Result<Self, TypeError> {
        if dims.size() != data.size() {
            return Err(TypeError::InvalidShape(data.size(), dims));
        }
        let ty = ResolvedTensorType::new(data.elem_type(), dims);
        Ok(Self { data, ty })
    }

    pub fn tensor_type(&self) -> TensorType {
        self.ty.clone().into()
    }

    pub fn zeros(ty: DataType, dims: ResolvedTensorDims) -> Self {
        let data = TensorData::zeros(ty, &dims);
        Self::new(dims, data).unwrap()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.data.raw_vec()
    }

    pub fn from_bytes(ty: ResolvedTensorType, raw: &[u8]) -> Result<Self, TypeError> {
        let data = TensorData::from_bytes(ty.elem_type, raw);
        Self::new(ty.dims, data)
    }
}

macro_rules! define_try_from {
    ($ty: ty, $data: ident) => {
        impl<D: ndarray::Dimension> TryFrom<ndarray::Array<$ty, D>> for Tensor {
            type Error = TypeError;
            fn try_from(array: ndarray::Array<$ty, D>) -> Result<Self, Self::Error> {
                let dim = ResolvedTensorDims::new(array.shape().to_vec());
                let data = TensorData::$data(array.flatten().to_vec());
                Self::new(dim, data)
            }
        }
    };
}

define_try_from!(f32, F32);
define_try_from!(f64, F64);
define_try_from!(i64, I64);

impl TensorData {
    pub fn from_bytes(ty: DataType, raw: &[u8]) -> Self {
        macro_rules! convert {
            ($v: expr, $ty: ty) => {{
                raw.chunks_exact(std::mem::size_of::<$ty>())
                    .map(|x| {
                        let mut bytes = [0; std::mem::size_of::<$ty>()];
                        bytes.copy_from_slice(x);
                        <$ty>::from_le_bytes(bytes)
                    })
                    .collect()
            }};
        }
        match ty {
            DataType::I64 => TensorData::I64(convert!(raw, i64)),
            DataType::F32 => TensorData::F32(convert!(raw, f32)),
            DataType::F64 => TensorData::F64(convert!(raw, f64)),
        }
    }

    pub fn zeros(ty: DataType, dims: &ResolvedTensorDims) -> Self {
        let size = dims.size();
        match ty {
            DataType::I64 => TensorData::I64(vec![0; size]),
            DataType::F32 => TensorData::F32(vec![0.0; size]),
            DataType::F64 => TensorData::F64(vec![0.0; size]),
        }
    }
}
