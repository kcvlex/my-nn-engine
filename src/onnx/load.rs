use crate::onnx::model::{Graph, Model, Node, NodeMeta, Nodes, ValueId, ValueInfo, Values};
use crate::onnx::operator::*;
use crate::tensor::{
    data::TensorData,
    dimensions::{Dimension, ResolvedTensorDims, UnresolvedTensorDims},
    types::{DataType, FloatType, SIntType, TensorType, TypeError, UIntType, UnresolvedTensorType},
    Tensor,
};
use itertools::{zip_eq, Itertools};
use prost::{DecodeError, Message};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
include!(concat!(env!("OUT_DIR"), "/onnx.rs"));

#[derive(Debug)]
pub enum ModelLoadError {
    FileRead(std::io::Error),
    Decode(DecodeError),
    ElemTypeUnspecified,
    NoGraph,
    UnsupportedElemType(tensor_proto::DataType),
    UnsupportedValueType(type_proto::Value),
    UnsupportedAttributeType(attribute_proto::AttributeType),
    UnsupportedOp(String),
    NegativeDimension(i64),
    TypeError(TypeError),
    Required(String),
    Unexpected(String),
}

type LoadResult<T> = Result<T, ModelLoadError>;

pub trait LoadProto {
    fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Self>
    where
        Self: Sized;
}

impl LoadProto for Model {
    fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Self> {
        let model = std::fs::read(p).map_err(ModelLoadError::FileRead)?;
        let model = ModelProto::decode(&*model).map_err(ModelLoadError::Decode)?;
        let graph = model.graph.ok_or(ModelLoadError::NoGraph)?;
        let graph = GraphLoader::default().load_graph(graph)?;
        Ok(Model { graph })
    }
}

impl LoadProto for Tensor {
    fn load_from_path<P: AsRef<Path>>(p: P) -> LoadResult<Self> {
        let tensor = std::fs::read(p).map_err(ModelLoadError::FileRead)?;
        let tensor = TensorProto::decode(&*tensor).map_err(ModelLoadError::Decode)?;
        load_tensor(tensor)
    }
}

#[derive(Default)]
struct GraphLoader {
    entries: HashMap<String, ValueId>,
    values: Values,
}

#[allow(dead_code)]
#[derive(Debug)]
enum Attribute {
    Float(f32),
    Int(i64),
    Ints(Vec<i64>),
    Str(String),
    Strings(Vec<String>),
}

impl Attribute {
    fn f(&self) -> LoadResult<f32> {
        match self {
            Attribute::Float(x) => Ok(*x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn i(&self) -> LoadResult<i64> {
        match self {
            Attribute::Int(x) => Ok(*x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn index(&self) -> LoadResult<TensorIndex> {
        self.i().map(|x| TensorIndex::new(x as isize))
    }

    fn b(&self) -> LoadResult<bool> {
        self.i().map(|x| x != 0)
    }

    fn ints<T>(&self) -> LoadResult<Vec<T>>
    where
        T: TryFrom<i64>,
    {
        match self {
            Attribute::Ints(x) => {
                let mut vec = Vec::with_capacity(x.len());
                for i in x.iter() {
                    vec.push(
                        T::try_from(*i).map_err(|_| ModelLoadError::Unexpected("".to_string()))?,
                    );
                }
                Ok(vec)
            }
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn indexes(&self) -> LoadResult<Vec<TensorIndex>> {
        let ints = self.ints()?;
        Ok(ints.into_iter().map(TensorIndex::new).collect())
    }

    fn s(&self) -> LoadResult<&str> {
        match self {
            Attribute::Str(x) => Ok(x),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }

    fn ty(&self) -> LoadResult<DataType> {
        self.i().and_then(|x| DataType::try_from(x as i32))
    }

    fn strings(&self) -> LoadResult<Vec<String>> {
        match self {
            Attribute::Strings(x) => Ok(x.clone()),
            x => Err(ModelLoadError::Unexpected(format!("{:?}", x))),
        }
    }
}

type Attributes = HashMap<String, Attribute>;

impl GraphLoader {
    fn load_graph(mut self, graph: GraphProto) -> LoadResult<Graph> {
        let initializer = self.load_initializer(graph.initializer)?;
        let inputs = self.load_value_info_vec(graph.input)?;
        let outputs = self.load_value_info_vec(graph.output)?;
        let mut nodes = self.load_nodes(graph.node)?;
        let inputs = {
            let mut res = Vec::with_capacity(inputs.len());
            for &x in inputs.iter() {
                let node = Node {
                    name: self.values[x].name.clone(),
                    inputs: Vec::new(),
                    outputs: vec![x],
                    op: Operator::Input(x),
                    meta: NodeMeta::default(),
                };
                res.push(nodes.alloc(node));
            }
            res
        };
        let outputs = outputs
            .into_iter()
            .map(|x| {
                let node = Node {
                    name: self.values[x].name.clone(),
                    inputs: vec![x],
                    outputs: Vec::new(),
                    op: Operator::Output(x),
                    meta: NodeMeta::default(),
                };
                nodes.alloc(node)
            })
            .collect();

        // Fill dummy values
        let mut initializer = initializer;
        let defined: HashSet<ValueId> = initializer
            .keys()
            .copied()
            .chain(inputs.iter().map(|&x| nodes[x].outputs[0]))
            .chain(initializer.keys().copied())
            .chain(
                nodes
                    .iter()
                    .flat_map(|(_, node)| node.outputs.iter().copied()),
            )
            .collect();
        let tensor = Tensor::new(
            ResolvedTensorDims::new(vec![]),
            TensorData::Float(FloatType::F32, vec![]),
        )
        .unwrap();
        for value_id in nodes
            .iter()
            .flat_map(|(_, node)| node.inputs.iter())
            .filter(|&x| !defined.contains(x))
            .unique()
        {
            initializer.insert(*value_id, tensor.clone());
            self.values[*value_id].ty = Some(TensorType::Resolved(tensor.tensor_type()));
        }

        Ok(Graph {
            name: graph.name,
            initializer,
            inputs,
            outputs,
            values: self.values,
            nodes,
            resolved_params: HashMap::new(),
        })
    }

    fn load_value_info_vec(&mut self, v: Vec<ValueInfoProto>) -> LoadResult<Vec<ValueId>> {
        let mut res = Vec::new();
        for info in v.into_iter() {
            let name = info.name;
            let ty = info.r#type.ok_or(ModelLoadError::Unexpected(
                "ValueInfo.type must be specified".to_string(),
            ))?;
            let ty = load_type(ty)?;
            let id = self
                .entries
                .entry(name.clone())
                .or_insert_with(|| self.values.alloc(ValueInfo { name, ty: Some(ty) }));
            res.push(*id);
        }
        Ok(res)
    }

    fn load_initializer(&mut self, v: Vec<TensorProto>) -> LoadResult<BTreeMap<ValueId, Tensor>> {
        let mut res = BTreeMap::new();
        for tensor in v.into_iter() {
            let name = tensor.name.clone();
            let tensor = load_tensor(tensor)?;
            let id = self.entries.entry(name.clone()).or_insert_with(|| {
                self.values.alloc(ValueInfo {
                    name,
                    ty: Some(TensorType::Resolved(tensor.tensor_type())),
                })
            });
            res.insert(*id, tensor);
        }
        Ok(res)
    }

    fn load_nodes(&mut self, nodes: Vec<NodeProto>) -> LoadResult<Nodes> {
        macro_rules! io {
            ($v: expr) => {{
                $v.into_iter()
                    .map(|x| {
                        *self.entries.entry(x.clone()).or_insert_with(|| {
                            self.values.alloc(ValueInfo {
                                name: x.clone(),
                                ty: None,
                            })
                        })
                    })
                    .collect()
            }};
        }
        let mut res = Nodes::default();
        for node in nodes.into_iter() {
            let name = node.name;
            let inputs = io!(node.input);
            let outputs = io!(node.output);
            let attributes = load_attributes(node.attribute)?;
            let op = load_op(&node.op_type, &attributes)?;
            res.alloc(Node {
                name,
                inputs,
                outputs,
                op,
                meta: NodeMeta::default(),
            });
        }
        Ok(res)
    }
}

fn load_tensor(tensor: TensorProto) -> LoadResult<Tensor> {
    let elem_type = DataType::try_from(tensor.data_type)?;
    let data = if tensor.raw_data.is_empty() {
        match elem_type {
            DataType::SInt(ty @ SIntType::I32) => {
                TensorData::SInt(ty, tensor.int32_data.into_iter().map(i64::from).collect())
            }
            DataType::SInt(ty @ SIntType::I64) => TensorData::SInt(ty, tensor.int64_data),
            DataType::UInt(ty @ UIntType::U64) => TensorData::UInt(ty, tensor.uint64_data),
            DataType::Float(ty @ FloatType::F32) => {
                TensorData::Float(ty, tensor.float_data.into_iter().map(f64::from).collect())
            }
            DataType::Float(ty @ FloatType::F64) => TensorData::Float(ty, tensor.double_data),
        }
    } else {
        TensorData::from_bytes(elem_type, tensor.raw_data.as_slice())
    };
    let mut dims = Vec::new();
    for dim in tensor.dims.into_iter() {
        let dim = usize::try_from(dim).map_err(|_| ModelLoadError::NegativeDimension(dim))?;
        dims.push(dim);
    }
    let dims = ResolvedTensorDims::new(dims);
    Tensor::new(dims, data).map_err(ModelLoadError::TypeError)
}

fn load_type(ty: TypeProto) -> LoadResult<TensorType> {
    let ty = ty.value.ok_or(ModelLoadError::Unexpected(
        "TypeProto.value must be specified".to_string(),
    ))?;
    match ty {
        type_proto::Value::TensorType(tensor) => load_tensor_type(tensor),
        x => Err(ModelLoadError::UnsupportedValueType(x)),
    }
}

fn load_tensor_type(tensor: type_proto::Tensor) -> LoadResult<TensorType> {
    let elem_type = DataType::try_from(tensor.elem_type)?;
    if let Some(dims) = tensor.shape {
        let mut shape = Vec::new();
        for dim in dims.dim.into_iter() {
            let dim: Dimension = dim
                .value
                .ok_or(ModelLoadError::Unexpected(
                    "Dimension.value must be specified".to_string(),
                ))?
                .try_into()?;
            shape.push(dim);
        }
        let mut ty = TensorType::Unresolved(UnresolvedTensorType {
            elem_type,
            dims: Some(UnresolvedTensorDims::new(shape)),
        });
        ty.normalize();
        Ok(ty)
    } else {
        Ok(TensorType::Unresolved(UnresolvedTensorType {
            elem_type,
            dims: None,
        }))
    }
}

impl TryFrom<i32> for DataType {
    type Error = ModelLoadError;
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        let value = tensor_proto::DataType::try_from(value)
            .map_err(|e| ModelLoadError::Unexpected(format!("Invalid DataType: {:?}", e)))?;
        match value {
            tensor_proto::DataType::Float => Ok(FloatType::F32.into()),
            tensor_proto::DataType::Double => Ok(FloatType::F64.into()),
            tensor_proto::DataType::Int32 => Ok(SIntType::I32.into()),
            tensor_proto::DataType::Int64 => Ok(SIntType::I64.into()),
            tensor_proto::DataType::Uint64 => Ok(UIntType::U64.into()),
            tensor_proto::DataType::Undefined => Err(ModelLoadError::ElemTypeUnspecified),
            x => Err(ModelLoadError::UnsupportedElemType(x)),
        }
    }
}

impl TryFrom<tensor_shape_proto::dimension::Value> for Dimension {
    type Error = ModelLoadError;
    fn try_from(value: tensor_shape_proto::dimension::Value) -> Result<Self, Self::Error> {
        match value {
            tensor_shape_proto::dimension::Value::DimValue(x) => {
                let x = x
                    .try_into()
                    .map_err(|_| ModelLoadError::NegativeDimension(x))?;
                Ok(Dimension::Const(x))
            }
            tensor_shape_proto::dimension::Value::DimParam(x) => Ok(Dimension::Param(x.into())),
        }
    }
}

fn load_utf8(v: Vec<u8>) -> LoadResult<String> {
    String::from_utf8(v).map_err(|err| ModelLoadError::Unexpected(err.to_string()))
}

fn load_attributes(v: Vec<AttributeProto>) -> LoadResult<Attributes> {
    let mut res = HashMap::new();
    for attr in v.into_iter() {
        let name = attr.name;
        let ty = attribute_proto::AttributeType::try_from(attr.r#type)
            .map_err(|err| ModelLoadError::Unexpected(err.to_string()))?;
        let value = match ty {
            attribute_proto::AttributeType::Float => Ok(Attribute::Float(attr.f)),
            attribute_proto::AttributeType::Int => Ok(Attribute::Int(attr.i)),
            attribute_proto::AttributeType::Ints => Ok(Attribute::Ints(attr.ints)),
            attribute_proto::AttributeType::String => load_utf8(attr.s).map(Attribute::Str),
            attribute_proto::AttributeType::Strings => attr
                .strings
                .into_iter()
                .map(load_utf8)
                .collect::<Result<Vec<_>, _>>()
                .map(|x| Attribute::Strings(x)),
            x => Err(ModelLoadError::UnsupportedAttributeType(x)),
        }?;
        res.insert(name, value);
    }
    Ok(res)
}

trait OptionalVecExt<T: Clone + Copy + PartialEq> {
    fn with_default(self, default: T) -> OptionalVec<T>;
}

impl<T: Clone + Copy + PartialEq> OptionalVecExt<T> for Option<Vec<T>> {
    fn with_default(self, default: T) -> OptionalVec<T> {
        OptionalVec::new(self, default)
    }
}

trait RequiredAttr {
    fn required(&self, name: &str) -> LoadResult<&Attribute>;
}

impl RequiredAttr for Attributes {
    fn required(&self, name: &str) -> LoadResult<&Attribute> {
        self.get(name)
            .ok_or(ModelLoadError::Required(name.to_string()))
    }
}

impl BatchNormalization {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let epsilon = attributes
            .get("epsilon")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1e-5);
        let momentum = attributes
            .get("momentum")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(0.9);
        Ok(BatchNormalization { epsilon, momentum })
    }
}

impl Cast {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        // TODO: saturate
        let to = attributes.required("to")?.ty()?;
        Ok(Cast { to })
    }
}

impl Concat {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes.required("axis")?.index()?;
        Ok(Concat { axis })
    }
}

impl ConvPad {
    fn load(attrs: &Attributes) -> LoadResult<Self> {
        let auto_pad = attrs.get("auto_pad").map_or(Ok("NOTSET"), |x| x.s())?;
        let pads = attrs
            .get("pads")
            .map(|x| x.ints::<usize>())
            .transpose()?
            .map(|v| {
                if v.len() % 2 != 0 {
                    Err(ModelLoadError::Unexpected("Invalid pads".to_string()))
                } else {
                    let half = v.len() / 2;
                    let mut res = Vec::with_capacity(half);
                    for i in 0..half {
                        res.push((v[i], v[i + half]));
                    }
                    Ok(res)
                }
            })
            .transpose()?;
        let pad = match (auto_pad, pads) {
            ("NOTSET", pads) => Self::NotSet(OptionalVec::new(pads, (0, 0))),
            ("SAME_UPPER", None) => Self::SameUpper,
            ("SAME_LOWER", None) => Self::SameLower,
            ("VALID", None) => Self::Valid,
            _ => return Err(ModelLoadError::Unexpected("Invalid padding".to_string())),
        };
        Ok(pad)
    }
}

impl Conv {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let dilations = attributes
            .get("dilations")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let groups = attributes.get("groups").map_or(Ok(1), |x| x.i())?;
        let kernel_shape = attributes
            .get("kernel_shape")
            .ok_or(ModelLoadError::Unexpected(
                "TODO: support inference of kernel_shape".to_string(),
            ))?
            .ints()?
            .into();
        let strides = attributes
            .get("strides")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let pad = ConvPad::load(attributes)?;
        Ok(Conv {
            pad,
            dilations,
            groups: groups as usize,
            kernel_shape,
            strides,
        })
    }
}

impl Gather {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        Ok(Gather { axis })
    }
}

impl Gemm {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let trans_a = attributes
            .get("transA")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let trans_b = attributes
            .get("transB")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let alpha = attributes
            .get("alpha")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1.0)
            .into();
        let beta = attributes
            .get("beta")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(1.0)
            .into();
        Ok(Gemm {
            trans_a,
            trans_b,
            alpha,
            beta,
        })
    }
}

impl LeakyReLU {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let alpha = attributes
            .get("alpha")
            .map(|x| x.f())
            .transpose()?
            .unwrap_or(0.01)
            .into();
        Ok(LeakyReLU { alpha })
    }
}

impl Pooling {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let dilations = attributes
            .get("dilations")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let ceil_mode = attributes
            .get("ceil_mode")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(false);
        let kernel_shape = attributes.required("kernel_shape")?.ints()?.into();
        let strides = attributes
            .get("strides")
            .map(|x| x.ints())
            .transpose()?
            .with_default(1);
        let pad = ConvPad::load(attributes)?;
        Ok(Pooling {
            pad,
            ceil_mode,
            dilations,
            kernel_shape,
            strides,
        })
    }
}

impl Reduce {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let keepdims = attributes
            .get("keepdims")
            .map(|x| x.b())
            .transpose()?
            .unwrap_or(true);
        let axes = attributes
            .get("axes")
            .map(|x| x.ints())
            .unwrap_or(Ok(Vec::new()))?;
        Ok(Self { keepdims, axes })
    }
}

impl ResizeNearestMode {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let nearest_mode = attributes.get("nearest_mode").map(|x| x.s()).transpose()?;
        let nearest_mode = match nearest_mode {
            Some("round_prefer_floor") | None => Self::RoundPreferFloor,
            Some("round_prefer_ceil") => Self::RoundPreferCeil,
            Some("floor") => Self::Floor,
            Some("ceil") => Self::Ceil,
            Some(_) => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid nearest_mode".to_string(),
                ))
            }
        };
        Ok(nearest_mode)
    }
}

impl ResizeMode {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let mode = attributes
            .get("mode")
            .map(|x| x.s())
            .transpose()?
            .unwrap_or("nearest");
        match mode {
            "nearest" => Ok(Self::Nearest(ResizeNearestMode::load(attributes)?)),
            _ => unimplemented!(),
        }
    }
}

impl ResizeCoordinateTransformationMode {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let mode = attributes
            .get("coordinate_transformation_mode")
            .map(|x| x.s())
            .transpose()?;
        let mode = match mode {
            Some("half_pixel") | None => Self::HalfPixel,
            Some("pytorch_half_pixel") | Some("align_corners") | Some("asymmetric") => {
                unimplemented!()
            }
            Some(_) => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid coordinate_transformation_mode".to_string(),
                ))
            }
        };
        Ok(mode)
    }
}

impl ResizeKeepAspectRatioPolicy {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let policy = attributes
            .get("keep_aspect_ratio")
            .map(|x| x.s())
            .transpose()?;
        let policy = match policy {
            Some("stretch") | None => Self::Stretch,
            Some("not_larger") => Self::NotLarger,
            Some("not_smaller") => Self::NotSmaller,
            Some(_) => {
                return Err(ModelLoadError::Unexpected(
                    "Invalid keep_aspect_ratio".to_string(),
                ))
            }
        };
        Ok(policy)
    }
}

impl Resize {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axes = attributes.get("axes").map(|x| x.indexes()).transpose()?;
        let coordinate_transformation_mode = ResizeCoordinateTransformationMode::load(attributes)?;
        let keep_aspect_ratio_policy = ResizeKeepAspectRatioPolicy::load(attributes)?;
        let mode = ResizeMode::load(attributes)?;
        Ok(Self {
            axes,
            coordinate_transformation_mode,
            keep_aspect_ratio_policy,
            mode,
        })
    }
}

impl Shape {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let start = attributes
            .get("start")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        let end = attributes.get("end").map(|x| x.index()).transpose()?;
        Ok(Shape { start, end })
    }
}

impl SplitOutputs {
    // TODO: Support version 18 (num_outputs)
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let split = attributes.required("split")?.ints()?.into_iter().collect();
        Ok(Self::Split(split))
    }
}

impl Split {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let axis = attributes
            .get("axis")
            .map(|x| x.index())
            .transpose()?
            .unwrap_or(TensorIndex::new(0));
        let outputs = SplitOutputs::load(attributes)?;
        Ok(Split { axis, outputs })
    }
}

impl Transpose {
    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let perm = attributes.get("perm").map(|x| x.ints()).transpose()?;
        Ok(Transpose { perm })
    }
}

/// # Attributes
///
/// - **operators - STRINGS:** \
///   A list of elementwise operators.
///
/// - **num_arguments - INTS:** \
///   A list of the number of arguments for each operator. The length of this list must be equal to
///   the length of `operators`.
///
/// - **argument_types - STRINGS:** \
///   A list of a type of each argument. Each element must be either "input" or "intermediate". The
///   arguments for the i-th operator corresponds to `argument_types[l..r]`, where `l` and `r` is
///   as follows.
///
///   ```text
///   l = sum(num_arguments[0..i])
///   r = l + num_arguments[i]
///   ```
///
///   The following values must be equal.
///
///   - The sum of the values of `num_arguments`.
///   - The length of `argument_types`.
///   - The sum of the length of `input_indices` and `intermediate_indices`.
///
/// - **input_indices - INTS:** \
///   TODO: Write
///
/// - **intermediate_indices - INTS:** \
///   TODO: Write
impl ElementwiseOps {
    fn load_elementwise_op(op: &str) -> LoadResult<Operator> {
        match op {
            "Add" => Ok(Operator::Add),
            "Exp" => Ok(Operator::Exp),
            "Log" => Ok(Operator::Log),
            "Mul" => Ok(Operator::Mul),
            "Sigmoid" => Ok(Operator::Sigmoid),
            "Sub" => Ok(Operator::Sub),
            "Tanh" => Ok(Operator::Tanh),
            x => Err(ModelLoadError::UnsupportedOp(x.to_string())),
        }
    }

    fn load(attributes: &Attributes) -> LoadResult<Self> {
        let operators = attributes
            .get("operators")
            .ok_or(ModelLoadError::Required("operators".to_string()))?
            .strings()?;
        let num_arguments = attributes
            .get("num_arguments")
            .ok_or(ModelLoadError::Required("num_arguments".to_string()))?
            .ints::<usize>()?;
        let argument_types = attributes
            .get("argument_types")
            .ok_or(ModelLoadError::Required("argument_types".to_string()))?
            .strings()?;
        let input_indices = attributes
            .get("input_indices")
            .ok_or(ModelLoadError::Required("input_indices".to_string()))?
            .ints::<usize>()?;
        let intermediate_indices = attributes
            .get("intermediate_indices")
            .ok_or(ModelLoadError::Required("intermediate_indices".to_string()))?
            .ints::<usize>()?;

        let mut ops = Vec::with_capacity(operators.len());
        let mut arg_types_index = 0;
        let mut input_index = 0;
        let mut intermediate_index = 0;
        for (op, num_arg) in zip_eq(operators, num_arguments) {
            let op = Self::load_elementwise_op(&op)?;
            let mut args = Vec::with_capacity(num_arg);
            for ty in &argument_types[arg_types_index..arg_types_index + num_arg] {
                match ty.as_str() {
                    "input" => {
                        args.push(ElementwiseOpArg::Input(input_indices[input_index]));
                        input_index += 1;
                    }
                    "intermediate" => {
                        args.push(ElementwiseOpArg::NthResult(
                            intermediate_indices[intermediate_index],
                        ));
                        intermediate_index += 1;
                    }
                    _ => {
                        return Err(ModelLoadError::Unexpected(
                            "Invalid argument type".to_string(),
                        ))
                    }
                }
            }
            arg_types_index += num_arg;
            ops.push((Box::new(op), args));
        }

        Ok(Self { ops })
    }
}

fn load_op(op: &str, attributes: &Attributes) -> LoadResult<Operator> {
    match op {
        "Add" => Ok(Operator::Add),
        "BatchNormalization" => Ok(Operator::BatchNormalization(BatchNormalization::load(
            attributes,
        )?)),
        "Cast" => Ok(Operator::Cast(Cast::load(attributes)?)),
        "Concat" => Ok(Operator::Concat(Concat::load(attributes)?)),
        "Conv" => Ok(Operator::Conv(Conv::load(attributes)?)),
        "Exp" => Ok(Operator::Exp),
        "Gather" => Ok(Operator::Gather(Gather::load(attributes)?)),
        "Gemm" => Ok(Operator::Gemm(Gemm::load(attributes)?)),
        "GlobalAveragePool" => Ok(Operator::GlobalAveragePool),
        "LeakyRelu" => Ok(Operator::LeakyReLU(LeakyReLU::load(attributes)?)),
        "Log" => Ok(Operator::Log),
        "Identity" => Ok(Operator::Identity),
        "MatMul" => Ok(Operator::MatMul),
        "MaxPool" => Ok(Operator::MaxPool(Pooling::load(attributes)?)),
        "Mul" => Ok(Operator::Mul),
        "ReduceMax" => Ok(Operator::ReduceMax(Reduce::load(attributes)?)),
        "ReduceMean" => Ok(Operator::ReduceMean(Reduce::load(attributes)?)),
        "ReduceSum" => Ok(Operator::ReduceSum(Reduce::load(attributes)?)),
        "Relu" => Ok(Operator::ReLU),
        "Reshape" => Ok(Operator::Reshape),
        "Resize" => Ok(Operator::Resize(Resize::load(attributes)?)),
        "Shape" => Ok(Operator::Shape(Shape::load(attributes)?)),
        "Sigmoid" => Ok(Operator::Sigmoid),
        "Sub" => Ok(Operator::Sub),
        "Tanh" => Ok(Operator::Tanh),
        "Slice" => Ok(Operator::Slice),
        "Split" => Ok(Operator::Split(Split::load(attributes)?)),
        "Transpose" => Ok(Operator::Transpose(Transpose::load(attributes)?)),

        // Custom
        "ElementwiseOps" => Ok(Operator::ElementwiseOps(ElementwiseOps::load(attributes)?)),
        x => Err(ModelLoadError::UnsupportedOp(x.to_string())),
    }
}
