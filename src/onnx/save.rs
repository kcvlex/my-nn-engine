use std::path::Path;

use prost::Message;

use crate::onnx::load::attribute_proto;
use crate::onnx::load::tensor_proto;
use crate::onnx::load::tensor_shape_proto;
use crate::onnx::load::tensor_to_proto;
use crate::onnx::load::type_proto;
use crate::onnx::load::AttributeProto;
use crate::onnx::load::GraphProto;
use crate::onnx::load::ModelProto;
use crate::onnx::load::NodeProto;
use crate::onnx::load::OperatorSetIdProto;
use crate::onnx::load::TensorProto;
use crate::onnx::load::TensorShapeProto;
use crate::onnx::load::TypeProto;
use crate::onnx::load::ValueInfoProto;
use crate::onnx::model::Graph;
use crate::onnx::model::Model;
use crate::onnx::model::OpsetImport;
use crate::onnx::model::ValueId;
use crate::onnx::operator::*;
use crate::tensor::data::ScalarData;
use crate::tensor::types::DataType;
use crate::tensor::types::Dimension;
use crate::tensor::types::FloatType;
use crate::tensor::types::SIntType;
use crate::tensor::types::TensorType;
use crate::tensor::types::UIntType;

fn attr_float(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Float as i32,
        f: value,
        ..Default::default()
    }
}

fn attr_int(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Int as i32,
        i: value,
        ..Default::default()
    }
}

fn attr_ints(name: &str, values: Vec<i64>) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Ints as i32,
        ints: values,
        ..Default::default()
    }
}

fn attr_string(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::String as i32,
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn attr_tensor(name: &str, proto: TensorProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Tensor as i32,
        t: Some(proto),
        ..Default::default()
    }
}

fn data_type_to_i32(dt: DataType) -> i32 {
    match dt {
        DataType::SInt(SIntType::I32) => tensor_proto::DataType::Int32 as i32,
        DataType::SInt(SIntType::I64) => tensor_proto::DataType::Int64 as i32,
        DataType::UInt(UIntType::U64) => tensor_proto::DataType::Uint64 as i32,
        DataType::Float(FloatType::F32) => tensor_proto::DataType::Float as i32,
        DataType::Float(FloatType::F64) => tensor_proto::DataType::Double as i32,
    }
}

fn scalar_to_tensor_proto(scalar: &ScalarData) -> TensorProto {
    match scalar {
        ScalarData::Float(FloatType::F32, v) => TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Float as i32,
            float_data: vec![*v as f32],
            ..Default::default()
        },
        ScalarData::Float(FloatType::F64, v) => TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Double as i32,
            double_data: vec![*v],
            ..Default::default()
        },
        ScalarData::SInt(SIntType::I32, v) => TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Int32 as i32,
            int32_data: vec![*v as i32],
            ..Default::default()
        },
        ScalarData::SInt(SIntType::I64, v) => TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Int64 as i32,
            int64_data: vec![*v],
            ..Default::default()
        },
        ScalarData::UInt(UIntType::U64, v) => TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Uint64 as i32,
            uint64_data: vec![*v],
            ..Default::default()
        },
    }
}

fn conv_pad_attrs(pad: &ConvPad) -> Vec<AttributeProto> {
    match pad {
        ConvPad::NotSet(pads) => {
            if let Some(v) = pads.inner() {
                let begins: Vec<i64> = v.iter().map(|(b, _)| *b as i64).collect();
                let ends: Vec<i64> = v.iter().map(|(_, e)| *e as i64).collect();
                let ints: Vec<i64> = begins.into_iter().chain(ends).collect();
                vec![attr_ints("pads", ints)]
            } else {
                vec![]
            }
        }
        ConvPad::SameUpper => vec![attr_string("auto_pad", "SAME_UPPER")],
        ConvPad::SameLower => vec![attr_string("auto_pad", "SAME_LOWER")],
        ConvPad::Valid => vec![attr_string("auto_pad", "VALID")],
    }
}

fn operator_attrs(op: &Operator) -> Vec<AttributeProto> {
    match op {
        Operator::Add |
        Operator::Sub |
        Operator::Mul |
        Operator::Div |
        Operator::Exp |
        Operator::Log |
        Operator::Pow |
        Operator::Sqrt |
        Operator::Reciprocal |
        Operator::ReLU |
        Operator::Sigmoid |
        Operator::Tanh |
        Operator::Identity |
        Operator::MatMul |
        Operator::Reshape |
        Operator::Slice |
        Operator::NonZero |
        Operator::GlobalAveragePool |
        Operator::Transfer(_) => vec![],

        Operator::Attention(attn) => vec![
            attr_int("is_causal", if attn.is_causal { 1 } else { 0 }),
            attr_float("scale", attn.scale),
        ],

        Operator::BatchedGemm(gemm) => vec![
            attr_float("alpha", gemm.alpha as f32),
            attr_float("beta", gemm.beta as f32),
            attr_int("trans_a", if gemm.trans_a { 1 } else { 0 }),
            attr_int("trans_b", if gemm.trans_b { 1 } else { 0 }),
        ],

        Operator::BatchNormalization(bn) => vec![
            attr_float("epsilon", bn.epsilon),
            attr_float("momentum", bn.momentum),
        ],

        Operator::Clip(c) => {
            let mut attrs = vec![];
            if let Some(min) = c.min {
                attrs.push(attr_float("min", min as f32));
            }
            if let Some(max) = c.max {
                attrs.push(attr_float("max", max as f32));
            }
            attrs
        }

        Operator::Cast(c) => vec![attr_int("to", data_type_to_i32(c.to) as i64)],

        Operator::Concat(c) => vec![attr_int("axis", c.axis.raw() as i64)],

        Operator::Constant(c) => vec![attr_tensor("value", tensor_to_proto(&c.value))],

        Operator::ConstantOfShape(c) => {
            vec![attr_tensor("value", scalar_to_tensor_proto(&c.value))]
        }

        Operator::Conv(c) => {
            let mut attrs = vec![];
            if let Some(v) = c.dilations.inner() {
                attrs.push(attr_ints(
                    "dilations",
                    v.iter().map(|&x| x as i64).collect(),
                ));
            }
            if c.groups != 1 {
                attrs.push(attr_int("groups", c.groups as i64));
            }
            attrs.push(attr_ints(
                "kernel_shape",
                c.kernel_shape.iter().map(|&x| x as i64).collect(),
            ));
            if let Some(v) = c.strides.inner() {
                attrs.push(attr_ints("strides", v.iter().map(|&x| x as i64).collect()));
            }
            attrs.extend(conv_pad_attrs(&c.pad));
            attrs
        }

        Operator::Gather(g) => vec![attr_int("axis", g.axis.raw() as i64)],

        Operator::GeLU(g) => {
            let approx = if g.approximate { "tanh" } else { "none" };
            vec![attr_string("approximate", approx)]
        }

        Operator::Gemm(g) => {
            let mut attrs = vec![];
            if g.trans_a {
                attrs.push(attr_int("transA", 1));
            }
            if g.trans_b {
                attrs.push(attr_int("transB", 1));
            }
            if g.alpha != 1.0 {
                attrs.push(attr_float("alpha", g.alpha as f32));
            }
            if g.beta != 1.0 {
                attrs.push(attr_float("beta", g.beta as f32));
            }
            attrs
        }

        Operator::LayerNormalization(l) => vec![
            attr_int("axis", l.axis.raw() as i64),
            attr_float("epsilon", l.epsilon as f32),
        ],

        Operator::LeakyReLU(l) => vec![attr_float("alpha", l.alpha as f32)],

        Operator::MaxPool(p) => {
            let mut attrs = vec![];
            if p.ceil_mode {
                attrs.push(attr_int("ceil_mode", 1));
            }
            if let Some(v) = p.dilations.inner() {
                attrs.push(attr_ints(
                    "dilations",
                    v.iter().map(|&x| x as i64).collect(),
                ));
            }
            attrs.push(attr_ints(
                "kernel_shape",
                p.kernel_shape.iter().map(|&x| x as i64).collect(),
            ));
            if let Some(v) = p.strides.inner() {
                attrs.push(attr_ints("strides", v.iter().map(|&x| x as i64).collect()));
            }
            attrs.extend(conv_pad_attrs(&p.pad));
            attrs
        }

        Operator::OneHot(o) => vec![attr_int("axis", o.axis as i64)],

        Operator::ReduceMax(r) => {
            let mut attrs = vec![];
            if !r.axes.is_empty() {
                attrs.push(attr_ints("axes", r.axes.clone()));
            }
            if !r.keepdims {
                attrs.push(attr_int("keepdims", 0));
            }
            attrs
        }

        Operator::ReduceMean(r) => {
            let mut attrs = vec![];
            if !r.axes.is_empty() {
                attrs.push(attr_ints("axes", r.axes.clone()));
            }
            if !r.keepdims {
                attrs.push(attr_int("keepdims", 0));
            }
            attrs
        }

        Operator::ReduceSum(r) => {
            let mut attrs = vec![];
            if !r.axes.is_empty() {
                attrs.push(attr_ints("axes", r.axes.clone()));
            }
            if !r.keepdims {
                attrs.push(attr_int("keepdims", 0));
            }
            attrs
        }

        Operator::Resize(r) => {
            let mut attrs = vec![];
            if let Some(ref axes) = r.axes {
                attrs.push(attr_ints(
                    "axes",
                    axes.iter().map(|x| x.raw() as i64).collect(),
                ));
            }
            match r.coordinate_transformation_mode {
                ResizeCoordinateTransformationMode::HalfPixel => {
                    attrs.push(attr_string("coordinate_transformation_mode", "half_pixel"));
                }
            }
            match r.keep_aspect_ratio_policy {
                ResizeKeepAspectRatioPolicy::Stretch => {}
                ResizeKeepAspectRatioPolicy::NotLarger => {
                    attrs.push(attr_string("keep_aspect_ratio", "not_larger"));
                }
                ResizeKeepAspectRatioPolicy::NotSmaller => {
                    attrs.push(attr_string("keep_aspect_ratio", "not_smaller"));
                }
            }
            match r.mode {
                ResizeMode::Nearest(nearest) => {
                    attrs.push(attr_string("mode", "nearest"));
                    match nearest {
                        ResizeNearestMode::RoundPreferFloor => {}
                        ResizeNearestMode::RoundPreferCeil => {
                            attrs.push(attr_string("nearest_mode", "round_prefer_ceil"));
                        }
                        ResizeNearestMode::Floor => {
                            attrs.push(attr_string("nearest_mode", "floor"));
                        }
                        ResizeNearestMode::Ceil => {
                            attrs.push(attr_string("nearest_mode", "ceil"));
                        }
                    }
                }
            }
            attrs
        }

        Operator::Shape(s) => {
            let mut attrs = vec![];
            if s.start.raw() != 0 {
                attrs.push(attr_int("start", s.start.raw() as i64));
            }
            if let Some(end) = s.end {
                attrs.push(attr_int("end", end.raw() as i64));
            }
            attrs
        }

        Operator::Softmax(s) => vec![attr_int("axis", s.axis.raw() as i64)],

        Operator::Split(s) => {
            let mut attrs = vec![attr_int("axis", s.axis.raw() as i64)];
            match &s.outputs {
                Some(SplitOutputs::NumOutputs(n)) => {
                    attrs.push(attr_int("num_outputs", *n as i64));
                }
                Some(SplitOutputs::Split(sizes)) => {
                    attrs.push(attr_ints(
                        "split",
                        sizes.iter().map(|&x| x as i64).collect(),
                    ));
                }
                None => {}
            }
            attrs
        }

        Operator::Squeeze(s) => {
            let mut attrs = vec![];
            if let Some(ref axes) = s.axes {
                attrs.push(attr_ints(
                    "axes",
                    axes.iter().map(|x| x.raw() as i64).collect(),
                ));
            }
            attrs
        }

        Operator::Transpose(t) => {
            let mut attrs = vec![];
            if let Some(ref perm) = t.perm {
                attrs.push(attr_ints("perm", perm.iter().map(|&x| x as i64).collect()));
            }
            attrs
        }

        Operator::Unsqueeze(u) => {
            let mut attrs = vec![];
            if !u.axes.is_empty() {
                attrs.push(attr_ints(
                    "axes",
                    u.axes.iter().map(|x| x.raw() as i64).collect(),
                ));
            }
            attrs
        }

        // Custom Operators.
        Operator::Contiguous(_) => vec![],
        Operator::NHWC2NCHW => vec![],

        // TODO
        Operator::Im2Col(_) => vec![],

        Operator::ReduceMatrix(op) => {
            let op = match op {
                ReduceOp::Max => "max",
                ReduceOp::Mean => "mean",
                ReduceOp::Variance => "variance",
                ReduceOp::Sum => "sum",
            };
            vec![attr_string("reduction", op)]
        }

        // TODO
        Operator::Reinterpret(_) => vec![],

        Operator::Input(_) | Operator::Output(_) => unreachable!(),
    }
}

fn type_to_proto(ty: &TensorType) -> TypeProto {
    match ty {
        TensorType::Resolved(resolved) => {
            let shape = TensorShapeProto {
                dim: resolved
                    .dims
                    .iter()
                    .map(|&d| tensor_shape_proto::Dimension {
                        value: Some(tensor_shape_proto::dimension::Value::DimValue(d as i64)),
                        denotation: String::new(),
                    })
                    .collect(),
            };
            TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: data_type_to_i32(resolved.elem_type),
                    shape: Some(shape),
                })),
                denotation: String::new(),
            }
        }
        TensorType::Unresolved(unresolved) => {
            let shape = unresolved.dims.as_ref().map(|dims| TensorShapeProto {
                dim: dims
                    .inner()
                    .iter()
                    .map(|d| {
                        let value = match d {
                            Dimension::Const(x) => {
                                tensor_shape_proto::dimension::Value::DimValue(*x as i64)
                            }
                            Dimension::Param(p) => {
                                tensor_shape_proto::dimension::Value::DimParam(p.to_string())
                            }
                        };
                        tensor_shape_proto::Dimension {
                            value: Some(value),
                            denotation: String::new(),
                        }
                    })
                    .collect(),
            });
            TypeProto {
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: data_type_to_i32(unresolved.elem_type),
                    shape,
                })),
                denotation: String::new(),
            }
        }
    }
}

fn value_info_to_proto(graph: &Graph, value_id: ValueId) -> ValueInfoProto {
    let info = &graph.values[value_id];
    ValueInfoProto {
        name: info.name.clone(),
        r#type: info.ty.as_ref().map(type_to_proto),
        doc_string: String::new(),
        metadata_props: vec![],
    }
}

fn graph_to_proto(graph: &Graph) -> GraphProto {
    let nodes: Vec<NodeProto> = graph
        .nodes
        .iter()
        .filter(|(_, node)| !node.is_dummy())
        .map(|(_, node)| {
            let attribute = operator_attrs(&node.op);
            NodeProto {
                input: node
                    .inputs
                    .iter()
                    .map(|v| match v {
                        Some(v) => graph.values[*v].name.clone(),
                        None => String::new(),
                    })
                    .collect(),
                output: node
                    .outputs
                    .iter()
                    .map(|&v| graph.values[v].name.clone())
                    .collect(),
                name: node.name.clone(),
                op_type: node.op.name().to_string(),
                domain: String::new(),
                attribute,
                doc_string: String::new(),
                overload: String::new(),
                metadata_props: vec![],
            }
        })
        .collect();

    let input: Vec<ValueInfoProto> = graph
        .input_values()
        .into_iter()
        .filter(|v| !graph.initializer.contains_key(v))
        .map(|v| value_info_to_proto(graph, v))
        .collect();

    let output: Vec<ValueInfoProto> = graph
        .output_values()
        .into_iter()
        .map(|v| value_info_to_proto(graph, v))
        .collect();

    let initializer: Vec<TensorProto> = graph
        .initializer
        .iter()
        .map(|(&value_id, tensor)| {
            let mut proto = tensor_to_proto(tensor);
            proto.name = graph.values[value_id].name.clone();
            proto
        })
        .collect();

    GraphProto {
        node: nodes,
        name: graph.name.clone(),
        initializer,
        input,
        output,
        doc_string: String::new(),
        ..Default::default()
    }
}

fn opset_to_proto(opset: &OpsetImport) -> OperatorSetIdProto {
    OperatorSetIdProto {
        domain: opset.domain.clone(),
        version: opset.version,
    }
}

impl Model {
    pub fn save_to_path<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let proto = ModelProto {
            ir_version: self.ir_version,
            producer_name: self.producer_name.clone(),
            producer_version: self.producer_version.clone(),
            domain: self.domain.clone(),
            model_version: self.model_version,
            doc_string: self.doc_string.clone(),
            graph: Some(graph_to_proto(&self.graph)),
            opset_import: self.opset_import.iter().map(opset_to_proto).collect(),
            ..Default::default()
        };
        let bytes = proto.encode_to_vec();
        std::fs::write(path, bytes)
    }
}

pub fn save_graph<P: AsRef<Path>>(graph: &Graph, path: P) {
    let proto = ModelProto {
        ir_version: 7,
        graph: Some(graph_to_proto(graph)),
        ..Default::default()
    };
    let bytes = proto.encode_to_vec();
    std::fs::write(path, bytes).unwrap();
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::onnx::load::LoadProto;
    use crate::onnx::model::Model;

    #[test]
    fn test_save_roundtrip_mnist12() {
        let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/validated/mnist-12");
        let original_path = root_dir.join("mnist-12.onnx");
        let saved_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/test_roundtrip_mnist12.onnx");

        let original = Model::load_from_path(&original_path).unwrap();
        original.save_to_path(&saved_path).unwrap();
        let reloaded = Model::load_from_path(&saved_path).unwrap();

        // Metadata
        assert_eq!(original.ir_version, reloaded.ir_version);
        assert_eq!(original.opset_import.len(), reloaded.opset_import.len());
        for (a, b) in original
            .opset_import
            .iter()
            .zip(reloaded.opset_import.iter())
        {
            assert_eq!(a.domain, b.domain);
            assert_eq!(a.version, b.version);
        }
        assert_eq!(original.producer_name, reloaded.producer_name);
        assert_eq!(original.producer_version, reloaded.producer_version);
        assert_eq!(original.domain, reloaded.domain);
        assert_eq!(original.model_version, reloaded.model_version);

        // Graph structure
        assert_eq!(original.graph.name, reloaded.graph.name);
        assert_eq!(original.graph.inputs.len(), reloaded.graph.inputs.len());
        assert_eq!(original.graph.outputs.len(), reloaded.graph.outputs.len());

        // Initializers
        assert_eq!(
            original.graph.initializer.len(),
            reloaded.graph.initializer.len()
        );
        for ((&oid, otensor), (&rid, rtensor)) in original
            .graph
            .initializer
            .iter()
            .zip(reloaded.graph.initializer.iter())
        {
            let oname = &original.graph.values[oid].name;
            let rname = &reloaded.graph.values[rid].name;
            assert_eq!(oname, rname, "initializer name mismatch");
            assert_eq!(otensor, rtensor, "initializer data mismatch for {oname}");
        }

        // Nodes
        let orig_nodes: Vec<_> = original
            .graph
            .nodes
            .iter()
            .filter(|(_, n)| !n.is_dummy())
            .collect();
        let rel_nodes: Vec<_> = reloaded
            .graph
            .nodes
            .iter()
            .filter(|(_, n)| !n.is_dummy())
            .collect();
        assert_eq!(orig_nodes.len(), rel_nodes.len(), "node count mismatch");

        for ((_, orig), (_, rel)) in orig_nodes.iter().zip(rel_nodes.iter()) {
            assert_eq!(orig.name, rel.name, "node name mismatch");
            assert_eq!(orig.op, rel.op, "operator mismatch for node {}", orig.name);
            assert_eq!(orig.inputs.len(), rel.inputs.len());
            assert_eq!(orig.outputs.len(), rel.outputs.len());
            for (oi, ri) in orig.inputs.iter().zip(rel.inputs.iter()) {
                match (oi, ri) {
                    (Some(oi), Some(ri)) => {
                        assert_eq!(
                            original.graph.values[*oi].name,
                            reloaded.graph.values[*ri].name,
                        );
                    }
                    (None, None) => {}
                    _ => panic!("input mismatch"),
                }
            }
            for (oo, ro) in orig.outputs.iter().zip(rel.outputs.iter()) {
                assert_eq!(
                    original.graph.values[*oo].name,
                    reloaded.graph.values[*ro].name,
                );
            }
        }

        let _ = std::fs::remove_file(&saved_path);
    }
}
