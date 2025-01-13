// use crate::onnx::load::Attribute;
// use crate::tensor::tensor::{DataType, Tensor, TensorData};
// use crate::onnx::operator::Operator;
// include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
//
// impl Attribute {
//     fn to_proto(self, name: String) -> AttributeProto {
//         let mut attr = AttributeProto {
//             name,
//             ..Default::default()
//         };
//
//         let ty = match self {
//             Attribute::Float(f) => {
//                 attr.f = f;
//                 attribute_proto::AttributeType::Float
//             }
//             Attribute::Int(i) => {
//                 attr.i = i;
//                 attribute_proto::AttributeType::Int
//             }
//             Attribute::Ints(v) => {
//                 attr.ints = v;
//                 attribute_proto::AttributeType::Ints
//             }
//             Attribute::Str(s) => {
//                 attr.s = s.into_bytes();
//                 attribute_proto::AttributeType::String
//             }
//         };
//         attr.set_type(ty);
//         attr
//     }
// }
//
// impl DataType {
//     fn to_proto(self) -> tensor_proto::DataType {
//         match self {
//             DataType::F32 => tensor_proto::DataType::Float,
//             DataType::F64 => tensor_proto::DataType::Double,
//             DataType::I64 => tensor_proto::DataType::Int64,
//         }
//     }
// }
//
// impl Tensor {
//     fn to_proto(self, name: String) -> TensorProto {
//         let data_type = self.ty.elem_type.to_proto();
//         let dims = self
//             .ty
//             .dims
//             .iter()
//             .map(|d| i64::try_from(*d).unwrap())
//             .collect();
//         let mut res = TensorProto {
//             name,
//             dims,
//             data_type: data_type.into(),
//             ..Default::default()
//         };
//
//         match self.data {
//             TensorData::I64(v) => res.int64_data = v,
//             TensorData::F32(v) => res.float_data = v,
//             TensorData::F64(v) => res.double_data = v,
//         };
//
//         res
//     }
// }
//
// impl BatchNormalization {
//     fn to_attrs(&self) -> Vec<AttributeProto> {
//         vec![
//         ]
//     }
// }
//
// impl Operator {
//     fn name_and_attrs(&self) -> (String, Vec<AttributeProto>) {
//         match self {
//             Operator::Add => ("Add".to_string(), vec![]),
//             Operator::BatchNormalization(ref batchnorm) => {
//
//             },
//             Operator::Conv => todo!(),
//             Operator::Gemm => todo!(),
//             Operator::MatMul => todo!(),
//             Operator::Relu => todo!(),
//             Operator::Reshape => todo!(),
//             Operator::Sigmoid => todo!(),
//             Operator::Softmax => todo!(),
//             Operator::Transpose => todo!(),
//         }
//     }
// }
