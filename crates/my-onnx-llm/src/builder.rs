use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::onnx::operator::*;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::ResolvedTensorType;
use my_onnx::tensor::types::TensorType;
use my_onnx::tensor::Tensor;

pub struct Builder {
    pub graph: Graph,
}

impl Builder {
    pub fn new(name: &str) -> Self {
        Self {
            graph: Graph::empty_graph(name.to_string()),
        }
    }

    pub fn input(&mut self, name: &str, ty: DataType, dims: &[usize]) -> ValueId {
        let value = self.graph.values.alloc(ValueInfo {
            name: name.to_string(),
            ty: Some(TensorType::Resolved(ResolvedTensorType::new(
                ty,
                ResolvedTensorDims::new(dims),
            ))),
        });
        let node = self.graph.nodes.alloc(Node::create_node(
            vec![],
            vec![value],
            format!("Input_{name}"),
            Operator::Input(value),
        ));
        self.graph.inputs.push(node);
        value
    }

    pub fn output(&mut self, value: ValueId) {
        let name = self.graph.values[value].name.clone();
        let node = self.graph.nodes.alloc(Node::create_node(
            vec![Some(value)],
            vec![],
            format!("Output_{name}"),
            Operator::Output(value),
        ));
        self.graph.outputs.push(node);
    }

    pub fn initializer(&mut self, name: &str, tensor: Tensor) -> ValueId {
        let ty = tensor.tensor_type();
        let value = self.graph.values.alloc(ValueInfo {
            name: name.to_string(),
            ty: Some(TensorType::Resolved(ty)),
        });
        self.graph.set_initializer(value, tensor);
        value
    }

    fn alloc_value(&mut self, name: &str) -> ValueId {
        self.graph.values.alloc(ValueInfo {
            name: name.to_string(),
            ty: None,
        })
    }

    fn add_node(&mut self, name: &str, op: Operator, inputs: Vec<ValueId>, output: ValueId) {
        self.graph.nodes.alloc(Node::create_node(
            inputs.into_iter().map(Some).collect(),
            vec![output],
            name.to_string(),
            op,
        ));
    }

    pub fn gather(&mut self, name: &str, data: ValueId, indices: ValueId, axis: isize) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(
            name,
            Operator::Gather(Gather {
                axis: TensorIndex::new(axis),
            }),
            vec![data, indices],
            out,
        );
        out
    }

    pub fn matmul(&mut self, name: &str, a: ValueId, b: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::MatMul, vec![a, b], out);
        out
    }

    pub fn rms_norm(
        &mut self,
        name: &str,
        x: ValueId,
        scale: ValueId,
        axis: i64,
        epsilon: f64,
    ) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(
            name,
            Operator::RMSNormalization(RMSNormalization {
                axis: TensorIndex::new(axis as isize),
                epsilon,
            }),
            vec![x, scale],
            out,
        );
        out
    }

    pub fn add(&mut self, name: &str, a: ValueId, b: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Add, vec![a, b], out);
        out
    }

    pub fn mul(&mut self, name: &str, a: ValueId, b: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Mul, vec![a, b], out);
        out
    }

    pub fn neg(&mut self, name: &str, x: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Neg, vec![x], out);
        out
    }

    pub fn sigmoid(&mut self, name: &str, x: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Sigmoid, vec![x], out);
        out
    }

    pub fn reshape(&mut self, name: &str, x: ValueId, shape: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Reshape, vec![x, shape], out);
        out
    }

    pub fn transpose(&mut self, name: &str, x: ValueId, perm: Vec<usize>) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(
            name,
            Operator::Transpose(Transpose { perm: Some(perm) }),
            vec![x],
            out,
        );
        out
    }

    pub fn concat(&mut self, name: &str, inputs: Vec<ValueId>, axis: isize) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(
            name,
            Operator::Concat(Concat {
                axis: TensorIndex::new(axis),
            }),
            inputs,
            out,
        );
        out
    }

    pub fn slice(
        &mut self,
        name: &str,
        x: ValueId,
        starts: ValueId,
        ends: ValueId,
        axes: Option<ValueId>,
        steps: Option<ValueId>,
    ) -> ValueId {
        let out = self.alloc_value(name);
        let inputs: Vec<Option<ValueId>> = match (axes, steps) {
            (None, None) => vec![Some(x), Some(starts), Some(ends)],
            (Some(a), None) => vec![Some(x), Some(starts), Some(ends), Some(a)],
            (a, Some(s)) => vec![Some(x), Some(starts), Some(ends), a, Some(s)],
        };
        self.graph.nodes.alloc(Node::create_node(
            inputs,
            vec![out],
            name.to_string(),
            Operator::Slice,
        ));
        out
    }

    pub fn expand(&mut self, name: &str, x: ValueId, shape: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Expand, vec![x, shape], out);
        out
    }

    pub fn kv_cache_update(
        &mut self,
        name: &str,
        cache: ValueId,
        new: ValueId,
        offset: ValueId,
    ) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::KVCacheUpdate, vec![cache, new, offset], out);
        out
    }

    pub fn attention(
        &mut self,
        name: &str,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        mask: Option<ValueId>,
        active_seq_kv: Option<ValueId>,
        is_causal: bool,
        scale: f32,
    ) -> ValueId {
        let out = self.alloc_value(name);
        let inputs: Vec<Option<ValueId>> = match (mask, active_seq_kv) {
            (None, None) => vec![Some(q), Some(k), Some(v)],
            (Some(m), None) => vec![Some(q), Some(k), Some(v), Some(m)],
            (m, Some(a)) => vec![Some(q), Some(k), Some(v), m, Some(a)],
        };
        self.graph.nodes.alloc(Node::create_node(
            inputs,
            vec![out],
            name.to_string(),
            Operator::Attention(Attention { is_causal, scale }),
        ));
        out
    }
}
