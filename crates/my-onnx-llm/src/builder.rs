use my_onnx::onnx::model::ExternalTensorRef;
use my_onnx::onnx::model::Graph;
use my_onnx::onnx::model::Node;
use my_onnx::onnx::model::ValueId;
use my_onnx::onnx::model::ValueInfo;
use my_onnx::onnx::operator::*;
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::DataType;
use my_onnx::tensor::types::FloatType;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::ResolvedTensorType;
use my_onnx::tensor::types::SIntType;
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

    pub fn external_initializer(&mut self, name: &str, r: ExternalTensorRef) -> ValueId {
        let ty = ResolvedTensorType::new(r.elem_type, r.dims.clone());
        let value = self.graph.values.alloc(ValueInfo {
            name: name.to_string(),
            ty: Some(TensorType::Resolved(ty)),
        });
        self.graph.set_external_ref(value, r);
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

    /// SiLU activation: `x * sigmoid(x)`. Equivalent to ONNX Swish with `alpha = 1.0`.
    pub fn silu(&mut self, name: &str, x: ValueId) -> ValueId {
        let out = self.alloc_value(name);
        self.add_node(name, Operator::Swish(Swish { alpha: 1.0 }), vec![x], out);
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

    pub fn i64_initializer(&mut self, name: &str, values: Vec<i64>) -> ValueId {
        let len = values.len();
        let t = Tensor::new(
            ResolvedTensorDims::new(&[len]),
            TensorData::SInt(SIntType::I64, values),
        )
        .unwrap();
        self.initializer(name, t)
    }

    /// Build the LLaMA-style RoPE cos/sin tables as f32 initializers of shape
    /// `[max_seq, head_dim]`. Each row `p` holds the cos/sin of the angles
    /// `p * inv_freq[j % (head_dim/2)]` where `inv_freq[i] = 1 / base^(2i/head_dim)`.
    /// The dim layout matches the half-split rotation in [`Self::rope`]:
    /// dim `j` and `j + head_dim/2` share the same frequency.
    pub fn rope_table(
        &mut self,
        name: &str,
        max_seq: usize,
        head_dim: usize,
        base: f32,
    ) -> (ValueId, ValueId) {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| 1.0 / (base as f64).powf(2.0 * i as f64 / head_dim as f64))
            .collect();
        let mut cos_data = Vec::with_capacity(max_seq * head_dim);
        let mut sin_data = Vec::with_capacity(max_seq * head_dim);
        for p in 0..max_seq {
            for j in 0..head_dim {
                let angle = p as f64 * inv_freq[j % half];
                cos_data.push(angle.cos());
                sin_data.push(angle.sin());
            }
        }
        let cos_t = Tensor::new(
            ResolvedTensorDims::new(&[max_seq, head_dim]),
            TensorData::Float(FloatType::F32, cos_data),
        )
        .unwrap();
        let sin_t = Tensor::new(
            ResolvedTensorDims::new(&[max_seq, head_dim]),
            TensorData::Float(FloatType::F32, sin_data),
        )
        .unwrap();
        let cos = self.initializer(&format!("{name}_cos"), cos_t);
        let sin = self.initializer(&format!("{name}_sin"), sin_t);
        (cos, sin)
    }

    /// Apply rotary position embedding to `x` of shape `[..., head_dim]` using
    /// pre-positioned `cos` and `sin` tables (broadcastable to `x`'s shape).
    /// Implements the standard half-split rotation:
    ///     x_low  = x[..., :head_dim/2]
    ///     x_high = x[..., head_dim/2:]
    ///     rotated = concat(-x_high, x_low, axis=-1)
    ///     out = x * cos + rotated * sin
    pub fn rope(
        &mut self,
        name: &str,
        x: ValueId,
        cos: ValueId,
        sin: ValueId,
        head_dim: usize,
    ) -> ValueId {
        let half = (head_dim / 2) as i64;
        let head_dim_i = head_dim as i64;
        let starts_low = self.i64_initializer(&format!("{name}_starts_low"), vec![0]);
        let ends_low = self.i64_initializer(&format!("{name}_ends_low"), vec![half]);
        let starts_high = self.i64_initializer(&format!("{name}_starts_high"), vec![half]);
        let ends_high = self.i64_initializer(&format!("{name}_ends_high"), vec![head_dim_i]);
        let axes = self.i64_initializer(&format!("{name}_axes"), vec![-1]);
        let x_low = self.slice(
            &format!("{name}_low"),
            x,
            starts_low,
            ends_low,
            Some(axes),
            None,
        );
        let x_high = self.slice(
            &format!("{name}_high"),
            x,
            starts_high,
            ends_high,
            Some(axes),
            None,
        );
        let neg_high = self.neg(&format!("{name}_neg_high"), x_high);
        let rotated = self.concat(&format!("{name}_rot"), vec![neg_high, x_low], -1);
        let x_cos = self.mul(&format!("{name}_xcos"), x, cos);
        let r_sin = self.mul(&format!("{name}_rsin"), rotated, sin);
        self.add(name, x_cos, r_sin)
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
