use my_nn_engine::graph::ExternalTensorRef;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::ValueId;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::FloatType;
use my_nn_engine::tensor::types::SIntType;
use typed_builder::TypedBuilder;

use crate::builder::Builder;
use crate::hf_config::HfConfig;
use crate::hf_weights::HfWeights;
use crate::hf_weights::HfWeightsError;
use crate::hf_weights::WeightRef;
use crate::session::KVCache;
use crate::session::KVScale;

/// A linear projection: matmul against a (transposed) weight, plus an optional
/// bias. `node` is the graph node name for the matmul; `weight_name` is the
/// on-disk tensor name.
#[derive(Debug, Clone)]
pub struct Proj {
    pub node: String,
    pub weight_name: String,
    pub weight: WeightRef,
    pub bias: Option<WeightRef>,
}

#[derive(Debug, Clone)]
pub enum NormSpec {
    Rms {
        node: String,
        weight_name: String,
        eps: f64,
        weight: ExternalTensorRef,
    },
}

#[derive(Debug, Clone)]
pub enum RopeSpec {
    Standard { theta: f32 },
}

#[derive(Debug, Clone)]
pub enum AttnSpec {
    Gqa {
        prefix: String,
        q: Proj,
        k: Proj,
        v: Proj,
        o: Proj,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        rope: RopeSpec,
        sliding_window: Option<usize>,
    },
}

#[derive(Debug, Clone)]
pub enum MlpSpec {
    SwiGlu { gate: Proj, up: Proj, down: Proj },
}

#[derive(Debug, Clone)]
pub enum HeadSpec {
    Separate(WeightRef),
}

#[derive(Debug, Clone)]
pub struct LayerSpec {
    pub norm1: NormSpec,
    pub attn: AttnSpec,
    pub norm2: NormSpec,
    pub mlp: MlpSpec,
}

pub struct ModelSpec {
    pub embed_tokens: WeightRef,
    pub n_layers: usize,
    pub layers: Vec<LayerSpec>,
    pub final_norm: NormSpec,
    pub head: HeadSpec,
    pub rope_theta: f32,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden: usize,
    pub eps: f64,
}

struct DecoderBuilder {
    b: Builder,
}

impl DecoderBuilder {
    fn load_weight_t(&mut self, name: &str, w: &WeightRef) -> ValueId {
        let id = self.b.load_weight(name, w.weight.clone(), w.scale.clone());
        self.b.transpose(&format!("{name}_t"), id, vec![1, 0])
    }

    fn linear(&mut self, x: ValueId, proj: &Proj) -> ValueId {
        let w = self.load_weight_t(&proj.weight_name, &proj.weight);
        let out = self.b.matmul(&proj.node, x, w);
        match &proj.bias {
            Some(bias) => {
                let bias_id = self.b.load_weight(
                    &format!("{}.bias", proj.weight_name.trim_end_matches(".weight")),
                    bias.weight.clone(),
                    bias.scale.clone(),
                );
                self.b.add(&format!("{}_bias", proj.node), out, bias_id)
            }
            None => out,
        }
    }

    fn build_norm(&mut self, x: ValueId, norm: &NormSpec) -> ValueId {
        match norm {
            NormSpec::Rms {
                node,
                weight_name,
                eps,
                weight,
            } => {
                let w = self.b.external_initializer(weight_name, weight.clone());
                self.b.rms_norm(node, x, w, -1, *eps)
            }
        }
    }

    fn build_layer(
        &mut self,
        ctx: &LayerCtx,
        x_in: ValueId,
        layer: &LayerSpec,
    ) -> (ValueId, KVCache) {
        let prefix = match &layer.attn {
            AttnSpec::Gqa { prefix, .. } => prefix.clone(),
        };
        let (attn_out, kv_cache) = self.build_attention(ctx, x_in, &layer.norm1, &layer.attn);
        let attn_residual = self.b.add(&format!("{prefix}_attn_resid"), x_in, attn_out);
        let mlp_out = self.build_mlp(&prefix, attn_residual, &layer.norm2, &layer.mlp);
        let final_out = self
            .b
            .add(&format!("{prefix}_mlp_resid"), attn_residual, mlp_out);
        (final_out, kv_cache)
    }

    fn build_attention(
        &mut self,
        ctx: &LayerCtx,
        x_in: ValueId,
        norm: &NormSpec,
        attn: &AttnSpec,
    ) -> (ValueId, KVCache) {
        let AttnSpec::Gqa {
            prefix,
            q,
            k,
            v,
            o,
            n_heads,
            n_kv_heads,
            head_dim,
            rope: _,
            sliding_window: _,
        } = attn;
        let num_q_heads = *n_heads;
        let num_kv_heads = *n_kv_heads;
        let head_dim = *head_dim;

        // Pre-attention RMSNorm
        let n1 = self.build_norm(x_in, norm);

        let q = self.linear(n1, q);
        let k = self.linear(n1, k);
        let v = self.linear(n1, v);

        // Reshape and transpose to [1, H, S, D] (S = 1 for decode, prefill_len for prefill)
        let q_shape = self.b.i64_initializer(
            &format!("{prefix}_q_shape"),
            vec![1, ctx.seq_q as i64, num_q_heads as i64, head_dim as i64],
        );
        let kv_shape = self.b.i64_initializer(
            &format!("{prefix}_kv_shape"),
            vec![1, ctx.seq_q as i64, num_kv_heads as i64, head_dim as i64],
        );
        let q = self.b.reshape(&format!("{prefix}_q_rs"), q, q_shape);
        let k = self.b.reshape(&format!("{prefix}_k_rs"), k, kv_shape);
        let v = self.b.reshape(&format!("{prefix}_v_rs"), v, kv_shape);
        let q = self
            .b
            .transpose(&format!("{prefix}_q_tr"), q, vec![0, 2, 1, 3]);
        let k = self
            .b
            .transpose(&format!("{prefix}_k_tr"), k, vec![0, 2, 1, 3]);
        let v = self
            .b
            .transpose(&format!("{prefix}_v_tr"), v, vec![0, 2, 1, 3]);

        // Q is always rotated per-token via the absolute (or recency-rank, in
        // streaming) position. K is rotated only in the legacy path; in streaming
        // we keep the cache unrotated and re-rotate the whole K cache against
        // per-slot recency ranks before attention.
        let q = self.b.rope(
            &format!("{prefix}_q_rope"),
            q,
            ctx.cos_4d,
            ctx.sin_4d,
            head_dim,
        );
        let k = if ctx.streaming.is_none() {
            self.b.rope(
                &format!("{prefix}_k_rope"),
                k,
                ctx.cos_4d,
                ctx.sin_4d,
                head_dim,
            )
        } else {
            k
        };

        let (k_updated, v_updated, kv_scales, kv_cache) =
            self.apply_kv_cache(ctx, prefix, num_kv_heads, head_dim, k, v);

        let scale = (head_dim as f32).sqrt().recip();
        let attn_out = match (ctx.streaming, ctx.quant_kv_cache) {
            (Some(s), true) => self.b.attention_streaming(
                &format!("{prefix}_attn"),
                q,
                k_updated,
                v_updated,
                kv_scales,
                None,
                ctx.active_seq_kv,
                (s.ring_sink, s.ring_window, s.ring_start),
                Some((ctx.cos_table, ctx.sin_table, s.kv_position)),
                ctx.is_prefill,
                scale,
            ),
            (Some(s), false) => {
                // TODO: Fuse the rope recomputation into the attention kernel to save memory and latency.
                let k_recomputed = self.b.rope_fused(
                    &format!("{prefix}_k_rope_recompute"),
                    k_updated,
                    ctx.cos_table,
                    ctx.sin_table,
                    s.kv_position,
                    head_dim,
                );
                self.b.attention_streaming(
                    &format!("{prefix}_attn"),
                    q,
                    k_recomputed,
                    v_updated,
                    kv_scales,
                    None,
                    ctx.active_seq_kv,
                    (s.ring_sink, s.ring_window, s.ring_start),
                    None,
                    ctx.is_prefill,
                    scale,
                )
            }
            (None, _) => self.b.attention_quant(
                &format!("{prefix}_attn"),
                q,
                k_updated,
                v_updated,
                kv_scales,
                None,
                Some(ctx.active_seq_kv),
                ctx.is_prefill,
                scale,
            ),
        };

        let attn_out = self
            .b
            .transpose(&format!("{prefix}_attn_tr"), attn_out, vec![0, 2, 1, 3]);
        let attn_back_shape = self.b.i64_initializer(
            &format!("{prefix}_attn_shape"),
            vec![1, ctx.seq_q as i64, (num_q_heads * head_dim) as i64],
        );
        let attn_out = self
            .b
            .reshape(&format!("{prefix}_attn_rs"), attn_out, attn_back_shape);

        let o = self.linear(attn_out, o);

        (o, kv_cache)
    }

    fn apply_kv_cache(
        &mut self,
        ctx: &LayerCtx,
        prefix: &str,
        num_kv_heads: usize,
        head_dim: usize,
        k: ValueId,
        v: ValueId,
    ) -> (ValueId, ValueId, Option<(ValueId, ValueId)>, KVCache) {
        let k_cache_name = format!("{prefix}.past_key");
        let v_cache_name = format!("{prefix}.past_value");
        let cache_dims = [1, num_kv_heads, ctx.max_seq_len, head_dim];
        let cache_dtype = if ctx.quant_kv_cache {
            DataType::SInt(SIntType::I8)
        } else {
            ctx.activation_ty
        };
        let k_cache = self.b.input(&k_cache_name, cache_dtype, &cache_dims);
        let v_cache = self.b.input(&v_cache_name, cache_dtype, &cache_dims);
        let cache_elem_bytes = cache_dtype.bit_width() / 8;
        let bytes_per_buffer = num_kv_heads * ctx.max_seq_len * head_dim * cache_elem_bytes;

        let (k_updated, v_updated, kv_scales, kv_cache_scale) = if ctx.quant_kv_cache {
            let scale_dtype = DataType::Float(ctx.scale_ty);
            let scale_dims = [1, num_kv_heads, ctx.max_seq_len];
            let k_scale_name = format!("{prefix}.past_key_scale");
            let v_scale_name = format!("{prefix}.past_value_scale");
            let k_scale = self.b.input(&k_scale_name, scale_dtype, &scale_dims);
            let v_scale = self.b.input(&v_scale_name, scale_dtype, &scale_dims);
            let (k_updated, v_updated) = if let Some(s) = ctx.streaming {
                let ring = (s.ring_sink, s.ring_window, s.ring_start);
                (
                    self.b.quantizing_kv_cache_update_streaming(
                        &format!("{prefix}_k_update"),
                        k_cache,
                        k_scale,
                        k,
                        ctx.past_len,
                        ring,
                    ),
                    self.b.quantizing_kv_cache_update_streaming(
                        &format!("{prefix}_v_update"),
                        v_cache,
                        v_scale,
                        v,
                        ctx.past_len,
                        ring,
                    ),
                )
            } else {
                (
                    self.b.quantizing_kv_cache_update(
                        &format!("{prefix}_k_update"),
                        k_cache,
                        k_scale,
                        k,
                        ctx.past_len,
                    ),
                    self.b.quantizing_kv_cache_update(
                        &format!("{prefix}_v_update"),
                        v_cache,
                        v_scale,
                        v,
                        ctx.past_len,
                    ),
                )
            };
            let scale_elem_bytes = scale_dtype.bit_width() / 8;
            let bytes_per_scale = num_kv_heads * ctx.max_seq_len * scale_elem_bytes;
            (
                k_updated,
                v_updated,
                Some((k_scale, v_scale)),
                Some(KVScale {
                    k_scale_name,
                    v_scale_name,
                    bytes_per_scale,
                }),
            )
        } else if let Some(s) = ctx.streaming {
            let ring = (s.ring_sink, s.ring_window, s.ring_start);
            let k_updated = self.b.kv_cache_update_streaming(
                &format!("{prefix}_k_update"),
                k_cache,
                k,
                ctx.past_len,
                ring,
            );
            let v_updated = self.b.kv_cache_update_streaming(
                &format!("{prefix}_v_update"),
                v_cache,
                v,
                ctx.past_len,
                ring,
            );
            (k_updated, v_updated, None, None)
        } else {
            let k_updated =
                self.b
                    .kv_cache_update(&format!("{prefix}_k_update"), k_cache, k, ctx.past_len);
            let v_updated =
                self.b
                    .kv_cache_update(&format!("{prefix}_v_update"), v_cache, v, ctx.past_len);
            (k_updated, v_updated, None, None)
        };

        let kv_cache = KVCache {
            k_name: k_cache_name,
            v_name: v_cache_name,
            bytes_per_buffer,
            scale: kv_cache_scale,
        };

        (k_updated, v_updated, kv_scales, kv_cache)
    }

    fn build_mlp(&mut self, prefix: &str, x: ValueId, norm: &NormSpec, mlp: &MlpSpec) -> ValueId {
        // Pre-MLP RMSNorm
        let n2 = self.build_norm(x, norm);

        match mlp {
            MlpSpec::SwiGlu { gate, up, down } => {
                // SwiGLU MLP: down(silu(gate(x)) * up(x))
                let gate = self.linear(n2, gate);
                let gate = self.b.silu(&format!("{prefix}_silu"), gate);
                let up = self.linear(n2, up);
                let mlp_in = self.b.mul(&format!("{prefix}_swiglu"), gate, up);
                self.linear(mlp_in, down)
            }
        }
    }
}

impl ModelSpec {
    pub fn from_hf(cfg: &HfConfig, hf: &HfWeights) -> Result<Self, HfWeightsError> {
        let qkv_bias = match cfg.model_type.as_str() {
            "llama" | "mistral" => false,
            "qwen2" => true,
            other => panic!("unsupported model_type: {other}"),
        };
        assert_eq!(cfg.hidden_act, "silu");
        assert_eq!(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim());
        assert!(cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads));

        let head_dim = cfg.head_dim();
        let proj = |hf: &HfWeights,
                    node: &str,
                    weight: &str,
                    with_bias: bool|
         -> Result<Proj, HfWeightsError> {
            Ok(Proj {
                node: node.to_string(),
                weight_name: weight.to_string(),
                weight: hf.weight_ref(weight)?,
                bias: if with_bias {
                    let base = weight.trim_end_matches(".weight");
                    Some(hf.weight_ref(&format!("{base}.bias"))?)
                } else {
                    None
                },
            })
        };

        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                Ok(LayerSpec {
                    norm1: NormSpec::Rms {
                        node: format!("{p}_input_norm"),
                        weight_name: format!("{p}.input_layernorm.weight"),
                        eps: cfg.rms_norm_eps,
                        weight: hf.external_ref(&format!("{p}.input_layernorm.weight"))?,
                    },
                    attn: AttnSpec::Gqa {
                        prefix: p.clone(),
                        q: proj(
                            hf,
                            &format!("{p}_q_proj"),
                            &format!("{p}.self_attn.q_proj.weight"),
                            qkv_bias,
                        )?,
                        k: proj(
                            hf,
                            &format!("{p}_k_proj"),
                            &format!("{p}.self_attn.k_proj.weight"),
                            qkv_bias,
                        )?,
                        v: proj(
                            hf,
                            &format!("{p}_v_proj"),
                            &format!("{p}.self_attn.v_proj.weight"),
                            qkv_bias,
                        )?,
                        o: proj(
                            hf,
                            &format!("{p}_o_proj"),
                            &format!("{p}.self_attn.o_proj.weight"),
                            false,
                        )?,
                        n_heads: cfg.num_attention_heads,
                        n_kv_heads: cfg.num_key_value_heads,
                        head_dim,
                        rope: RopeSpec::Standard {
                            theta: cfg.rope_theta,
                        },
                        sliding_window: None,
                    },
                    norm2: NormSpec::Rms {
                        node: format!("{p}_post_norm"),
                        weight_name: format!("{p}.post_attention_layernorm.weight"),
                        eps: cfg.rms_norm_eps,
                        weight: hf.external_ref(&format!("{p}.post_attention_layernorm.weight"))?,
                    },
                    mlp: MlpSpec::SwiGlu {
                        gate: proj(
                            hf,
                            &format!("{p}_gate_proj"),
                            &format!("{p}.mlp.gate_proj.weight"),
                            false,
                        )?,
                        up: proj(
                            hf,
                            &format!("{p}_up_proj"),
                            &format!("{p}.mlp.up_proj.weight"),
                            false,
                        )?,
                        down: proj(
                            hf,
                            &format!("{p}_down_proj"),
                            &format!("{p}.mlp.down_proj.weight"),
                            false,
                        )?,
                    },
                })
            })
            .collect::<Result<Vec<_>, HfWeightsError>>()?;

        Ok(Self {
            embed_tokens: hf.weight_ref("model.embed_tokens.weight")?,
            n_layers: cfg.num_hidden_layers,
            layers,
            final_norm: NormSpec::Rms {
                node: "final_norm".to_string(),
                weight_name: "model.norm.weight".to_string(),
                eps: cfg.rms_norm_eps,
                weight: hf.external_ref("model.norm.weight")?,
            },
            head: HeadSpec::Separate(hf.weight_ref("lm_head.weight")?),
            rope_theta: cfg.rope_theta,
            head_dim,
            n_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
        })
    }
}

#[derive(Debug, Clone, Default, TypedBuilder)]
#[builder(field_defaults(default))]
pub struct BuildOptions {
    #[builder(setter(strip_option))]
    pub prefill_len: Option<usize>,
    pub quant_kv_cache: bool,
    pub streaming_kv: bool,
}

pub struct DecoderGraph {
    pub graph: Graph,
    pub input_ids: ValueId,
    pub position_id: ValueId,
    pub past_len: ValueId,
    pub active_seq_kv: ValueId,
    pub logits: ValueId,
    pub kv_cache_names: Vec<KVCache>,
    pub streaming: Option<StreamingInputs>,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamingInputs {
    pub ring_sink: ValueId,
    pub ring_window: ValueId,
    pub ring_start: ValueId,
    pub kv_position: ValueId,
}

#[derive(Debug, Clone, Copy)]
pub enum Mode {
    Decode,
    Prefill { len: usize },
}

impl Mode {
    fn seq_q(&self) -> usize {
        match self {
            Mode::Decode => 1,
            Mode::Prefill { len } => *len,
        }
    }

    fn is_prefill(&self) -> bool {
        matches!(self, Mode::Prefill { .. })
    }
}

struct LayerCtx {
    cos_4d: ValueId,
    sin_4d: ValueId,
    cos_table: ValueId,
    sin_table: ValueId,
    past_len: ValueId,
    active_seq_kv: ValueId,
    max_seq_len: usize,
    seq_q: usize,
    is_prefill: bool,
    activation_ty: DataType,
    quant_kv_cache: bool,
    scale_ty: FloatType,
    streaming: Option<StreamingInputs>,
}

pub fn build_decoder(
    config: &HfConfig,
    spec: &ModelSpec,
    max_seq_len: usize,
    options: &BuildOptions,
) -> DecoderGraph {
    let mode = match options.prefill_len {
        Some(len) => Mode::Prefill { len },
        None => Mode::Decode,
    };
    build_decoder_inner(config, spec, max_seq_len, mode, options)
}

fn build_decoder_inner(
    config: &HfConfig,
    spec: &ModelSpec,
    max_seq_len: usize,
    mode: Mode,
    options: &BuildOptions,
) -> DecoderGraph {
    assert!(max_seq_len <= config.max_position_embeddings);
    if let Mode::Prefill { len } = mode {
        assert!(
            len <= max_seq_len,
            "prefill_len ({len}) must be <= max_seq_len ({max_seq_len})"
        );
    }
    assert_eq!(spec.layers.len(), spec.n_layers);

    let weight_float_ty = match spec.embed_tokens.scale.as_ref() {
        Some(scale) => match scale.elem_type {
            DataType::Float(t) => t,
            _ => panic!("expected float scale"),
        },
        None => match spec.embed_tokens.weight.elem_type {
            DataType::Float(t) => t,
            _ => panic!("expected float weights or scale"),
        },
    };
    let i64_ty = DataType::SInt(SIntType::I64);
    let head_dim = spec.head_dim;
    let seq_q = mode.seq_q();
    let is_prefill = mode.is_prefill();

    let mut b = {
        let phase = if is_prefill { "prefill" } else { "decode" };
        Builder::new(&format!("{}_{phase}", config.model_type))
    };

    let input_ids = b.input("input_ids", i64_ty, &[1, seq_q]);
    let position_id = b.input("position_id", i64_ty, &[seq_q]);
    let past_len = b.input("past_len", i64_ty, &[]);
    let active_seq_kv = b.input("active_seq_kv", i64_ty, &[]);
    let streaming = options.streaming_kv.then(|| StreamingInputs {
        ring_sink: b.input("ring_sink", i64_ty, &[]),
        ring_window: b.input("ring_window", i64_ty, &[]),
        ring_start: b.input("ring_start", i64_ty, &[]),
        kv_position: b.input("kv_position", i64_ty, &[max_seq_len]),
    });

    // RoPE table, shared across layers
    let (cos_table, sin_table) = b.rope_table(
        "rope",
        max_seq_len,
        head_dim,
        spec.rope_theta,
        weight_float_ty,
    );
    let cos_row = b.gather("rope_cos_row", cos_table, position_id, 0);
    let sin_row = b.gather("rope_sin_row", sin_table, position_id, 0);
    let rope_4d_shape = b.i64_initializer(
        "rope_row_4d_shape",
        vec![1, 1, seq_q as i64, head_dim as i64],
    );
    let cos_4d = b.reshape("rope_cos_4d", cos_row, rope_4d_shape);
    let sin_4d = b.reshape("rope_sin_4d", sin_row, rope_4d_shape);

    let ctx = LayerCtx {
        cos_4d,
        sin_4d,
        cos_table,
        sin_table,
        past_len,
        active_seq_kv,
        max_seq_len,
        seq_q,
        is_prefill,
        activation_ty: DataType::Float(weight_float_ty),
        quant_kv_cache: options.quant_kv_cache,
        scale_ty: weight_float_ty,
        streaming,
    };

    let mut db = DecoderBuilder { b };
    let mut kv_cache_names = Vec::new();

    let mut x = {
        let embed_w = db.b.load_weight(
            "model.embed_tokens.weight",
            spec.embed_tokens.weight.clone(),
            spec.embed_tokens.scale.clone(),
        );
        db.b.gather("embed", embed_w, input_ids, 0)
    };

    for layer in spec.layers.iter() {
        let (x_, kv_cache) = db.build_layer(&ctx, x, layer);
        x = x_;
        kv_cache_names.push(kv_cache);
    }

    let final_norm = db.build_norm(x, &spec.final_norm);

    let logits = match &spec.head {
        HeadSpec::Separate(lm_head) => {
            let lm_head_w = db.load_weight_t("lm_head.weight", lm_head);
            db.b.matmul("lm_head", final_norm, lm_head_w)
        }
    };
    db.b.output(logits);

    DecoderGraph {
        graph: db.b.graph,
        input_ids,
        position_id,
        past_len,
        active_seq_kv,
        logits,
        kv_cache_names,
        streaming,
    }
}
