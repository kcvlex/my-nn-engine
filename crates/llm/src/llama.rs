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

fn load_weight_t(b: &mut Builder, name: &str, w: &WeightRef) -> ValueId {
    let id = b.load_weight(name, w.weight.clone(), w.scale.clone());
    b.transpose(&format!("{name}_t"), id, vec![1, 0])
}

#[derive(Debug, Clone, Default, TypedBuilder)]
#[builder(field_defaults(default))]
pub struct LlamaOptions {
    /// Prefill chunk length. `None` = decode (single-token graph). `Some(n)`
    /// = prefill graph that consumes `n` tokens per step.
    #[builder(setter(strip_option))]
    pub prefill_len: Option<usize>,
    /// If true, KV cache is stored as INT8 with per-token BF16 scale.
    /// Halves KV cache footprint at small accuracy cost.
    pub quant_kv_cache: bool,
    /// If true, lay out KV cache as `[sink][ring]` (StreamingLLM): K is stored
    /// unrotated and re-rotated against per-slot recency ranks each step.
    /// Mutually exclusive with `quant_kv_cache` for now.
    pub streaming_kv: bool,
}

pub struct LlamaGraph {
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

pub struct LlamaWeights {
    pub embed_tokens: WeightRef,
    pub lm_head: WeightRef,
    pub final_norm: ExternalTensorRef,
    pub layers: Vec<LlamaLayerWeights>,
}

pub struct LlamaLayerWeights {
    pub input_norm: ExternalTensorRef,
    pub post_norm: ExternalTensorRef,
    pub q_proj: WeightRef,
    pub k_proj: WeightRef,
    pub v_proj: WeightRef,
    pub o_proj: WeightRef,
    pub gate_proj: WeightRef,
    pub up_proj: WeightRef,
    pub down_proj: WeightRef,
}

impl LlamaWeights {
    pub fn from_hf(hf: &HfWeights, num_layers: usize) -> Result<Self, HfWeightsError> {
        let layers = (0..num_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                Ok(LlamaLayerWeights {
                    input_norm: hf.external_ref(&format!("{p}.input_layernorm.weight"))?,
                    post_norm: hf.external_ref(&format!("{p}.post_attention_layernorm.weight"))?,
                    q_proj: hf.weight_ref(&format!("{p}.self_attn.q_proj.weight"))?,
                    k_proj: hf.weight_ref(&format!("{p}.self_attn.k_proj.weight"))?,
                    v_proj: hf.weight_ref(&format!("{p}.self_attn.v_proj.weight"))?,
                    o_proj: hf.weight_ref(&format!("{p}.self_attn.o_proj.weight"))?,
                    gate_proj: hf.weight_ref(&format!("{p}.mlp.gate_proj.weight"))?,
                    up_proj: hf.weight_ref(&format!("{p}.mlp.up_proj.weight"))?,
                    down_proj: hf.weight_ref(&format!("{p}.mlp.down_proj.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, HfWeightsError>>()?;
        Ok(Self {
            embed_tokens: hf.weight_ref("model.embed_tokens.weight")?,
            lm_head: hf.weight_ref("lm_head.weight")?,
            final_norm: hf.external_ref("model.norm.weight")?,
            layers,
        })
    }
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
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    hidden: usize,
    max_seq_len: usize,
    seq_q: usize,
    is_prefill: bool,
    eps: f64,
    activation_ty: DataType,
    quant_kv_cache: bool,
    scale_ty: FloatType,
    streaming: Option<StreamingInputs>,
}

pub fn build_llama(
    config: &HfConfig,
    weights: &LlamaWeights,
    max_seq_len: usize,
    options: &LlamaOptions,
) -> LlamaGraph {
    let mode = match options.prefill_len {
        Some(len) => Mode::Prefill { len },
        None => Mode::Decode,
    };
    build_llama_inner(config, weights, max_seq_len, mode, options)
}

fn build_llama_inner(
    config: &HfConfig,
    weights: &LlamaWeights,
    max_seq_len: usize,
    mode: Mode,
    options: &LlamaOptions,
) -> LlamaGraph {
    assert!(max_seq_len <= config.max_position_embeddings);
    if let Mode::Prefill { len } = mode {
        assert!(
            len <= max_seq_len,
            "prefill_len ({len}) must be <= max_seq_len ({max_seq_len})"
        );
    }
    assert_eq!(config.hidden_act, "silu");
    assert_eq!(
        config.hidden_size,
        config.num_attention_heads * config.head_dim()
    );
    assert!(config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads));
    assert_eq!(weights.layers.len(), config.num_hidden_layers);

    let weight_float_ty = match weights.embed_tokens.scale.as_ref() {
        Some(scale) => match scale.elem_type {
            DataType::Float(t) => t,
            _ => panic!("expected float scale"),
        },
        None => match weights.embed_tokens.weight.elem_type {
            DataType::Float(t) => t,
            _ => panic!("expected float weights or scale"),
        },
    };
    let i64_ty = DataType::SInt(SIntType::I64);
    let head_dim = config.head_dim();
    let seq_q = mode.seq_q();
    let is_prefill = mode.is_prefill();

    let mut b = Builder::new(if is_prefill { "llama_prefill" } else { "llama" });

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

    let embed_w = b.load_weight(
        "model.embed_tokens.weight",
        weights.embed_tokens.weight.clone(),
        weights.embed_tokens.scale.clone(),
    );
    let mut x = b.gather("embed", embed_w, input_ids, 0);

    // RoPE table, shared across layers
    let (cos_table, sin_table) = b.rope_table(
        "rope",
        max_seq_len,
        head_dim,
        config.rope_theta,
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
        num_q_heads: config.num_attention_heads,
        num_kv_heads: config.num_key_value_heads,
        head_dim,
        hidden: config.hidden_size,
        max_seq_len,
        seq_q,
        is_prefill,
        eps: config.rms_norm_eps,
        activation_ty: DataType::Float(weight_float_ty),
        quant_kv_cache: options.quant_kv_cache,
        scale_ty: weight_float_ty,
        streaming,
    };

    let mut kv_cache_names = Vec::new();

    for (li, lw) in weights.layers.iter().enumerate() {
        let prefix = format!("model.layers.{li}");
        let (x_, kv_cache) = build_layer(&mut b, &ctx, &prefix, x, lw);
        x = x_;
        kv_cache_names.push(kv_cache);
    }

    let final_norm_w = b.external_initializer("model.norm.weight", weights.final_norm.clone());
    let final_norm = b.rms_norm("final_norm", x, final_norm_w, -1, ctx.eps);

    let lm_head_w = load_weight_t(&mut b, "lm_head.weight", &weights.lm_head);
    let logits = b.matmul("lm_head", final_norm, lm_head_w);
    b.output(logits);

    LlamaGraph {
        graph: b.graph,
        input_ids,
        position_id,
        past_len,
        active_seq_kv,
        logits,
        kv_cache_names,
        streaming,
    }
}

fn build_layer(
    b: &mut Builder,
    ctx: &LayerCtx,
    prefix: &str,
    x_in: ValueId,
    lw: &LlamaLayerWeights,
) -> (ValueId, KVCache) {
    let (attn_out, kv_cache) = build_attention(b, ctx, prefix, x_in, lw);
    let attn_residual = b.add(&format!("{prefix}_attn_resid"), x_in, attn_out);
    let mlp_out = build_mlp(b, ctx, prefix, attn_residual, lw);
    let final_out = b.add(&format!("{prefix}_mlp_resid"), attn_residual, mlp_out);
    (final_out, kv_cache)
}

fn build_attention(
    b: &mut Builder,
    ctx: &LayerCtx,
    prefix: &str,
    x_in: ValueId,
    lw: &LlamaLayerWeights,
) -> (ValueId, KVCache) {
    // Pre-attention RMSNorm
    let in_norm_w = b.external_initializer(
        &format!("{prefix}.input_layernorm.weight"),
        lw.input_norm.clone(),
    );
    let n1 = b.rms_norm(
        &format!("{prefix}_input_norm"),
        x_in,
        in_norm_w,
        -1,
        ctx.eps,
    );

    let q_w = load_weight_t(b, &format!("{prefix}.self_attn.q_proj.weight"), &lw.q_proj);
    let k_w = load_weight_t(b, &format!("{prefix}.self_attn.k_proj.weight"), &lw.k_proj);
    let v_w = load_weight_t(b, &format!("{prefix}.self_attn.v_proj.weight"), &lw.v_proj);
    let q = b.matmul(&format!("{prefix}_q_proj"), n1, q_w);
    let k = b.matmul(&format!("{prefix}_k_proj"), n1, k_w);
    let v = b.matmul(&format!("{prefix}_v_proj"), n1, v_w);

    // Reshape and transpose to [1, H, S, D] (S = 1 for decode, prefill_len for prefill)
    let q_shape = b.i64_initializer(
        &format!("{prefix}_q_shape"),
        vec![
            1,
            ctx.seq_q as i64,
            ctx.num_q_heads as i64,
            ctx.head_dim as i64,
        ],
    );
    let kv_shape = b.i64_initializer(
        &format!("{prefix}_kv_shape"),
        vec![
            1,
            ctx.seq_q as i64,
            ctx.num_kv_heads as i64,
            ctx.head_dim as i64,
        ],
    );
    let q = b.reshape(&format!("{prefix}_q_rs"), q, q_shape);
    let k = b.reshape(&format!("{prefix}_k_rs"), k, kv_shape);
    let v = b.reshape(&format!("{prefix}_v_rs"), v, kv_shape);
    let q = b.transpose(&format!("{prefix}_q_tr"), q, vec![0, 2, 1, 3]);
    let k = b.transpose(&format!("{prefix}_k_tr"), k, vec![0, 2, 1, 3]);
    let v = b.transpose(&format!("{prefix}_v_tr"), v, vec![0, 2, 1, 3]);

    // Q is always rotated per-token via the absolute (or recency-rank, in
    // streaming) position. K is rotated only in the legacy path; in streaming
    // we keep the cache unrotated and re-rotate the whole K cache against
    // per-slot recency ranks before attention.
    let q = b.rope(
        &format!("{prefix}_q_rope"),
        q,
        ctx.cos_4d,
        ctx.sin_4d,
        ctx.head_dim,
    );
    let k = if ctx.streaming.is_none() {
        b.rope(
            &format!("{prefix}_k_rope"),
            k,
            ctx.cos_4d,
            ctx.sin_4d,
            ctx.head_dim,
        )
    } else {
        k
    };

    let (k_updated, v_updated, kv_scales, kv_cache) = apply_kv_cache(b, ctx, prefix, k, v);

    let scale = (ctx.head_dim as f32).sqrt().recip();
    let attn_out = match (ctx.streaming, ctx.quant_kv_cache) {
        (Some(s), true) => b.attention_streaming(
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
            let k_recomputed = b.rope_fused(
                &format!("{prefix}_k_rope_recompute"),
                k_updated,
                ctx.cos_table,
                ctx.sin_table,
                s.kv_position,
                ctx.head_dim,
            );
            b.attention_streaming(
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
        (None, _) => b.attention_quant(
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

    let attn_out = b.transpose(&format!("{prefix}_attn_tr"), attn_out, vec![0, 2, 1, 3]);
    let attn_back_shape = b.i64_initializer(
        &format!("{prefix}_attn_shape"),
        vec![1, ctx.seq_q as i64, ctx.hidden as i64],
    );
    let attn_out = b.reshape(&format!("{prefix}_attn_rs"), attn_out, attn_back_shape);

    let o_w = load_weight_t(b, &format!("{prefix}.self_attn.o_proj.weight"), &lw.o_proj);
    let o = b.matmul(&format!("{prefix}_o_proj"), attn_out, o_w);

    (o, kv_cache)
}

fn apply_kv_cache(
    b: &mut Builder,
    ctx: &LayerCtx,
    prefix: &str,
    k: ValueId,
    v: ValueId,
) -> (ValueId, ValueId, Option<(ValueId, ValueId)>, KVCache) {
    let k_cache_name = format!("{prefix}.past_key");
    let v_cache_name = format!("{prefix}.past_value");
    let cache_dims = [1, ctx.num_kv_heads, ctx.max_seq_len, ctx.head_dim];
    let cache_dtype = if ctx.quant_kv_cache {
        DataType::SInt(SIntType::I8)
    } else {
        ctx.activation_ty
    };
    let k_cache = b.input(&k_cache_name, cache_dtype, &cache_dims);
    let v_cache = b.input(&v_cache_name, cache_dtype, &cache_dims);
    let cache_elem_bytes = cache_dtype.bit_width() / 8;
    let bytes_per_buffer = ctx.num_kv_heads * ctx.max_seq_len * ctx.head_dim * cache_elem_bytes;

    let (k_updated, v_updated, kv_scales, kv_cache_scale) = if ctx.quant_kv_cache {
        let scale_dtype = DataType::Float(ctx.scale_ty);
        let scale_dims = [1, ctx.num_kv_heads, ctx.max_seq_len];
        let k_scale_name = format!("{prefix}.past_key_scale");
        let v_scale_name = format!("{prefix}.past_value_scale");
        let k_scale = b.input(&k_scale_name, scale_dtype, &scale_dims);
        let v_scale = b.input(&v_scale_name, scale_dtype, &scale_dims);
        let (k_updated, v_updated) = if let Some(s) = ctx.streaming {
            let ring = (s.ring_sink, s.ring_window, s.ring_start);
            (
                b.quantizing_kv_cache_update_streaming(
                    &format!("{prefix}_k_update"),
                    k_cache,
                    k_scale,
                    k,
                    ctx.past_len,
                    ring,
                ),
                b.quantizing_kv_cache_update_streaming(
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
                b.quantizing_kv_cache_update(
                    &format!("{prefix}_k_update"),
                    k_cache,
                    k_scale,
                    k,
                    ctx.past_len,
                ),
                b.quantizing_kv_cache_update(
                    &format!("{prefix}_v_update"),
                    v_cache,
                    v_scale,
                    v,
                    ctx.past_len,
                ),
            )
        };
        let scale_elem_bytes = scale_dtype.bit_width() / 8;
        let bytes_per_scale = ctx.num_kv_heads * ctx.max_seq_len * scale_elem_bytes;
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
        let k_updated = b.kv_cache_update_streaming(
            &format!("{prefix}_k_update"),
            k_cache,
            k,
            ctx.past_len,
            ring,
        );
        let v_updated = b.kv_cache_update_streaming(
            &format!("{prefix}_v_update"),
            v_cache,
            v,
            ctx.past_len,
            ring,
        );
        (k_updated, v_updated, None, None)
    } else {
        let k_updated = b.kv_cache_update(&format!("{prefix}_k_update"), k_cache, k, ctx.past_len);
        let v_updated = b.kv_cache_update(&format!("{prefix}_v_update"), v_cache, v, ctx.past_len);
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

fn build_mlp(
    b: &mut Builder,
    ctx: &LayerCtx,
    prefix: &str,
    x: ValueId,
    lw: &LlamaLayerWeights,
) -> ValueId {
    // Pre-MLP RMSNorm
    let post_norm_w = b.external_initializer(
        &format!("{prefix}.post_attention_layernorm.weight"),
        lw.post_norm.clone(),
    );
    let n2 = b.rms_norm(&format!("{prefix}_post_norm"), x, post_norm_w, -1, ctx.eps);

    // SwiGLU MLP: down(silu(gate(x)) * up(x))
    let gate_w = load_weight_t(b, &format!("{prefix}.mlp.gate_proj.weight"), &lw.gate_proj);
    let up_w = load_weight_t(b, &format!("{prefix}.mlp.up_proj.weight"), &lw.up_proj);
    let down_w = load_weight_t(b, &format!("{prefix}.mlp.down_proj.weight"), &lw.down_proj);

    let gate = b.matmul(&format!("{prefix}_gate_proj"), n2, gate_w);
    let gate = b.silu(&format!("{prefix}_silu"), gate);
    let up = b.matmul(&format!("{prefix}_up_proj"), n2, up_w);
    let mlp_in = b.mul(&format!("{prefix}_swiglu"), gate, up);
    b.matmul(&format!("{prefix}_down_proj"), mlp_in, down_w)
}
