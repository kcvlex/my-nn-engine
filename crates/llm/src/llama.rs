use my_nn_engine::graph::ExternalTensorRef;
use my_nn_engine::graph::Graph;
use my_nn_engine::graph::ValueId;
use my_nn_engine::tensor::types::DataType;
use my_nn_engine::tensor::types::SIntType;

use crate::builder::Builder;
use crate::hf_config::HfConfig;
use crate::hf_weights::HfWeights;
use crate::hf_weights::HfWeightsError;
use crate::session::KVCache;

pub struct LlamaGraph {
    pub graph: Graph,
    pub input_ids: ValueId,
    pub position_id: ValueId,
    pub past_len: ValueId,
    pub active_seq_kv: ValueId,
    pub logits: ValueId,
    pub kv_cache_names: Vec<KVCache>,
}

pub struct LlamaWeights {
    pub embed_tokens: ExternalTensorRef,
    pub lm_head: ExternalTensorRef,
    pub final_norm: ExternalTensorRef,
    pub layers: Vec<LlamaLayerWeights>,
}

pub struct LlamaLayerWeights {
    pub input_norm: ExternalTensorRef,
    pub post_norm: ExternalTensorRef,
    pub q_proj: ExternalTensorRef,
    pub k_proj: ExternalTensorRef,
    pub v_proj: ExternalTensorRef,
    pub o_proj: ExternalTensorRef,
    pub gate_proj: ExternalTensorRef,
    pub up_proj: ExternalTensorRef,
    pub down_proj: ExternalTensorRef,
}

impl LlamaWeights {
    pub fn from_hf(hf: &HfWeights, num_layers: usize) -> Result<Self, HfWeightsError> {
        let layers = (0..num_layers)
            .map(|i| {
                let p = format!("model.layers.{i}");
                Ok(LlamaLayerWeights {
                    input_norm: hf.external_ref(&format!("{p}.input_layernorm.weight"))?,
                    post_norm: hf.external_ref(&format!("{p}.post_attention_layernorm.weight"))?,
                    q_proj: hf.external_ref(&format!("{p}.self_attn.q_proj.weight"))?,
                    k_proj: hf.external_ref(&format!("{p}.self_attn.k_proj.weight"))?,
                    v_proj: hf.external_ref(&format!("{p}.self_attn.v_proj.weight"))?,
                    o_proj: hf.external_ref(&format!("{p}.self_attn.o_proj.weight"))?,
                    gate_proj: hf.external_ref(&format!("{p}.mlp.gate_proj.weight"))?,
                    up_proj: hf.external_ref(&format!("{p}.mlp.up_proj.weight"))?,
                    down_proj: hf.external_ref(&format!("{p}.mlp.down_proj.weight"))?,
                })
            })
            .collect::<Result<Vec<_>, HfWeightsError>>()?;
        Ok(Self {
            embed_tokens: hf.external_ref("model.embed_tokens.weight")?,
            lm_head: hf.external_ref("lm_head.weight")?,
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
}

pub fn build_llama(config: &HfConfig, weights: &LlamaWeights, max_seq_len: usize) -> LlamaGraph {
    build_llama_inner(config, weights, max_seq_len, Mode::Decode)
}

pub fn build_llama_prefill(
    config: &HfConfig,
    weights: &LlamaWeights,
    max_seq_len: usize,
    prefill_len: usize,
) -> LlamaGraph {
    build_llama_inner(
        config,
        weights,
        max_seq_len,
        Mode::Prefill { len: prefill_len },
    )
}

fn build_llama_inner(
    config: &HfConfig,
    weights: &LlamaWeights,
    max_seq_len: usize,
    mode: Mode,
) -> LlamaGraph {
    assert!(max_seq_len <= config.max_position_embeddings);
    assert_eq!(config.hidden_act, "silu");
    assert_eq!(
        config.hidden_size,
        config.num_attention_heads * config.head_dim()
    );
    assert!(config
        .num_attention_heads
        .is_multiple_of(config.num_key_value_heads));
    assert_eq!(weights.layers.len(), config.num_hidden_layers);

    let weight_float_ty = match weights.embed_tokens.elem_type {
        DataType::Float(t) => t,
        _ => panic!("expected float weights"),
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

    let embed_w = b.external_initializer("model.embed_tokens.weight", weights.embed_tokens.clone());
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

    let lm_head_w = b.external_initializer("lm_head.weight", weights.lm_head.clone());
    let lm_head_w = b.transpose("lm_head_t", lm_head_w, vec![1, 0]);
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
    }
}

fn build_layer(
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

    // Q/K/V projection
    let q_w = b.external_initializer(
        &format!("{prefix}.self_attn.q_proj.weight"),
        lw.q_proj.clone(),
    );
    let k_w = b.external_initializer(
        &format!("{prefix}.self_attn.k_proj.weight"),
        lw.k_proj.clone(),
    );
    let v_w = b.external_initializer(
        &format!("{prefix}.self_attn.v_proj.weight"),
        lw.v_proj.clone(),
    );
    let q_w = b.transpose(&format!("{prefix}_q_w_t"), q_w, vec![1, 0]);
    let k_w = b.transpose(&format!("{prefix}_k_w_t"), k_w, vec![1, 0]);
    let v_w = b.transpose(&format!("{prefix}_v_w_t"), v_w, vec![1, 0]);
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

    // RoPE on Q and K (V is unrotated)
    let q = b.rope(
        &format!("{prefix}_q_rope"),
        q,
        ctx.cos_4d,
        ctx.sin_4d,
        ctx.head_dim,
    );
    let k = b.rope(
        &format!("{prefix}_k_rope"),
        k,
        ctx.cos_4d,
        ctx.sin_4d,
        ctx.head_dim,
    );

    // K/V cache (graph inputs; SessionConfig converts to SessionState)
    let k_cache_name = format!("{prefix}.past_key");
    let v_cache_name = format!("{prefix}.past_value");
    let k_cache = b.input(
        &k_cache_name,
        ctx.activation_ty,
        &[1, ctx.num_kv_heads, ctx.max_seq_len, ctx.head_dim],
    );
    let v_cache = b.input(
        &v_cache_name,
        ctx.activation_ty,
        &[1, ctx.num_kv_heads, ctx.max_seq_len, ctx.head_dim],
    );

    let elem_bytes = ctx.activation_ty.bit_width() / 8;
    let kv_cache = KVCache {
        k_name: k_cache_name.clone(),
        v_name: v_cache_name.clone(),
        bytes_per_buffer: ctx.num_kv_heads * ctx.max_seq_len * ctx.head_dim * elem_bytes,
    };

    let scale = (1.0_f64 / (ctx.head_dim as f64).sqrt()) as f32;
    let k_updated = b.kv_cache_update(&format!("{prefix}_k_update"), k_cache, k, ctx.past_len);
    let v_updated = b.kv_cache_update(&format!("{prefix}_v_update"), v_cache, v, ctx.past_len);
    let attn_out = b.attention(
        &format!("{prefix}_attn"),
        q,
        k_updated,
        v_updated,
        None,
        Some(ctx.active_seq_kv),
        ctx.is_prefill,
        scale,
    );

    let attn_out = b.transpose(&format!("{prefix}_attn_tr"), attn_out, vec![0, 2, 1, 3]);
    let attn_back_shape = b.i64_initializer(
        &format!("{prefix}_attn_shape"),
        vec![1, ctx.seq_q as i64, ctx.hidden as i64],
    );
    let attn_out = b.reshape(&format!("{prefix}_attn_rs"), attn_out, attn_back_shape);

    let o_w = b.external_initializer(
        &format!("{prefix}.self_attn.o_proj.weight"),
        lw.o_proj.clone(),
    );
    let o_w = b.transpose(&format!("{prefix}_o_w_t"), o_w, vec![1, 0]);
    let o = b.matmul(&format!("{prefix}_o_proj"), attn_out, o_w);

    let attn_residual = b.add(&format!("{prefix}_attn_resid"), x_in, o);

    // Pre-MLP RMSNorm
    let post_norm_w = b.external_initializer(
        &format!("{prefix}.post_attention_layernorm.weight"),
        lw.post_norm.clone(),
    );
    let n2 = b.rms_norm(
        &format!("{prefix}_post_norm"),
        attn_residual,
        post_norm_w,
        -1,
        ctx.eps,
    );

    // MLP (SwiGLU)
    let gate_w = b.external_initializer(
        &format!("{prefix}.mlp.gate_proj.weight"),
        lw.gate_proj.clone(),
    );
    let up_w = b.external_initializer(&format!("{prefix}.mlp.up_proj.weight"), lw.up_proj.clone());
    let down_w = b.external_initializer(
        &format!("{prefix}.mlp.down_proj.weight"),
        lw.down_proj.clone(),
    );
    let gate_w = b.transpose(&format!("{prefix}_gate_w_t"), gate_w, vec![1, 0]);
    let up_w = b.transpose(&format!("{prefix}_up_w_t"), up_w, vec![1, 0]);
    let down_w = b.transpose(&format!("{prefix}_down_w_t"), down_w, vec![1, 0]);

    let gate = b.matmul(&format!("{prefix}_gate_proj"), n2, gate_w);
    let gate = b.silu(&format!("{prefix}_silu"), gate);
    let up = b.matmul(&format!("{prefix}_up_proj"), n2, up_w);
    let mlp_in = b.mul(&format!("{prefix}_swiglu"), gate, up);
    let mlp_out = b.matmul(&format!("{prefix}_down_proj"), mlp_in, down_w);

    (
        b.add(&format!("{prefix}_mlp_resid"), attn_residual, mlp_out),
        kv_cache,
    )
}
