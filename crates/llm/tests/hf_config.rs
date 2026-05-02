use std::path::PathBuf;

use my_nn_engine_llm::HfConfig;

#[test]
fn parse_tinyllama_config() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tinyllama/config.json");
    let cfg = HfConfig::from_path(&path).unwrap();

    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.intermediate_size, 5632);
    assert_eq!(cfg.num_hidden_layers, 22);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.max_position_embeddings, 2048);
    assert_eq!(cfg.rope_theta, 10000.0);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert_eq!(cfg.hidden_act, "silu");
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.bos_token_id, 1);
    assert_eq!(cfg.eos_token_id, 2);
    assert_eq!(cfg.head_dim(), 64);
}
