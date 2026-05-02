use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::GenerateOptions;
use my_nn_engine_llm::LlmConfig;
use my_nn_engine_llm::LlmSession;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mock_decode")
}

fn make_session() -> LlmSession {
    let dir = fixtures_dir();
    let opts = Options::builder().target(Target::CPU).build();
    let config = LlmConfig {
        max_seq_len: 16,
        eos_token_id: 999,
        session_states: vec![],
    };
    LlmSession::new(
        dir.join("model.onnx"),
        dir.join("tokenizer.json"),
        &opts,
        config,
    )
    .unwrap()
}

#[test]
fn generate_ids_follows_logit_table() {
    let mut llm = make_session();
    let new_ids = llm.generate_ids("a b c", 5).unwrap();
    assert_eq!(new_ids, vec![3, 4, 5, 0, 1]);
}

#[test]
fn generate_decodes_to_text() {
    let mut llm = make_session();
    let text = llm.generate("a b c", 5).unwrap();
    assert_eq!(text, "c d e <unk> a");
}

#[test]
fn past_len_tracks_decode_steps() {
    let mut llm = make_session();
    assert_eq!(llm.past_len(), 0);
    llm.generate_ids("a b c", 5).unwrap();
    // 2 prompt tokens (fake-prefill) + 5 generated = 7 decode_step calls.
    assert_eq!(llm.past_len(), 7);
    llm.reset();
    assert_eq!(llm.past_len(), 0);
}

#[test]
fn empty_prompt_returns_no_tokens() {
    let mut llm = make_session();
    let new_ids = llm.generate_ids("", 5).unwrap();
    assert!(new_ids.is_empty());
}

#[test]
fn generate_with_stop_string_truncates_text() {
    // Without stop: "c d e <unk> a"
    let mut llm = make_session();
    let opts = GenerateOptions {
        max_new_tokens: 5,
        stop_strings: vec!["<unk>".to_string()],
    };
    let text = llm.generate_with("a b c", &opts).unwrap();
    assert_eq!(text, "c d e");
    assert!(!text.contains("<unk>"));
}

#[test]
fn past_len_rewinds_after_stop() {
    let mut llm = make_session();
    let opts = GenerateOptions {
        max_new_tokens: 5,
        stop_strings: vec!["<unk>".to_string()],
    };
    let new_ids = llm.generate_ids_with("a b c", &opts).unwrap();
    // 2 prompt tokens (mock prefill) consumes the first 2 decode_step calls,
    // and the surviving generated tokens contribute the rest. With stop hit
    // and rewound past_len, past_len == prompt_steps + new_ids.len().
    assert_eq!(llm.past_len(), 2 + new_ids.len());
}
