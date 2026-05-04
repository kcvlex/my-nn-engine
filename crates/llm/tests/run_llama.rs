#![cfg(all(feature = "local", feature = "cuda"))]

use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::build_llama;
use my_nn_engine_llm::build_llama_with_options;
use my_nn_engine_llm::llama::build_llama_prefill;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlamaOptions;
use my_nn_engine_llm::LlamaWeights;
use my_nn_engine_llm::LlmSession;
use serial_test::serial;
use tokenizers::Tokenizer;

const PROMPT: &str = "The capital of France is";

#[test]
#[serial(gpu)]
fn tinyllama() {
    const N_GENERATE: usize = 16;
    const EXPECTED_TEXT: &str = "Paris, which is also the largest city in the country.\n\n2.";

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.safetensors")).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 256;
    let prefill_len = 16;
    let r = build_llama(&config, &weights, max_seq_len);
    let p = build_llama_prefill(&config, &weights, max_seq_len, prefill_len);

    let opts = Options::builder().target(Target::CUDA).build();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let mut llm = LlmSession::for_llama_with_prefill(
        r.graph,
        p.graph,
        r.kv_cache_names,
        prefill_len,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
    )
    .unwrap();

    let text = llm.generate(PROMPT, N_GENERATE).unwrap();
    assert_eq!(text, EXPECTED_TEXT);
}

#[test]
#[serial(gpu)]
fn llama2_int8() {
    const N_GENERATE: usize = 24;
    const EXPECTED_TEXT: &str = "Paris.\nThe capital of Germany is Berlin.\nThe capital of Greece is Athens.\nThe capital of India";

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/llama2-7b-sft");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.int8.safetensors")).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 2048;
    let llama_opts = LlamaOptions {
        quant_kv_cache: true,
    };
    let r = build_llama_with_options(&config, &weights, max_seq_len, &llama_opts);

    let opts = Options::builder().target(Target::CUDA).build();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let mut llm = LlmSession::for_llama(
        r.graph,
        r.kv_cache_names,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
    )
    .unwrap();

    let text = llm.generate(PROMPT, N_GENERATE).unwrap();
    assert_eq!(text, EXPECTED_TEXT);
}
