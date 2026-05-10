#![cfg(feature = "local")]

use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
use my_nn_engine_llm::build_llama_with_options;
use my_nn_engine_llm::llama::build_llama_prefill_with_options;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlamaOptions;
use my_nn_engine_llm::LlamaWeights;
use my_nn_engine_llm::LlmSession;
#[cfg(feature = "cuda")]
use serial_test::serial;
use tokenizers::Tokenizer;

const PROMPT: &str = "The capital of France is";

fn streaming_session(
    decode_graph: my_nn_engine::graph::Graph,
    prefill_graph: my_nn_engine::graph::Graph,
    prefill_len: usize,
    kv_cache_names: Vec<my_nn_engine_llm::session::KVCache>,
    tokenizer: Tokenizer,
    opts: &Options,
    max_seq_len: usize,
    eos_token_id: u32,
) -> LlmSession {
    let sink = 4;
    let window = max_seq_len - sink;
    LlmSession::for_llama_streaming(
        decode_graph,
        Some((prefill_graph, prefill_len)),
        kv_cache_names,
        tokenizer,
        opts,
        max_seq_len,
        eos_token_id,
        sink,
        window,
    )
    .unwrap()
}

fn run_tinyllama(target: Target) -> String {
    const N_GENERATE: usize = 16;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.safetensors")).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 256;
    let prefill_len = 16;
    let llama_opts = LlamaOptions {
        quant_kv_cache: false,
        streaming_kv: true,
    };
    let r = build_llama_with_options(&config, &weights, max_seq_len, &llama_opts);
    let p =
        build_llama_prefill_with_options(&config, &weights, max_seq_len, prefill_len, &llama_opts);

    let opts = Options::builder().target(target).build();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let mut llm = streaming_session(
        r.graph,
        p.graph,
        prefill_len,
        r.kv_cache_names,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
    );

    llm.generate(PROMPT, N_GENERATE).unwrap()
}

fn run_llama2_int8(target: Target) -> String {
    const N_GENERATE: usize = 24;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/llama2-7b-sft");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.int8.safetensors")).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 2048;
    let prefill_len = 16;
    // CPU codegen does not yet implement the rope+dequant fuse in attention,
    // so streaming-KV + INT8 only works on CUDA.
    let streaming_kv = matches!(target, Target::CUDA);
    let llama_opts = LlamaOptions {
        quant_kv_cache: true,
        streaming_kv,
    };
    let r = build_llama_with_options(&config, &weights, max_seq_len, &llama_opts);
    let p =
        build_llama_prefill_with_options(&config, &weights, max_seq_len, prefill_len, &llama_opts);

    let opts = Options::builder().target(target).build();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let mut llm = if streaming_kv {
        streaming_session(
            r.graph,
            p.graph,
            prefill_len,
            r.kv_cache_names,
            tokenizer,
            &opts,
            max_seq_len,
            config.eos_token_id,
        )
    } else {
        LlmSession::for_llama_with_prefill(
            r.graph,
            p.graph,
            r.kv_cache_names,
            prefill_len,
            tokenizer,
            &opts,
            max_seq_len,
            config.eos_token_id,
        )
        .unwrap()
    };

    llm.generate(PROMPT, N_GENERATE).unwrap()
}

#[cfg(feature = "cuda")]
#[test]
#[serial(gpu)]
fn tinyllama() {
    const EXPECTED_TEXT: &str = "Paris, which is also the largest city in the country.\n\n2.";
    let text = run_tinyllama(Target::CUDA);
    assert_eq!(text, EXPECTED_TEXT);
}

#[cfg(feature = "cuda")]
#[test]
#[serial(gpu)]
fn llama2_int8() {
    const EXPECTED_TEXT: &str = "Paris.\nThe capital of Germany is Berlin.\nThe capital of Greece is Athens.\nThe capital of India";
    let text = run_llama2_int8(Target::CUDA);
    assert_eq!(text, EXPECTED_TEXT);
}

#[test]
fn tinyllama_cpu() {
    const EXPECTED_TEXT: &str = "Paris, which is also the largest city in the country.\n\n2.";
    let text = run_tinyllama(Target::CPU);
    assert_eq!(text, EXPECTED_TEXT);
}

#[test]
fn llama2_int8_cpu() {
    const EXPECTED_TEXT: &str = "Paris.\nThe capital of Germany is Berlin.\nThe capital of Greece is Athens.\nThe capital of India";
    let text = run_llama2_int8(Target::CPU);
    assert_eq!(text, EXPECTED_TEXT);
}
