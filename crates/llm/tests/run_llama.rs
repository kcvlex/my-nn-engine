#![cfg(feature = "local")]

use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
#[cfg(feature = "cuda")]
use my_nn_engine::schedule::scheduler::PlacementStrategy;
use my_nn_engine_llm::build_llama;
use my_nn_engine_llm::build_llama_with_options;
use my_nn_engine_llm::llama::build_llama_prefill;
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

fn run_tinyllama(target: Target) -> String {
    run_tinyllama_with(Options::builder().target(target).build())
}

fn run_tinyllama_with(opts: Options) -> String {
    const N_GENERATE: usize = 16;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.safetensors")).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 256;
    let prefill_len = 16;
    let r = build_llama(&config, &weights, max_seq_len);
    let p = build_llama_prefill(&config, &weights, max_seq_len, prefill_len);
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
    let llama_opts = LlamaOptions {
        quant_kv_cache: true,
        streaming_kv: false,
    };
    let r = build_llama_with_options(&config, &weights, max_seq_len, &llama_opts);
    let p =
        build_llama_prefill_with_options(&config, &weights, max_seq_len, prefill_len, &llama_opts);

    let opts = Options::builder().target(target).build();
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
    const EXPECTED_TEXT: &str = "Paris.\nThe capital of Germany is Berlin.\nThe capital of Greece is Athens.\nThe capital of Italy";
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

fn run_llama3_int8_with(opts: Options) -> String {
    const N_GENERATE: usize = 32;

    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/llama3-8b-instruct-int8");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_dir(&dir).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 512;
    let prefill_len = 16;
    let llama_opts = LlamaOptions {
        quant_kv_cache: true,
    };
    let r = build_llama_with_options(&config, &weights, max_seq_len, &llama_opts);
    let p =
        build_llama_prefill_with_options(&config, &weights, max_seq_len, prefill_len, &llama_opts);

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let init_start = std::time::Instant::now();
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
    let init_secs = init_start.elapsed().as_secs_f64();

    let gen_start = std::time::Instant::now();
    let text = llm.generate(PROMPT, N_GENERATE).unwrap();
    let gen_secs = gen_start.elapsed().as_secs_f64();

    eprintln!(
        "TIMING init={:.2}s prefill+decode({} new tokens)={:.2}s ({:.1} tok/s)",
        init_secs,
        N_GENERATE,
        gen_secs,
        (N_GENERATE as f64) / gen_secs,
    );
    text
}

#[cfg(feature = "cuda")]
#[test]
#[serial(gpu)]
#[ignore = "experimental: llama3-8b-int8 hybrid (slow, CPU-bound)"]
fn llama3_int8_hybrid() {
    let opts = Options::builder()
        .target(Target::CUDA)
        .placement_strategy(Some(PlacementStrategy::StructuralKvTouch))
        .build();
    let text = run_llama3_int8_with(opts);
    eprintln!("llama3_int8_hybrid output: {:?}", text);
}

#[cfg(feature = "cuda")]
#[test]
#[serial(gpu)]
#[ignore = "experimental: llama3-8b-int8 attention-subgraph"]
fn llama3_int8_attention_subgraph() {
    let opts = Options::builder()
        .target(Target::CUDA)
        .placement_strategy(Some(PlacementStrategy::AttentionSubgraph))
        .build();
    let text = run_llama3_int8_with(opts);
    eprintln!("llama3_int8_attention_subgraph output: {:?}", text);
}

#[cfg(feature = "cuda")]
#[test]
#[serial(gpu)]
fn tinyllama_hybrid() {
    const EXPECTED_TEXT: &str = "Paris, which is also the largest city in the country.\n\n2.";
    let opts = Options::builder()
        .target(Target::CUDA)
        .placement_strategy(Some(PlacementStrategy::StructuralKvTouch))
        .build();
    let text = run_tinyllama_with(opts);
    assert_eq!(text, EXPECTED_TEXT);
}

#[cfg(feature = "cuda")]
#[test]
#[serial(gpu)]
fn tinyllama_attention_subgraph() {
    const EXPECTED_TEXT: &str = "Paris, which is also the largest city in the country.\n\n2.";
    let opts = Options::builder()
        .target(Target::CUDA)
        .placement_strategy(Some(PlacementStrategy::AttentionSubgraph))
        .build();
    let text = run_tinyllama_with(opts);
    assert_eq!(text, EXPECTED_TEXT);
}
