#![cfg(feature = "local")]

use std::path::PathBuf;

use my_nn_engine::options::Options;
use my_nn_engine::options::Target;
#[cfg(feature = "cuda")]
use my_nn_engine::schedule::scheduler::PlacementStrategy;
use my_nn_engine_llm::build_decoder;
use my_nn_engine_llm::BuildOptions;
use my_nn_engine_llm::HfConfig;
use my_nn_engine_llm::HfWeights;
use my_nn_engine_llm::LlmSession;
use my_nn_engine_llm::ModelSpec;
#[cfg(feature = "cuda")]
use serial_test::serial;
use tokenizers::Tokenizer;

const PROMPT: &str = "The capital of France is";
const STREAM_SINK: usize = 4;

fn run_tinyllama(target: Target) -> String {
    run_tinyllama_with(Options::builder().target(target).build())
}

fn run_tinyllama_with(opts: Options) -> String {
    const N_GENERATE: usize = 16;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.safetensors")).unwrap();
    let spec = ModelSpec::from_hf(&config, &hf).unwrap();

    let max_seq_len = 256;
    let prefill_len = 16;
    let llama_opts = BuildOptions::builder().streaming_kv(true).build();
    let r = build_decoder(&config, &spec, max_seq_len, &llama_opts);
    let p = build_decoder(
        &config,
        &spec,
        max_seq_len,
        &BuildOptions {
            prefill_len: Some(prefill_len),
            ..llama_opts.clone()
        },
    );

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let mut llm = LlmSession::for_llama_streaming(
        r.graph,
        Some((p.graph, prefill_len)),
        r.kv_cache_names,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
        STREAM_SINK,
        max_seq_len - STREAM_SINK,
    )
    .unwrap();
    llm.generate(PROMPT, N_GENERATE).unwrap()
}

fn run_llama2_int8(target: Target) -> String {
    const N_GENERATE: usize = 24;

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/llama2-7b-sft");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_safetensors(dir.join("model.int8.safetensors")).unwrap();
    let spec = ModelSpec::from_hf(&config, &hf).unwrap();

    let max_seq_len = 2048;
    let prefill_len = 16;
    // CPU codegen does not yet implement the rope+dequant fuse in attention,
    // so streaming-KV + INT8 only works on CUDA.
    let streaming_kv = matches!(target, Target::CUDA);
    let llama_opts = BuildOptions::builder()
        .quant_kv_cache(true)
        .streaming_kv(streaming_kv)
        .build();
    let r = build_decoder(&config, &spec, max_seq_len, &llama_opts);
    let p = build_decoder(
        &config,
        &spec,
        max_seq_len,
        &BuildOptions {
            prefill_len: Some(prefill_len),
            ..llama_opts.clone()
        },
    );

    let opts = Options::builder().target(target).build();
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let mut llm = if streaming_kv {
        LlmSession::for_llama_streaming(
            r.graph,
            Some((p.graph, prefill_len)),
            r.kv_cache_names,
            tokenizer,
            &opts,
            max_seq_len,
            config.eos_token_id,
            STREAM_SINK,
            max_seq_len - STREAM_SINK,
        )
        .unwrap()
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

fn run_llama3_int8_with(opts: Options) -> String {
    const N_GENERATE: usize = 8;

    let dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/llama3-8b-instruct-int8");
    let config = HfConfig::from_path(dir.join("config.json")).unwrap();
    let hf = HfWeights::from_dir(&dir).unwrap();
    let spec = ModelSpec::from_hf(&config, &hf).unwrap();

    let max_seq_len = 512;
    let prefill_len = 16;
    let llama_opts = BuildOptions::builder().quant_kv_cache(true).build();
    let r = build_decoder(&config, &spec, max_seq_len, &llama_opts);
    let p = build_decoder(
        &config,
        &spec,
        max_seq_len,
        &BuildOptions {
            prefill_len: Some(prefill_len),
            ..llama_opts.clone()
        },
    );

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
fn tinyllama_hybrid() {
    const EXPECTED_TEXT: &str = "Paris, which is also the largest city in the country.\n\n2.";
    let opts = Options::builder()
        .target(Target::CUDA)
        .placement_strategy(Some(PlacementStrategy::StructuralKvTouch))
        .build();
    let text = run_tinyllama_with(opts);
    assert_eq!(text, EXPECTED_TEXT);
}
