use std::path::PathBuf;

use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx_llm::build_llama;
use my_onnx_llm::llama::build_llama_prefill;
use my_onnx_llm::HfConfig;
use my_onnx_llm::HfWeights;
use my_onnx_llm::LlamaWeights;
use my_onnx_llm::LlmSession;
use tokenizers::Tokenizer;

fn main() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");

    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let hf = HfWeights::from_dir(&model_dir).unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 256;
    let prefill_len = 16;
    let n_generate = 16;

    println!("building decode graph...");
    let t0 = std::time::Instant::now();
    let r = build_llama(&config, &weights, max_seq_len);
    println!("  decode graph built in {:.2?}", t0.elapsed());

    println!("building prefill graph...");
    let t0 = std::time::Instant::now();
    let p = build_llama_prefill(&config, &weights, max_seq_len, prefill_len);
    println!("  prefill graph built in {:.2?}", t0.elapsed());

    println!("compiling sessions (CUDA)...");
    let t0 = std::time::Instant::now();
    let opts = Options::builder().target(Target::CUDA).build();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
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
    println!("  compiled in {:.2?}", t0.elapsed());

    let prompt = "The capital of France is";
    let t0 = std::time::Instant::now();
    let text = llm.generate(prompt, n_generate).unwrap();
    println!("decode in {:.2?}", t0.elapsed());
    println!("prompt: {prompt:?}");
    println!("output: {text}");
}
