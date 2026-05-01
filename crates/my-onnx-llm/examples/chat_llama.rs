// Demonstrates KV-cache persistence across multiple `generate()` calls on a
// single LlmSession: the second turn should recall the name from the first.

use std::path::PathBuf;

use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx_llm::build_llama;
use my_onnx_llm::HfConfig;
use my_onnx_llm::HfWeights;
use my_onnx_llm::LlamaWeights;
use my_onnx_llm::LlmSession;
use tokenizers::Tokenizer;

fn main() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");

    let config = HfConfig::from_path(model_dir.join("config.json")).unwrap();
    let hf = HfWeights::from_index(
        model_dir.join("weights.f32.bin"),
        model_dir.join("weights.f32.json"),
    )
    .unwrap();
    let weights = LlamaWeights::from_hf(&hf, config.num_hidden_layers).unwrap();

    let max_seq_len = 256;
    let r = build_llama(&config, &weights, max_seq_len);
    let opts = Options::builder().target(Target::CUDA).build();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let mut llm = LlmSession::for_llama(
        r.graph,
        r.kv_cache_names,
        tokenizer,
        &opts,
        max_seq_len,
        config.eos_token_id,
    )
    .unwrap();

    let turn1 = "Hello, my name is Alice. Please remember it.";
    let r1 = llm.generate(turn1, 32).unwrap();
    println!("turn1 prompt: {turn1:?}");
    println!("turn1 reply : {r1}");
    println!("past_len after turn1: {}", llm.past_len());

    let turn2 = "\nWhat name did I tell you?";
    let r2 = llm.generate(turn2, 32).unwrap();
    println!("\nturn2 prompt: {turn2:?}");
    println!("turn2 reply : {r2}");
    println!("past_len after turn2: {}", llm.past_len());
}
