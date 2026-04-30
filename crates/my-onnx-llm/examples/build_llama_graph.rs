use std::path::PathBuf;

use my_onnx_llm::build_llama;
use my_onnx_llm::HfConfig;
use my_onnx_llm::HfWeights;
use my_onnx_llm::LlamaWeights;

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

    let t0 = std::time::Instant::now();
    let r = build_llama(&config, &weights, max_seq_len);
    println!("built graph in {:.2?}", t0.elapsed());
    println!("  nodes: {}", r.graph.nodes.iter().count());
    println!("  kv caches: {}", r.kv_cache_names.len());
    println!(
        "  kv cache names (first 4): {:?}",
        &r.kv_cache_names[..4.min(r.kv_cache_names.len())]
    );
}
