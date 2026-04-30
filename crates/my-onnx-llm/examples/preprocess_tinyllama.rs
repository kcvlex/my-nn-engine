use std::path::PathBuf;

use my_onnx_llm::HfWeights;

fn main() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/hf/tinyllama");
    let src = model_dir.join("model.safetensors");
    let bin = model_dir.join("weights.f32.bin");
    let idx = model_dir.join("weights.f32.json");
    println!("preprocessing {:?} → {:?}", src, bin);
    let t0 = std::time::Instant::now();
    HfWeights::preprocess_safetensors(&src, &bin, &idx).unwrap();
    println!("done in {:.2?}", t0.elapsed());

    let weights = HfWeights::from_index(&bin, &idx).unwrap();
    let names: Vec<&str> = weights.names().collect();
    println!("loaded index with {} entries", names.len());
    for n in [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
    ] {
        let r = weights.external_ref(n).unwrap();
        println!(
            "  {n}: dims={:?} offset={} length={}",
            r.dims.iter().copied().collect::<Vec<_>>(),
            r.offset,
            r.length.unwrap_or(0)
        );
    }
}
