use std::path::PathBuf;

use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx::session::Session;
use my_onnx::session::SessionConfig;
use my_onnx::session::SessionStateSpec;
use my_onnx::session::StateInit;
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::SIntType;
use my_onnx::tensor::Tensor;
use my_onnx_llm::build_llama;
use my_onnx_llm::HfConfig;
use my_onnx_llm::HfWeights;
use my_onnx_llm::LlamaWeights;
use tokenizers::Tokenizer;

fn make_i64(dims: &[usize], values: Vec<i64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::SInt(SIntType::I64, values),
    )
    .unwrap()
}

fn argmax_logits(t: &Tensor) -> usize {
    let v = match &t.data {
        TensorData::Float(_, v) => v,
        _ => panic!("logits must be float"),
    };
    let mut best_i = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best_i = i;
        }
    }
    best_i
}

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
    let n_generate = 16;

    println!("building graph...");
    let t0 = std::time::Instant::now();
    let r = build_llama(&config, &weights, max_seq_len);
    println!("  graph built in {:.2?}", t0.elapsed());

    let session_config = SessionConfig {
        session_states: r
            .kv_cache_names
            .iter()
            .map(|n| SessionStateSpec {
                name: n.clone(),
                init: StateInit::Zero,
            })
            .collect(),
    };

    println!("compiling session (CUDA)...");
    let t0 = std::time::Instant::now();
    let opts = Options::builder().target(Target::CUDA).build();
    let mut session = Session::from_graph(r.graph, &opts, &session_config).unwrap();
    println!("  compiled in {:.2?}", t0.elapsed());

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let prompt = "The capital of France is";
    let encoding = tokenizer.encode(prompt, true).unwrap();
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("prompt: {prompt:?}");
    println!("prompt ids: {prompt_ids:?}");

    let mut all_ids = prompt_ids.clone();
    let total = prompt_ids.len() + n_generate;

    let t0 = std::time::Instant::now();
    for step in 0..total {
        let token = all_ids[step] as i64;
        let past_len = step as i64;
        let active = (step + 1) as i64;

        let inputs = vec![
            make_i64(&[1, 1], vec![token]),
            make_i64(&[1], vec![past_len]),
            make_i64(&[], vec![past_len]),
            make_i64(&[], vec![active]),
        ];
        let outputs = session.run(&inputs).unwrap();
        let logits = &outputs[0];
        let next = argmax_logits(logits);

        if step + 1 >= prompt_ids.len() {
            all_ids.push(next as u32);
        }
    }
    println!("decode {} steps in {:.2?}", total, t0.elapsed());

    let text = tokenizer.decode(&all_ids, false).unwrap();
    println!("output: {text}");
}
