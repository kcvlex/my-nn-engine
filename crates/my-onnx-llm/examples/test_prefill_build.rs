use std::path::PathBuf;
use std::sync::Arc;

use my_onnx::options::Options;
use my_onnx::options::Target;
use my_onnx::session::DeviceBuffer;
use my_onnx::session::Session;
use my_onnx::session::SessionConfig;
use my_onnx::session::SessionStateSpec;
use my_onnx::tensor::data::TensorData;
use my_onnx::tensor::types::ResolvedTensorDims;
use my_onnx::tensor::types::SIntType;
use my_onnx::tensor::Tensor;
use my_onnx_llm::llama::build_llama_prefill;
use my_onnx_llm::HfConfig;
use my_onnx_llm::HfWeights;
use my_onnx_llm::LlamaWeights;

fn make_i64(dims: &[usize], values: Vec<i64>) -> Tensor {
    Tensor::new(
        ResolvedTensorDims::new(dims),
        TensorData::SInt(SIntType::I64, values),
    )
    .unwrap()
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
    let prefill_len = 32;

    let r = build_llama_prefill(&config, &weights, max_seq_len, prefill_len);
    println!("prefill graph built: {} kv caches", r.kv_cache_names.len());

    let session_config = SessionConfig {
        session_states: r
            .kv_cache_names
            .iter()
            .flat_map(|kv| {
                let k_buf = Arc::new(DeviceBuffer::alloc_zeroed(kv.bytes_per_buffer).unwrap());
                let v_buf = Arc::new(DeviceBuffer::alloc_zeroed(kv.bytes_per_buffer).unwrap());
                [
                    SessionStateSpec {
                        name: kv.k_name.clone(),
                        buffer: k_buf,
                    },
                    SessionStateSpec {
                        name: kv.v_name.clone(),
                        buffer: v_buf,
                    },
                ]
            })
            .collect(),
        ..SessionConfig::default()
    };

    let opts = Options::builder().target(Target::CUDA).build();
    let t0 = std::time::Instant::now();
    let mut session = Session::from_graph(r.graph, &opts, &session_config).unwrap();
    println!("prefill session compiled in {:.2?}", t0.elapsed());

    let prompt_ids: Vec<i64> = vec![1, 450, 7483, 310, 3444, 338]; // "<s> The capital of France is"
    let m = prompt_ids.len();
    let mut padded = prompt_ids.clone();
    padded.resize(prefill_len, 0);

    let positions: Vec<i64> = (0..prefill_len as i64).collect();

    let inputs = vec![
        make_i64(&[1, prefill_len], padded),
        make_i64(&[prefill_len], positions),
        make_i64(&[], vec![0]),                  // past_len
        make_i64(&[], vec![prefill_len as i64]), // active_seq_kv (unused in prefill)
    ];

    // warmup
    let _ = session.run(&inputs).unwrap();

    let t0 = std::time::Instant::now();
    let outputs = session.run(&inputs).unwrap();
    println!("prefill forward (steady) in {:.2?}", t0.elapsed());

    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        let _ = session.run(&inputs).unwrap();
    }
    println!("5 prefill forwards in {:.2?}", t0.elapsed());

    let logits = &outputs[0];
    let TensorData::Float(_, ref data) = logits.data else {
        panic!()
    };
    let dims = &logits.dims;
    println!(
        "logits shape: {:?}",
        dims.iter().copied().collect::<Vec<_>>()
    );

    // Take logits at position M-1 (last valid prompt position)
    let vocab_size = dims[2];
    let last_pos = m - 1;
    let start = last_pos * vocab_size;
    let end = (last_pos + 1) * vocab_size;
    let last_logits = &data[start..end];
    let (best_i, _) = last_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    println!(
        "argmax at position {} (last prompt token): {}",
        last_pos, best_i
    );
    println!("expected first response token (decode baseline): around 4001 ('Paris')");
}
