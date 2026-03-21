use std::path::PathBuf;
use std::time::Instant;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::tensor::Tensor;

fn bench(model_name: &str, model_file: &str, num_inputs: usize, label: &str) {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model_name);
    let model_path = root_dir.join(model_file);
    let data_dir = root_dir.join("test_data_set_0");

    let inputs: Vec<Tensor> = (0..num_inputs)
        .map(|i| Tensor::load_from_path(data_dir.join(format!("input_{}.pb", i))).unwrap())
        .collect();
    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();

    for attn_threshold in [usize::MAX, 0] {
        let tag = if attn_threshold == usize::MAX {
            "attn OMP off"
        } else {
            "attn OMP on"
        };

        let session = Session::new(
            &model_path,
            Some(&input_types),
            &Options::builder()
                .omp_attention_threshold(attn_threshold)
                .build(),
        )
        .unwrap();

        let _ = session.run(&inputs).unwrap();

        let n = 5;
        let start = Instant::now();
        for _ in 0..n {
            let _ = session.run(&inputs).unwrap();
        }
        let elapsed = start.elapsed();
        eprintln!(
            "[{label}] {tag}: {n} runs in {:.2?}, avg {:.2?}/run",
            elapsed,
            elapsed / n,
        );
    }
}

#[test]
#[ignore]
fn bench_gpt2_cpu() {
    bench("GPT2", "model.onnx", 1, "GPT-2");
}

#[test]
#[ignore]
fn bench_bert_cpu() {
    bench("bertsquad-12", "bertsquad-12.onnx", 4, "BERT");
}
