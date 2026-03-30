use std::path::PathBuf;
use std::time::Instant;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::tensor::Tensor;

fn bench_with_options(
    model_name: &str,
    model_file: &str,
    num_inputs: usize,
    label: &str,
    options: &Options,
) {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model_name);
    let model_path = root_dir.join(model_file);
    let data_dir = root_dir.join("test_data_set_0");

    let inputs: Vec<Tensor> = (0..num_inputs)
        .map(|i| Tensor::load_from_path(data_dir.join(format!("input_{}.pb", i))).unwrap())
        .collect();
    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();

    let session = Session::new(&model_path, Some(&input_types), options).unwrap();

    let _ = session.run(&inputs).unwrap();

    let n = 5;
    let start = Instant::now();
    for _ in 0..n {
        let _ = session.run(&inputs).unwrap();
    }
    let elapsed = start.elapsed();
    eprintln!(
        "[{label}] {n} runs in {:.2?}, avg {:.2?}/run",
        elapsed,
        elapsed / n,
    );
}

fn bench(model_name: &str, model_file: &str, num_inputs: usize, label: &str, target: Target) {
    bench_with_options(
        model_name,
        model_file,
        num_inputs,
        label,
        &Options::builder().target(target).build(),
    );
}

fn profile(model_name: &str, model_file: &str, num_inputs: usize, label: &str) {
    let root_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/validated")
        .join(model_name);
    let model_path = root_dir.join(model_file);
    let data_dir = root_dir.join("test_data_set_0");

    let inputs: Vec<Tensor> = (0..num_inputs)
        .map(|i| Tensor::load_from_path(data_dir.join(format!("input_{}.pb", i))).unwrap())
        .collect();
    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();

    let session = Session::new(
        &model_path,
        Some(&input_types),
        &Options::builder().profile(true).build(),
    )
    .unwrap();

    let _ = session.run(&inputs).unwrap();
    let start = Instant::now();
    let _ = session.run(&inputs).unwrap();
    eprintln!("[{label}] profile run: {:.2?}", start.elapsed());
}

#[test]
#[ignore]
fn profile_bert_cpu() {
    profile("bertsquad-12", "bertsquad-12.onnx", 4, "BERT");
}

#[test]
#[ignore]
fn profile_gpt2_cpu() {
    profile("GPT2", "model.onnx", 1, "GPT-2");
}

#[test]
#[ignore]
fn profile_resnet18_cpu() {
    profile("resnet18-v2-7", "resnet18-v2-7.onnx", 1, "ResNet18");
}

#[test]
#[ignore]
fn bench_gpt2_cpu() {
    bench("GPT2", "model.onnx", 1, "GPT-2", Target::CPU);
}

#[test]
#[ignore]
fn bench_bert_cpu() {
    bench("bertsquad-12", "bertsquad-12.onnx", 4, "BERT", Target::CPU);
}

#[test]
#[ignore]
fn bench_resnet18_cpu() {
    bench(
        "resnet18-v2-7",
        "resnet18-v2-7.onnx",
        1,
        "ResNet18",
        Target::CPU,
    );
}

#[test]
#[ignore]
fn bench_resnet18_cpu_nhwc_comparison() {
    bench_with_options(
        "resnet18-v2-7",
        "resnet18-v2-7.onnx",
        1,
        "ResNet18 CPU (NHWC off)",
        &Options::builder()
            .target(Target::CPU)
            .enable_nhwc_optimization(Some(false))
            .build(),
    );
    bench_with_options(
        "resnet18-v2-7",
        "resnet18-v2-7.onnx",
        1,
        "ResNet18 CPU (NHWC on)",
        &Options::builder()
            .target(Target::CPU)
            .enable_nhwc_optimization(Some(true))
            .build(),
    );
}

#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn bench_gpt2_cuda() {
    bench("GPT2", "model.onnx", 1, "GPT-2 CUDA", Target::CUDA);
}

#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn bench_bert_cuda() {
    bench(
        "bertsquad-12",
        "bertsquad-12.onnx",
        4,
        "BERT CUDA",
        Target::CUDA,
    );
}

#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn bench_resnet18_cuda() {
    bench(
        "resnet18-v2-7",
        "resnet18-v2-7.onnx",
        1,
        "ResNet18 CUDA",
        Target::CUDA,
    );
}

#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn bench_resnet18_cuda_nhwc_comparison() {
    bench_with_options(
        "resnet18-v2-7",
        "resnet18-v2-7.onnx",
        1,
        "ResNet18 CUDA (NHWC off)",
        &Options::builder()
            .target(Target::CUDA)
            .enable_nhwc_optimization(Some(false))
            .build(),
    );
    bench_with_options(
        "resnet18-v2-7",
        "resnet18-v2-7.onnx",
        1,
        "ResNet18 CUDA (NHWC on)",
        &Options::builder()
            .target(Target::CUDA)
            .enable_nhwc_optimization(Some(true))
            .build(),
    );
}

#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn bench_resnet152_cuda_nhwc_comparison() {
    bench_with_options(
        "resnet152-v2-7",
        "resnet152-v2-7.onnx",
        1,
        "ResNet152 CUDA (NHWC off)",
        &Options::builder()
            .target(Target::CUDA)
            .enable_nhwc_optimization(Some(false))
            .build(),
    );
    bench_with_options(
        "resnet152-v2-7",
        "resnet152-v2-7.onnx",
        1,
        "ResNet152 CUDA (NHWC on)",
        &Options::builder()
            .target(Target::CUDA)
            .enable_nhwc_optimization(Some(true))
            .build(),
    );
}
