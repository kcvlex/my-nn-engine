use std::path::PathBuf;

use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let extract_dir = PathBuf::from(
        std::env::var("EXTRACTED_MODEL_DIR")
            .map_err(|_| "EXTRACTED_MODEL_DIR env var must be set")?,
    );
    let epsilon: f64 = std::env::var("EPSILON")
        .unwrap_or_else(|_| "0.01".to_string())
        .parse()
        .map_err(|_| "invalid EPSILON value")?;

    let model_path = extract_dir.join("model.onnx");
    if !model_path.exists() {
        return Err(format!("Model not found: {}", model_path.display()).into());
    }

    // Count and load inputs
    let mut inputs = Vec::new();
    for i in 0.. {
        let input_path = extract_dir.join(format!("input_{}.pb", i));
        if !input_path.exists() {
            break;
        }
        inputs.push(
            Tensor::load_from_path(&input_path)
                .map_err(|e| format!("Failed to load {}: {:?}", input_path.display(), e))?,
        );
    }

    if inputs.is_empty() {
        return Err("No input files found (input_0.pb, input_1.pb, ...)".into());
    }

    // Count expected outputs
    let mut num_outputs = 0;
    for i in 0.. {
        let output_path = extract_dir.join(format!("output_{}.pb", i));
        if !output_path.exists() {
            break;
        }
        num_outputs = i + 1;
    }

    if num_outputs == 0 {
        return Err("No output files found (output_0.pb, output_1.pb, ...)".into());
    }

    eprintln!("Testing extracted model:");
    eprintln!("  Model: {}", model_path.display());
    eprintln!("  Inputs: {}", inputs.len());
    eprintln!("  Expected outputs: {}", num_outputs);
    eprintln!("  Epsilon: {}", epsilon);

    // Create session
    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();
    let session = Session::new(
        &model_path,
        Some(&input_types),
        &Options::builder().target(Target::CPU).build(),
    )
    .map_err(|e| format!("Failed to create session: {:?}", e))?;

    // Run model
    let outputs = session
        .run(&inputs)
        .map_err(|e| format!("Failed to run model: {:?}", e))?;

    if outputs.len() != num_outputs {
        return Err(format!("Expected {} outputs but got {}", num_outputs, outputs.len()).into());
    }

    // Load and compare expected outputs
    for (i, output) in outputs.iter().enumerate() {
        let expected_path = extract_dir.join(format!("output_{}.pb", i));
        let expected = Tensor::load_from_path(&expected_path)
            .map_err(|e| format!("Failed to load {}: {:?}", expected_path.display(), e))?;

        if !output.eq_with_epsilon(&expected, epsilon, CompPolicy::Either) {
            eprintln!("\nFAIL: Output {} mismatch!", i);
            eprintln!("  Expected shape: {:?}", expected.dims);
            eprintln!("  Got shape: {:?}", output.dims);

            return Err(format!(
                "Output {} does not match expected within epsilon {}",
                i, epsilon
            )
            .into());
        }

        eprintln!("  OK: Output {} matches (shape: {:?})", i, output.dims);
    }

    eprintln!("\nOK: All outputs match within epsilon {}", epsilon);
    Ok(())
}
