#!/bin/bash
# Helper script to test an extracted model directory
# Usage: test_extracted.sh <extract_dir> <epsilon>

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <extract_dir> [epsilon]"
    exit 1
fi

EXTRACT_DIR="$1"
EPSILON="${2:-0.01}"  # Default epsilon 0.01

# Create a temporary test in Rust
cat > /tmp/test_extracted_temp.rs <<EOF
use my_onnx::onnx::load::*;
use my_onnx::options::*;
use my_onnx::session::Session;
use my_onnx::tensor::data::CompPolicy;
use my_onnx::tensor::Tensor;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let extract_dir = PathBuf::from("${EXTRACT_DIR}");
    let model_path = extract_dir.join("model.onnx");

    // Count inputs
    let mut num_inputs = 0;
    while extract_dir.join(format!("input_{}.pb", num_inputs)).exists() {
        num_inputs += 1;
    }

    // Count outputs
    let mut num_outputs = 0;
    while extract_dir.join(format!("output_{}.pb", num_outputs)).exists() {
        num_outputs += 1;
    }

    // Load inputs
    let inputs: Vec<Tensor> = (0..num_inputs)
        .map(|i| Tensor::load_from_path(extract_dir.join(format!("input_{}.pb", i))))
        .collect::<Result<Vec<_>, _>>()?;

    let input_types: Vec<_> = inputs.iter().map(|t| t.tensor_type()).collect();

    // Create session and run
    let session = Session::new(
        &model_path,
        Some(&input_types),
        &Options::builder().target(Target::CPU).build(),
    )?;
    let outputs = session.run(&inputs)?;

    // Load expected outputs
    let expected: Vec<Tensor> = (0..num_outputs)
        .map(|i| Tensor::load_from_path(extract_dir.join(format!("output_{}.pb", i))))
        .collect::<Result<Vec<_>, _>>()?;

    // Compare
    let epsilon = ${EPSILON};
    for (i, (output, expected)) in outputs.iter().zip(expected.iter()).enumerate() {
        if !output.eq_with_epsilon(expected, epsilon, CompPolicy::Either) {
            eprintln!("Output {} differs by more than {}", i, epsilon);
            eprintln!("Expected shape: {:?}", expected.dims());
            eprintln!("Got shape: {:?}", output.dims());
            // Print first few values for debugging
            return Err(format!("Output mismatch at index {}", i).into());
        }
    }

    println!("✓ All outputs match within epsilon {}", epsilon);
    Ok(())
}
EOF

# Run the temporary test
cargo run --manifest-path="$(dirname "$0")/../../Cargo.toml" \
    --bin test_extracted_temp \
    -- 2>&1
