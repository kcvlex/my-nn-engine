#!/usr/bin/env python3
"""
Extract a subgraph from an ONNX model and generate test data.

This script extracts a portion of an ONNX model from specified inputs to outputs,
runs it with the original input data to generate expected outputs, and saves
everything in the format expected by the test suite.
"""

import argparse
import os
import sys
from pathlib import Path

import onnx
import onnx.helper as oh
from onnx.utils import extract_model
import onnxruntime as rt
import numpy as np


def extract_and_run(
    input_model_path: str,
    input_tensor_paths: list[str],
    output_dir: str,
    output_names: list[str],
    input_names: list[str] | None = None,
    check_model: bool = True,
):
    """
    Extract a subgraph from an ONNX model and generate test data.

    Args:
        input_model_path: Path to the input ONNX model
        input_tensor_paths: Paths to input tensor .pb files
        output_dir: Directory to save extracted model and test data
        output_names: Names of the output nodes to extract up to
        input_names: Names of the input nodes (if None, uses original model inputs)
        check_model: Whether to check the extracted model
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the original model to get input names if not specified
    original_model = onnx.load(input_model_path)
    if input_names is None:
        input_names = [inp.name for inp in original_model.graph.input]

    # Verify we have the right number of input tensors
    if len(input_tensor_paths) != len(input_names):
        print(f"Error: Expected {len(input_names)} input tensors but got {len(input_tensor_paths)}")
        print(f"Input names: {input_names}")
        sys.exit(1)

    # Extract the model
    output_model_path = os.path.join(output_dir, "model.onnx")
    print(f"Extracting model from {input_model_path}")
    print(f"  Input nodes: {input_names}")
    print(f"  Output nodes: {output_names}")

    extract_model(
        input_model_path,
        output_model_path,
        input_names,
        output_names,
        check_model=check_model
    )
    print(f"Extracted model saved to {output_model_path}")

    # Load input tensors
    print(f"\nLoading {len(input_tensor_paths)} input tensor(s)...")
    input_data = {}
    for i, (name, tensor_path) in enumerate(zip(input_names, input_tensor_paths)):
        print(f"  Loading {tensor_path} as '{name}'")
        tensor = onnx.load_tensor(tensor_path)
        array = onnx.numpy_helper.to_array(tensor)
        input_data[name] = array

        # Save input tensor to output directory
        output_tensor_path = os.path.join(output_dir, f"input_{i}.pb")
        onnx.save_tensor(tensor, output_tensor_path)
        print(f"    Shape: {array.shape}, dtype: {array.dtype}")
        print(f"    Saved to {output_tensor_path}")

    # Run the extracted model with onnxruntime to get expected outputs
    print(f"\nRunning extracted model with onnxruntime...")
    sess = rt.InferenceSession(output_model_path)
    result = sess.run(None, input_data)

    # Save output tensors
    print(f"\nSaving {len(result)} output tensor(s)...")
    for i, output_array in enumerate(result):
        output_tensor_path = os.path.join(output_dir, f"output_{i}.pb")
        output_tensor = onnx.numpy_helper.from_array(output_array)
        onnx.save_tensor(output_tensor, output_tensor_path)
        print(f"  Output {i}: shape={output_array.shape}, dtype={output_array.dtype}")
        print(f"    Saved to {output_tensor_path}")

    print(f"\n✓ Extraction complete! Test files saved to {output_dir}")
    print(f"  - model.onnx")
    print(f"  - input_*.pb (x{len(input_tensor_paths)})")
    print(f"  - output_*.pb (x{len(result)})")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a subgraph from an ONNX model and generate test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from model with single input/output
  %(prog)s \\
    --model models/validated/yolov4/yolov4.onnx \\
    --inputs models/validated/yolov4/test_data_set_0/input_0.pb \\
    --outputs "lambda_5/add:0" \\
    --output-dir models/extracted/yolov4/until_lambda_5_add

  # Extract with multiple inputs
  %(prog)s \\
    --model models/validated/bertsquad-12/bertsquad-12.onnx \\
    --inputs models/validated/bertsquad-12/test_data_set_0/input_*.pb \\
    --outputs "output_node_name" \\
    --output-dir models/extracted/bertsquad-12/until_output
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the input ONNX model"
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to input tensor .pb files"
    )

    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        help="Names of output nodes to extract up to"
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save extracted model and test data"
    )

    parser.add_argument(
        "--input-names",
        nargs="+",
        help="Names of input nodes (if different from original model inputs)"
    )

    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip model validation check"
    )

    args = parser.parse_args()

    extract_and_run(
        input_model_path=args.model,
        input_tensor_paths=args.inputs,
        output_dir=args.output_dir,
        output_names=args.outputs,
        input_names=args.input_names,
        check_model=not args.no_check
    )


if __name__ == "__main__":
    main()
