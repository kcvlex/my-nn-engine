#!/usr/bin/env python3
"""
Binary search to find the node that produces incorrect results.

This script performs binary search on nodes to find where the implementation
diverges from ONNX Runtime's expected output.
"""

import argparse
import os
import sys
import subprocess

import onnx
from onnx.utils import extract_model
import onnxruntime as rt


def get_node_info(model_path: str, node_index: int):
    """Get information about a node at a given index."""
    model = onnx.load(model_path)
    if node_index < 0 or node_index >= len(model.graph.node):
        raise ValueError(f"Invalid node index: {node_index}")

    node = model.graph.node[node_index]
    return {
        'index': node_index,
        'op_type': node.op_type,
        'name': node.name,
        'outputs': list(node.output),
    }


def extract_and_test(
    model_path: str,
    input_tensor_paths: list[str],
    node_index: int,
    test_command: list[str],
    temp_dir: str,
) -> tuple[bool, dict]:
    """
    Extract subgraph up to node_index and test it.

    Returns:
        (passes, node_info): Whether the test passes and node information
    """
    model = onnx.load(model_path)
    node_info = get_node_info(model_path, node_index)

    # Get input names
    input_names = [inp.name for inp in model.graph.input]

    # Get output names from the node
    output_names = node_info['outputs']
    if not output_names:
        return False, node_info

    # Create a temporary directory for this extraction
    extract_dir = os.path.join(temp_dir, f"node_{node_index}")
    os.makedirs(extract_dir, exist_ok=True)

    # Extract the model
    output_model_path = os.path.join(extract_dir, "model.onnx")
    try:
        extract_model(
            model_path,
            output_model_path,
            input_names,
            output_names,
            check_model=False
        )
    except Exception as e:
        print(f"  Failed to extract: {e}")
        return False, node_info

    # Load and run with ONNX Runtime to get expected output
    try:
        input_data = {}
        for i, (name, tensor_path) in enumerate(zip(input_names, input_tensor_paths)):
            tensor = onnx.load_tensor(tensor_path)
            array = onnx.numpy_helper.to_array(tensor)
            input_data[name] = array

            # Save input tensor
            output_tensor_path = os.path.join(extract_dir, f"input_{i}.pb")
            onnx.save_tensor(tensor, output_tensor_path)

        # Run with ONNX Runtime
        sess = rt.InferenceSession(output_model_path)
        result = sess.run(None, input_data)
    except Exception as e:
        print(f"  Failed to prepare inputs or run ONNX Runtime: {e}")
        return False, node_info

    # Save expected outputs
    for i, output_array in enumerate(result):
        output_tensor_path = os.path.join(extract_dir, f"output_{i}.pb")
        output_tensor = onnx.numpy_helper.from_array(output_array)
        onnx.save_tensor(output_tensor, output_tensor_path)

    # Run the test command
    # Replace placeholders in test command
    test_cmd = []
    for arg in test_command:
        if arg == "{extract_dir}":
            test_cmd.append(extract_dir)
        elif arg == "{num_inputs}":
            test_cmd.append(str(len(input_names)))
        elif arg == "{num_outputs}":
            test_cmd.append(str(len(result)))
        else:
            test_cmd.append(arg)

    print(f"  Running: {' '.join(test_cmd)}")
    result = subprocess.run(test_cmd, capture_output=True)
    passes = result.returncode == 0

    if not passes and result.stdout:
        print(f"  stdout: {result.stdout.decode()[:200]}")
    if not passes and result.stderr:
        print(f"  stderr: {result.stderr.decode()[:200]}")

    return passes, node_info


def binary_search_nodes(
    model_path: str,
    input_tensor_paths: list[str],
    test_command: list[str],
    temp_dir: str,
) -> dict | None:
    """
    Perform binary search to find the first failing node.

    Returns:
        Information about the first failing node
    """
    model = onnx.load(model_path)
    total_nodes = len(model.graph.node)

    print(f"Model: {model_path}")
    print(f"Total nodes: {total_nodes}")
    print(f"Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"Outputs: {[out.name for out in model.graph.output]}")
    print("\nStarting binary search...\n")

    left, right = 0, total_nodes - 1
    first_fail = None

    while left <= right:
        mid = (left + right) // 2
        node_info = get_node_info(model_path, mid)

        print(f"Testing node [{mid}/{total_nodes-1}] {node_info['op_type']} ({node_info['name']})")

        passes, node_info = extract_and_test(
            model_path,
            input_tensor_paths,
            mid,
            test_command,
            temp_dir
        )

        if passes:
            print("  PASS\n")
            left = mid + 1
        else:
            print("  FAIL\n")
            first_fail = node_info
            right = mid - 1

    return first_fail


def main():
    parser = argparse.ArgumentParser(
        description="Binary search to find the node producing incorrect results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  %(prog)s \\
    --model models/validated/bertsquad-12/bertsquad-12.onnx \\
    --inputs models/validated/bertsquad-12/test_data_set_0/input_*.pb \\
    --test-command "cargo" "test" "--test" "extracted_models" \\
                   "test_extracted_{extract_dir}" "--" "--nocapture"

The test command can use these placeholders:
  {extract_dir}   - Path to the extracted model directory
  {num_inputs}    - Number of inputs
  {num_outputs}   - Number of outputs
        """
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to the ONNX model"
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Paths to input tensor .pb files"
    )

    parser.add_argument(
        "--test-command",
        nargs="+",
        required=True,
        help="Command to run tests (can use {extract_dir}, {num_inputs}, {num_outputs})"
    )

    parser.add_argument(
        "--temp-dir",
        default="/tmp/binary_search_nodes",
        help="Temporary directory for extracted models (default: /tmp/binary_search_nodes)"
    )

    args = parser.parse_args()

    # Create temp directory
    os.makedirs(args.temp_dir, exist_ok=True)

    # Run binary search
    first_fail = binary_search_nodes(
        args.model,
        args.inputs,
        args.test_command,
        args.temp_dir
    )

    if first_fail:
        print("\n" + "="*80)
        print("FOUND: First failing node")
        print("="*80)
        print(f"Index:   {first_fail['index']}")
        print(f"OpType:  {first_fail['op_type']}")
        print(f"Name:    {first_fail['name']}")
        print(f"Outputs: {first_fail['outputs']}")
        print("\nExtracted model saved in:")
        print(f"  {args.temp_dir}/node_{first_fail['index']}/")
        sys.exit(1)
    else:
        print("\n" + "="*80)
        print("All nodes pass! No failing node found.")
        print("="*80)
        sys.exit(0)


if __name__ == "__main__":
    main()
