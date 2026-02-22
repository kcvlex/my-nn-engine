#!/usr/bin/env python3
"""
Binary search helpers for finding the node that produces incorrect results.

Provides two subcommands:
  list-nodes    — print total node count
  extract-node  — extract submodel up to a given node and run onnxruntime
"""

import argparse
import os
import sys

import onnx
from onnx.utils import extract_model
import onnxruntime as rt


def cmd_list_nodes(args):
    """Print total node count to stdout."""
    model = onnx.load(args.model)
    print(len(model.graph.node))


def cmd_extract_node(args):
    """Extract submodel up to node_index, run onnxruntime, save results."""
    model = onnx.load(args.model)
    node_index = args.node_index

    if node_index < 0 or node_index >= len(model.graph.node):
        print(f"Invalid node index: {node_index}", file=sys.stderr)
        sys.exit(1)

    node = model.graph.node[node_index]
    input_names = [inp.name for inp in model.graph.input]
    output_names = list(node.output)

    if not output_names:
        print(f"Node {node_index} has no outputs", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract the model
    output_model_path = os.path.join(args.output_dir, "model.onnx")
    try:
        extract_model(
            args.model,
            output_model_path,
            input_names,
            output_names,
            check_model=False,
        )
    except Exception as e:
        print(f"Failed to extract: {e}", file=sys.stderr)
        sys.exit(1)

    # Load inputs and run with ONNX Runtime
    try:
        input_data = {}
        for i, (name, tensor_path) in enumerate(zip(input_names, args.inputs)):
            tensor = onnx.load_tensor(tensor_path)
            array = onnx.numpy_helper.to_array(tensor)
            input_data[name] = array

            # Save input tensor
            onnx.save_tensor(tensor, os.path.join(args.output_dir, f"input_{i}.pb"))

        sess = rt.InferenceSession(output_model_path)
        result = sess.run(None, input_data)
    except Exception as e:
        print(f"Failed to prepare inputs or run ONNX Runtime: {e}", file=sys.stderr)
        sys.exit(1)

    # Save expected outputs
    for i, output_array in enumerate(result):
        output_tensor = onnx.numpy_helper.from_array(output_array)
        onnx.save_tensor(output_tensor, os.path.join(args.output_dir, f"output_{i}.pb"))

    # Print node info to stdout
    print(f"{node.op_type} ({node.name})")


def main():
    parser = argparse.ArgumentParser(
        description="Binary search helpers for ONNX node debugging",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-nodes
    p_list = subparsers.add_parser("list-nodes", help="Print total node count")
    p_list.add_argument("--model", required=True, help="Path to the ONNX model")

    # extract-node
    p_extract = subparsers.add_parser(
        "extract-node", help="Extract submodel up to a given node index"
    )
    p_extract.add_argument("--model", required=True, help="Path to the ONNX model")
    p_extract.add_argument(
        "--inputs", nargs="+", required=True, help="Paths to input tensor .pb files"
    )
    p_extract.add_argument(
        "--node-index", type=int, required=True, help="Node index to extract up to"
    )
    p_extract.add_argument(
        "--output-dir", required=True, help="Directory to save extracted model and data"
    )

    args = parser.parse_args()

    if args.command == "list-nodes":
        cmd_list_nodes(args)
    elif args.command == "extract-node":
        cmd_extract_node(args)


if __name__ == "__main__":
    main()
