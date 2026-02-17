#!/usr/bin/env python3
"""
List all nodes in an ONNX model to help find output node names for extraction.
"""

import argparse
import onnx


def list_nodes(model_path: str, filter_op: str = None, limit: int = None):
    """List all nodes in an ONNX model."""
    model = onnx.load(model_path)

    print(f"Model: {model_path}")
    print(f"Inputs: {[inp.name for inp in model.graph.input]}")
    print(f"Outputs: {[out.name for out in model.graph.output]}")
    print(f"\nTotal nodes: {len(model.graph.node)}\n")

    count = 0
    for i, node in enumerate(model.graph.node):
        if filter_op and node.op_type != filter_op:
            continue

        if limit and count >= limit:
            print(f"... (showing first {limit} nodes, total: {len(model.graph.node)})")
            break

        print(f"[{i}] {node.op_type}")
        if node.name:
            print(f"    Name: {node.name}")
        print(f"    Inputs: {list(node.input)}")
        print(f"    Outputs: {list(node.output)}")
        print()
        count += 1


def main():
    parser = argparse.ArgumentParser(
        description="List nodes in an ONNX model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "model",
        help="Path to ONNX model"
    )

    parser.add_argument(
        "--filter",
        help="Filter by operation type (e.g., 'Cast', 'Conv')"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Limit number of nodes to display (default: 50)"
    )

    args = parser.parse_args()
    list_nodes(args.model, args.filter, args.limit)


if __name__ == "__main__":
    main()
