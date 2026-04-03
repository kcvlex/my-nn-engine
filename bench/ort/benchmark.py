import argparse
import time
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper


MODELS_DIR = "/workspace/models/validated"

DEFAULT_MODELS = [
    "resnet18-v2-7",
    "resnet152-v2-7",
]

SPECIAL_MODEL_FILES = {
    "GPT2": "model.onnx",
}


def load_model_path(model_name):
    filename = SPECIAL_MODEL_FILES.get(model_name, f"{model_name}.onnx")
    return os.path.join(MODELS_DIR, model_name, filename)


def load_test_input(model_name):
    pb_path = os.path.join(
        MODELS_DIR, model_name, "test_data_set_0", "input_0.pb"
    )
    tensor = TensorProto()
    with open(pb_path, "rb") as f:
        tensor.ParseFromString(f.read())
    return numpy_helper.to_array(tensor)


def benchmark_model(model_name, providers, num_runs):
    model_path = load_model_path(model_name)
    input_data = load_test_input(model_name)

    sess = ort.InferenceSession(model_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    # Warmup
    sess.run(None, {input_name: input_data})

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        sess.run(None, {input_name: input_data})
    elapsed_ms = (time.perf_counter() - start) * 1000

    avg_ms = elapsed_ms / num_runs
    return avg_ms


def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime benchmark")
    parser.add_argument(
        "target",
        choices=["cpu", "cuda"],
        help="Execution provider target",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    args = parser.parse_args()

    if args.target == "cpu":
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    print(f"Provider: {args.target.upper()}")
    print(f"Runs: {args.runs}")
    print()

    for model_name in args.models:
        try:
            avg_ms = benchmark_model(model_name, providers, args.runs)
            print(f"{model_name}: {avg_ms:.2f} ms/run")
        except Exception as e:
            print(f"{model_name}: ERROR - {e}")


if __name__ == "__main__":
    main()
