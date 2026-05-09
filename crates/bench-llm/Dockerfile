# Base image is publicly pullable from Docker Hub. NGC Deep Learning Container
# License (governs the image, not this Dockerfile):
#   https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
FROM docker.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir onnxruntime-genai-cuda==0.11.4

# For onnxruntime_genai.models.builder (export/quantization) and
# llama.cpp's convert_hf_to_gguf.py.
RUN pip3 install --no-cache-dir onnx-ir transformers torch onnx sentencepiece

WORKDIR /workspace
