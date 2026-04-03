FROM docker.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir onnxruntime-gpu==1.20.1 onnx numpy

WORKDIR /workspace
COPY benchmark.py .
