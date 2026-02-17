FROM python:3.11-slim

WORKDIR /workspace

# Install ONNX tools
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime \
    numpy

# Copy the extraction script
COPY extract_model.py /usr/local/bin/extract_model.py
RUN chmod +x /usr/local/bin/extract_model.py

ENTRYPOINT ["python", "/usr/local/bin/extract_model.py"]
