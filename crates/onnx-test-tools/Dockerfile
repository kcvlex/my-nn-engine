FROM python:3.11-slim

# Install ONNX tools (as root)
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime \
    numpy

# Copy the extraction scripts (as root)
COPY extract_model.py /usr/local/bin/extract_model.py
COPY binary_search.py /usr/local/bin/binary_search.py
RUN chmod +x /usr/local/bin/extract_model.py /usr/local/bin/binary_search.py

# Create a non-root user to run the application
# Using UID/GID 1000 as default, will be overridden at runtime via --userns=keep-id
RUN groupadd -g 1000 onnxuser && \
    useradd -m -u 1000 -g onnxuser onnxuser

# Set working directory and ensure it's writable by onnxuser
WORKDIR /workspace
RUN chown onnxuser:onnxuser /workspace

# Switch to non-root user
USER onnxuser

ENTRYPOINT ["python", "/usr/local/bin/extract_model.py"]
