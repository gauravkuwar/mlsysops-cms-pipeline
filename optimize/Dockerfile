FROM python:3.11-slim

WORKDIR /app

# Install base ML packages
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Additional packages
RUN pip install --no-cache-dir \
    mlflow \
    transformers \
    onnx \
    onnxruntime

# Git is needed for transformers + MLflow operations
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV GIT_PYTHON_REFRESH=quiet

# Copy ONNX export script
COPY main.py /app/main.py

# Entry point
ENTRYPOINT ["python", "main.py"]