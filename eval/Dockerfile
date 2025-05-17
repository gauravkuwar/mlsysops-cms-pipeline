FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    prefect \
    mlflow \
    transformers \
    scikit-learn \
    pandas

RUN apt-get update && apt-get install -y git
ENV GIT_PYTHON_REFRESH=quiet

# Copy evaluation code
COPY flow.py /app/flow.py

# Run evaluation when container starts
ENTRYPOINT ["python", "flow.py"]
