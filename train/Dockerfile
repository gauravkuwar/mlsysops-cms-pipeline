FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    prefect \
    mlflow \
    fastapi \
    uvicorn \
    transformers

RUN apt-get update && apt-get install -y git
ENV GIT_PYTHON_REFRESH=quiet

# Copy application files
COPY flow.py /app/flow.py
# COPY model.pth /app/model.pth

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI server
ENTRYPOINT ["python", "flow.py"]
