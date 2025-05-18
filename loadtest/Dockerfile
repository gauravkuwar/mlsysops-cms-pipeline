FROM python:3.11-slim

WORKDIR /app
COPY main.py .

RUN pip install --no-cache-dir httpx mlflow

ENV GIT_PYTHON_REFRESH=quiet
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]