FROM python:3.10-slim

WORKDIR /app

COPY main.py /app/main.py

# Install dependencies
RUN pip install --no-cache-dir kaggle pandas scikit-learn

# Default CMD
ENTRYPOINT ["python", "main.py"]