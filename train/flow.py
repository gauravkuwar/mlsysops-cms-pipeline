import os
import torch
import mlflow
import asyncio
from fastapi import FastAPI, HTTPException
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "mlsysops-cms-model"

app = FastAPI()
pipeline_lock = asyncio.Lock()

mock_config = {
    "initial_epochs": 2,
    "total_epochs": 1,
    "patience": 2,
    "batch_size": 128,
    "lr": 2e-5,
    "fine_tune_lr": 1e-5,
    "max_len": 128,
    "dropout_probability": 0.3,
    "model_name": "google/bert_uncased_L-2_H-128_A-2"
}

@task
def load_and_train_model():
    logger = get_run_logger()
    logger.info("Pretending to train, actually just loading a model (because this is not my part)...")

    model = AutoModelForSequenceClassification.from_pretrained(
        mock_config["model_name"],
        num_labels=1
    )

    tokenizer = AutoTokenizer.from_pretrained(mock_config["model_name"])
    inputs = tokenizer("Example input for classification", return_tensors="pt", padding=True, truncation=True)

    input_example = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }

    logger.info("Logging model and config to MLflow...")

    mlflow.log_param("run_type", "mock")
    for k, v in mock_config.items():
        mlflow.log_param(k, v)

    mlflow.pytorch.log_model(model, artifact_path="model", input_example=input_example)

    return model

@task
def register_model():
    logger = get_run_logger()

    logger.info("Registering model in MLflow Model Registry...")
    run = mlflow.active_run()
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    client = MlflowClient()

    try:
        client.get_registered_model(MODEL_NAME)
    except mlflow.exceptions.RestException:
        client.create_registered_model(MODEL_NAME)

    model_version = client.create_model_version(name=MODEL_NAME, source=model_uri, run_id=run_id)

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="development",
        version=model_version.version
    )

    logger.info(f"Model registered (v{model_version.version}) and alias 'development' assigned.")
    return model_version.version

@flow(name="mlflow_flow")
def ml_pipeline_flow():
    with mlflow.start_run():
        load_and_train_model()
        version = register_model()
        return version

@app.post("/trigger-training")
async def trigger_training():
    if pipeline_lock.locked():
        raise HTTPException(status_code=423, detail="Pipeline is already running. Please wait.")

    async with pipeline_lock:
        loop = asyncio.get_event_loop()
        version = await loop.run_in_executor(None, ml_pipeline_flow)
        if version:
            return {"status": "Pipeline executed successfully", "new_model_version": version}
        else:
            return {"status": "Pipeline executed, but no new model registered"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
