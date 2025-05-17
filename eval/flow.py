import torch
import mlflow
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

MODEL_NAME = "mlsysops-cms-model"
ALIAS = "development"
STAGING_THRESHOLD = 0.80

@task
def load_model_and_tokenizer():
    logger = get_run_logger()
    logger.info(f"Loading model from alias: {MODEL_NAME}/{ALIAS}")
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    model_uri = f"models:/{MODEL_NAME}/{version_info.version}"
    model = mlflow.pytorch.load_model(model_uri)
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    model.eval()
    return model, tokenizer

@task
def load_test_data():
    texts = ["you are amazing", "you are terrible"]
    labels = [1, 0]
    return texts, labels

@task
def run_evaluation(model, tokenizer, texts, labels):
    logger = get_run_logger()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1).tolist()

    acc = accuracy_score(labels, preds)
    logger.info(f"Evaluation accuracy: {acc:.4f}")
    mlflow.log_metric("eval_accuracy", acc)
    return acc

@task
def promote_to_staging_if_good(acc: float):
    logger = get_run_logger()
    if acc < STAGING_THRESHOLD:
        logger.info("Model did not meet staging threshold.")
        return None

    logger.info("Promoting model to 'Staging' stage and 'staging' alias")
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, ALIAS).version
        client.transition_model_version_stage(MODEL_NAME, version, stage="Staging")
        client.set_registered_model_alias(MODEL_NAME, "staging", version)
        logger.info(f"Promoted version {version} to 'Staging' and aliased as 'staging'")
        return version
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return None

@flow(name="evaluation_flow")
def evaluation_flow():
    with mlflow.start_run(run_name="evaluation"):
        model, tokenizer = load_model_and_tokenizer()
        texts, labels = load_test_data()
        acc = run_evaluation(model, tokenizer, texts, labels)
        version = promote_to_staging_if_good(acc)
        return version

if __name__ == "__main__":
    evaluation_flow()
