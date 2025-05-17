import torch
import mlflow
from prefect import flow, task, get_run_logger
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import pandas as pd

MODEL_NAME = "mlsysops-cms-model"
ALIAS = "development"
STAGING_ACC_THRESHOLD = 0.80

mock_config = {
    "batch_size": 32,
    "max_len": 128,
    "model_name": "google/bert_uncased_L-2_H-128_A-2"
}

@task
def load_model_and_tokenizer():
    logger = get_run_logger()
    logger.info(f"Loading model from alias: {MODEL_NAME}/{ALIAS}")
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    model_uri = f"models:/{MODEL_NAME}/{version_info.version}"
    model = mlflow.pytorch.load_model(model_uri)
    tokenizer = AutoTokenizer.from_pretrained(mock_config["model_name"])
    model.eval()
    return model, tokenizer

@task
def run_slice_eval(model, tokenizer) -> bool:
    logger = get_run_logger()

    identity_terms = ["gay", "muslim", "jewish", "woman", "trans", "black"]

    # Example test data with identity mentions
    test_data = [
        ("I love my gay friends", 1),
        ("Muslim people are kind", 1),
        ("He is a disgusting Muslim", 0),
        ("That Jewish guy is annoying", 0),
        ("She is a strong woman", 1),
        ("Women are the worst", 0),
        ("Black people are amazing", 1),
        ("All black people are criminals", 0),
        ("You are cool", 1),
        ("You are horrible", 0),
    ]

    texts = [t for t, _ in test_data]
    labels = [l for _, l in test_data]
    overall_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        overall_preds = model(**overall_inputs).logits.argmax(dim=1).tolist()
    overall_acc = accuracy_score(labels, overall_preds)

    logger.info(f"Overall accuracy: {overall_acc:.2f}")
    mlflow.log_metric("slice_overall_accuracy", overall_acc)

    passed = True
    for term in identity_terms:
        slice_indices = [i for i, text in enumerate(texts) if term in text.lower()]
        if not slice_indices:
            continue

        slice_texts = [texts[i] for i in slice_indices]
        slice_labels = [labels[i] for i in slice_indices]

        inputs = tokenizer(slice_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            preds = model(**inputs).logits.argmax(dim=1).tolist()

        acc = accuracy_score(slice_labels, preds)
        mlflow.log_metric(f"accuracy_{term}", acc)
        logger.info(f"{term} slice accuracy: {acc:.2f}")

        # Fail if drop is >10%
        if acc < overall_acc - 0.10:
            logger.info(f"Accuracy drop for '{term}' slice too high.")
            passed = False

    return passed

@task
def run_failure_mode_eval(model, tokenizer) -> bool:
    logger = get_run_logger()

    test_cases = [
        # Negation
        ("I don't think he's toxic at all", 1),
        ("She’s not rude", 1),
        ("I wouldn’t call him an idiot", 1),

        # Sarcasm
        ("Oh great, another genius plan", 0),
        ("Wow, just what we needed. A complete moron.", 0),

        # Spelling noise
        ("u r a looserrr", 0),
        ("shutttt uppp", 0),

        # Toxicity at the end
        ("He helped her all day and then called her stupid", 0),

        # Long benign context
        ("She was nice. They had a picnic. Then he said she's garbage.", 0),
        ("Everyone had fun. He ruined it by yelling insults.", 0),

        # Control
        ("You are a wonderful human", 1),
        ("They worked really well together", 1),
    ]

    texts = [t for t, _ in test_cases]
    labels = [l for _, l in test_cases]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        preds = model(**inputs).logits.argmax(dim=1).tolist()

    correct = sum([p == l for p, l in zip(preds, labels)])
    total = len(labels)
    acc = correct / total

    logger.info(f"Failure mode test accuracy: {acc:.2f} ({correct}/{total})")
    mlflow.log_metric("failure_mode_accuracy", acc)

    return acc >= 0.8

@task
def run_template_tests(model, tokenizer) -> bool:
    logger = get_run_logger()
    
    # (text, expected_label) where 1 = non-toxic, 0 = toxic
    test_cases = [
        ("You are a genius", 1),
        ("You are a kind person", 1),
        ("You are a moron", 0),
        ("You are disgusting", 0),
        ("He is annoying", 0),
        ("She is annoying", 0),
        ("They are nice", 1),
        ("They are horrible", 0),
        ("I think you are smart", 1),
        ("I think you are stupid", 0),
    ]
    
    texts = [x[0] for x in test_cases]
    expected = [x[1] for x in test_cases]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=1).tolist()

    correct = sum([p == e for p, e in zip(preds, expected)])
    total = len(expected)
    pass_rate = correct / total

    logger.info(f"Template unit test pass rate: {pass_rate:.2f} ({correct}/{total})")
    mlflow.log_metric("template_test_pass_rate", pass_rate)

    return pass_rate >= 0.9

@task
def all_checks_pass(acc: float, slice_pass: bool, fail_pass: bool, template_pass: bool) -> bool:
    logger = get_run_logger()
    if acc < STAGING_ACC_THRESHOLD or not (slice_pass and fail_pass and template_pass):
        logger.info("Model failed one or more checks.")
        return False
    logger.info("All checks passed.")
    return True

@task
def load_offline_eval_data():
    df = pd.read_csv("/mnt/data/preprocessed/offline_eval.csv")
    texts = df["comment_text"].tolist()
    labels = (df["target"] >= 0.5).astype(int).tolist()
    return texts, labels

@task
def run_evaluation(model, tokenizer, texts, labels) -> float:
    logger = get_run_logger()
    batch_size = mock_config["batch_size"]
    all_preds = []

    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=mock_config["max_len"])
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=1).tolist()
            all_preds.extend(preds)

    acc = accuracy_score(labels, all_preds)
    logger.info(f"Evaluation accuracy: {acc:.4f}")
    mlflow.log_metric("eval_accuracy", acc)
    return acc

@task
def promote_to_staging_if_good():
    logger = get_run_logger()
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
    with mlflow.start_run(run_name="offline-evaluation"):
        model, tokenizer = load_model_and_tokenizer()
        texts, labels = load_offline_eval_data()
        acc = run_evaluation(model, tokenizer, texts, labels)
        slice_pass = run_slice_eval(model, tokenizer)
        fail_pass = run_failure_mode_eval(model, tokenizer)
        template_pass = run_template_tests(model, tokenizer)
        checks_ok = all_checks_pass(acc, slice_pass, fail_pass, template_pass)

        version = None
        if checks_ok:
            version = promote_to_staging_if_good()
        return version

if __name__ == "__main__":
    evaluation_flow()
