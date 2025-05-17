import torch
import mlflow
import mlflow.onnx
import onnx
from transformers import AutoTokenizer
from mlflow.tracking import MlflowClient

MODEL_NAME = "mlsysops-cms-model"
SRC_ALIAS = "development"
DST_ALIAS = "staging"
ONNX_PATH = "/app/model.onnx"

client = MlflowClient()
version_info = client.get_model_version_by_alias(MODEL_NAME, SRC_ALIAS)
model_uri = f"models:/{MODEL_NAME}/{version_info.version}"

print(f"Loading PyTorch model from alias '{SRC_ALIAS}': {model_uri}")
model = mlflow.pytorch.load_model(model_uri)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
inputs = tokenizer("This is a test.", return_tensors="pt", padding=True, truncation=True, max_length=128)

print(f"Exporting ONNX to {ONNX_PATH}")
torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=16
)

# Log ONNX model and set alias
mlflow.onnx.log_model(
    onnx_model=onnx.load(ONNX_PATH),
    artifact_path="onnx_model",
    registered_model_name=MODEL_NAME
)

registered = mlflow.register_model(
    model_uri=f"runs:/{mlflow.active_run().info.run_id}/onnx_model",
    name=MODEL_NAME
)

client.set_registered_model_alias(MODEL_NAME, DST_ALIAS, registered.version)

print(f"Model exported to ONNX and aliased as '{DST_ALIAS}' (version {registered.version})")
