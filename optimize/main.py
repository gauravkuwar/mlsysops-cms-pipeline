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
MAX_LEN = 128

client = MlflowClient()
version_info = client.get_model_version_by_alias(MODEL_NAME, SRC_ALIAS)
model_uri = f"models:/{MODEL_NAME}/{version_info.version}"

print(f"Loading PyTorch model from alias '{SRC_ALIAS}': {model_uri}")
model = mlflow.pytorch.load_model(model_uri)
model.eval()

# Tokenizer + dummy input
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
dummy_input_ids = torch.randint(
    low=0,
    high=tokenizer.vocab_size,
    size=(1, MAX_LEN),
    dtype=torch.long,
    device=model.device
)
dummy_attention_mask = torch.ones(
    (1, MAX_LEN),
    dtype=torch.long,
    device=model.device
)

print(f"Exporting ONNX to {ONNX_PATH}")
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":       {0: "batch_size", 1: "seq_len"},
        "attention_mask":  {0: "batch_size", 1: "seq_len"},
        "logits":          {0: "batch_size"}
    }
)

# Sanity check
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

# Log and register
mlflow.onnx.log_model(
    onnx_model=onnx_model,
    artifact_path="onnx_model",
    registered_model_name=MODEL_NAME
)

registered = mlflow.register_model(
    model_uri=f"runs:/{mlflow.active_run().info.run_id}/onnx_model",
    name=MODEL_NAME
)

client.set_registered_model_alias(MODEL_NAME, DST_ALIAS, registered.version)

print(f"Model exported to ONNX and aliased as '{DST_ALIAS}' (version {registered.version})")
