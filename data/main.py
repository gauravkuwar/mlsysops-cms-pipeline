import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath("/root/.kaggle")
from kaggle.api.kaggle_api_extended import KaggleApi

def download_jigsaw(kaggle_dir):
    os.makedirs(kaggle_dir, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(
        "jigsaw-unintended-bias-in-toxicity-classification",
        path=kaggle_dir
    )

    zip_path = os.path.join(kaggle_dir, "jigsaw-unintended-bias-in-toxicity-classification.zip")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(kaggle_dir)
    print("Downloaded and extracted Jigsaw dataset.")

def preprocess(kaggle_dir, output_dir, val_ratio, eval_ratio):
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(kaggle_dir, "train.csv")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"train.csv not found in {kaggle_dir}")
    
    df = pd.read_csv(input_path).dropna(subset=["comment_text"])
    df = df[["comment_text", "target"]]
    df, eval_df = train_test_split(df, test_size=eval_ratio, random_state=42)
    train_df, val_df = train_test_split(df, test_size=val_ratio / (1 - eval_ratio), random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, "offline_eval.csv"), index=False)

    print(f"Saved {len(train_df)} train, {len(val_df)} val, and {len(eval_df)} eval samples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-split", type=float, default=0.01)
    args = parser.parse_args()

    download_jigsaw(args.kaggle_dir)
    preprocess(args.kaggle_dir, args.output_dir, args.val_split, args.eval_split)
