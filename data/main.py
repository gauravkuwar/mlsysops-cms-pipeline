import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile
import time

os.environ["KAGGLE_CONFIG_DIR"] = os.path.abspath("/root/.kaggle")
from kaggle.api.kaggle_api_extended import KaggleApi

def download_jigsaw(kaggle_dir):
    print(f"[INFO] Preparing to download Jigsaw dataset to: {kaggle_dir}")
    os.makedirs(kaggle_dir, exist_ok=True)

    print("[INFO] Authenticating Kaggle API...")
    api = KaggleApi()
    api.authenticate()

    print("[INFO] Downloading competition files...")
    api.competition_download_files(
        "jigsaw-unintended-bias-in-toxicity-classification",
        path=kaggle_dir
    )

    zip_path = os.path.join(kaggle_dir, "jigsaw-unintended-bias-in-toxicity-classification.zip")
    print(f"[INFO] Unzipping file: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(kaggle_dir)
    print("[SUCCESS] Downloaded and extracted Jigsaw dataset.")

def preprocess(kaggle_dir, output_dir, val_ratio, eval_ratio):
    print(f"[INFO] Starting preprocessing...")
    os.makedirs(output_dir, exist_ok=True)
    input_path = os.path.join(kaggle_dir, "train.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[ERROR] train.csv not found in {kaggle_dir}")
    
    print(f"[INFO] Reading CSV in chunks from: {input_path}")
    start = time.time()

    chunks = pd.read_csv(
        input_path,
        usecols=["comment_text", "target"],
        low_memory=False,
        chunksize=100_000
    )

    filtered_chunks = []
    total_rows = 0
    for i, chunk in enumerate(chunks):
        chunk.dropna(subset=["comment_text"], inplace=True)
        filtered_chunks.append(chunk)
        total_rows += len(chunk)
        print(f"[INFO] Processed chunk {i + 1}, running total: {total_rows} rows")

    df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"[INFO] Loaded {len(df)} filtered rows in {time.time() - start:.2f}s")

    print(f"[INFO] Splitting eval ({eval_ratio * 100:.1f}%)...")
    df, eval_df = train_test_split(df, test_size=eval_ratio, random_state=42)

    print(f"[INFO] Splitting val ({val_ratio * 100:.1f}%) from remaining data...")
    train_df, val_df = train_test_split(df, test_size=val_ratio / (1 - eval_ratio), random_state=42)

    print(f"[INFO] Writing preprocessed output to {output_dir}...")
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    eval_df.to_csv(os.path.join(output_dir, "offline_eval.csv"), index=False)

    print(f"[SUCCESS] Saved {len(train_df)} train, {len(val_df)} val, and {len(eval_df)} eval samples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--eval-split", type=float, default=0.01)
    args = parser.parse_args()

    print("[START] Running Jigsaw data pipeline")
    download_jigsaw(args.kaggle_dir)
    preprocess(args.kaggle_dir, args.output_dir, args.val_split, args.eval_split)
    print("[COMPLETE] Data pipeline finished successfully.")
