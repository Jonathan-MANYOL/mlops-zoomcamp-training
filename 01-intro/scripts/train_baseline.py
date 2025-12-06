#!/usr/bin/env python
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a baseline linear regression model for NYC taxi trip duration."
    )

    parser.add_argument(
        "--train-path",
        type=str,
        required=True,
        help="Path to the training parquet file (e.g. green_tripdata_2024-01.parquet)",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        required=True,
        help="Path to the validation parquet file (e.g. green_tripdata_2024-02.parquet)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="../models/linreg_model.bin",
        help="Where to save the trained model + DictVectorizer (pickle).",
    )

    return parser.parse_args()


def read_data(path: Path) -> pd.DataFrame:
    print(f"📂 Reading data from {path}")
    df = pd.read_parquet(path)

    # Duration in minutes
    df["duration"] = (
        df["lpep_dropoff_datetime"] - df["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    # Filter outliers
    df = df[(df["duration"] >= 1) & (df["duration"] <= 60)].copy()

    # Cast categorical features to string
    categorical = ["PULocationID", "DOLocationID"]
    for col in categorical:
        df[col] = df[col].astype(str)

    print(
        f"✅ Loaded {len(df)} rows after filtering duration between 1 and 60 minutes."
    )
    return df


def prepare_features(
    df: pd.DataFrame,
    dv: DictVectorizer | None = None,
    fit_dv: bool = False,
):
    features = ["PULocationID", "DOLocationID", "trip_distance"]
    target = "duration"

    dicts = df[features].to_dict(orient="records")
    y = df[target].values

    if dv is None:
        dv = DictVectorizer(sparse=True)

    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, y, dv


def main():
    args = parse_args()

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    model_path = Path(args.model_path)

    # 1) Load data
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # 2) Prepare features
    print("🔧 Preparing features...")
    X_train, y_train, dv = prepare_features(df_train, dv=None, fit_dv=True)
    X_val, y_val, _ = prepare_features(df_val, dv=dv, fit_dv=False)

    print(f"🔹 X_train shape: {X_train.shape}")
    print(f"🔹 X_val   shape: {X_val.shape}")

    # 3) Train model
    print("🧠 Training LinearRegression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4) Evaluate
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred)
    print(f"📊 Validation RMSE: {rmse:.2f} minutes")

    # 5) Save model + DictVectorizer
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print(f"💾 Saved model + DictVectorizer to {model_path.resolve()}")


if __name__ == "__main__":
    main()

## How to run the script
# cd ~/Projects/mlops-zoomcamp-training
# source .venv/bin/activate

# # Make sure you have the parquet files in 01-intro/data/
# # (use the real URLs from the homework)
# ls 01-intro/data
# # -> green_tripdata_2024-01.parquet  green_tripdata_2024-02.parquet  (for example)

# python 01-intro/scripts/train_baseline.py \
#   --train-path 01-intro/data/green_tripdata_2024-01.parquet \
#   --val-path   01-intro/data/green_tripdata_2024-02.parquet \
#   --model-path 01-intro/models/linreg_model.bin
