import argparse
import logging
import os
import numpy as np
import pandas as pd
import wandb
import mlflow.sklearn as msk
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("test_regression_model")

def _resolve_mlflow_model_dir(root: str) -> str:
    """
    Find the MLflow model directory (contains 'MLmodel') inside a wandb artifact download.
    """
    # direct path?
    if os.path.isfile(os.path.join(root, "MLmodel")):
        return root
    # common subdir name we used when logging
    sub = os.path.join(root, "random_forest_dir")
    if os.path.isfile(os.path.join(sub, "MLmodel")):
        return sub
    # fallback: search children
    for name in os.listdir(root):
        candidate = os.path.join(root, name)
        if os.path.isfile(os.path.join(candidate, "MLmodel")):
            return candidate
    raise FileNotFoundError("Could not locate MLflow model directory (no 'MLmodel' file found).")

def go(args):
    run = wandb.init(job_type="test_regression_model")
    run.config.update(vars(args))

    # Fetch model artifact (directory)
    logger.info("Downloading model artifact: %s", args.model_export)
    model_art = run.use_artifact(args.model_export)
    model_root = model_art.download()
    model_path = _resolve_mlflow_model_dir(model_root)
    logger.info("Loading model from %s", model_path)
    model = msk.load_model(model_path)

    # Fetch test data CSV
    logger.info("Downloading test data artifact: %s", args.test_data)
    test_art = run.use_artifact(args.test_data)
    test_csv = test_art.file()
    df = pd.read_csv(test_csv)

    y = df[args.target].values
    X = df.drop(columns=[args.target])

    logger.info("Scoring on hold-out")
    preds = model.predict(X)
    mae = float(mean_absolute_error(y, preds))
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    r2  = float(r2_score(y, preds))

    logger.info("MAE=%.4f RMSE=%.4f R2=%.4f", mae, rmse, r2)
    wandb.log({"mae_test": mae, "rmse_test": rmse, "r2_test": r2})
    run.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate exported model on hold-out test set")
    p.add_argument("--model_export", type=str, required=True, help="Model artifact, e.g. random_forest_export:prod")
    p.add_argument("--test_data", type=str, required=True, help="Test CSV artifact, e.g. data_test.csv:latest")
    p.add_argument("--target", type=str, required=True, help="Target column name (e.g., price)")
    args = p.parse_args()
    go(args)
