import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # "test_regression_model",  # run explicitly when ready
]

@hydra.main(version_base=None, config_name="config")
def go(config: DictConfig):
    # Group runs in W&B
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    with tempfile.TemporaryDirectory() as tmp_dir:

        # -------- download --------
        if "download" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version="main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
                env_manager="local",
            )

        # ----- basic_cleaning -----
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(to_absolute_path("src"), "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": "sample.csv:latest",
                    "output_artifact": "clean_sample.csv",
                    "output_type": "clean_sample",
                    "output_description": "Cleaned sample (price filter, parsed dates, geo bounds)",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
                env_manager="local",
            )

        # -------- data_check -------
        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(to_absolute_path("src"), "data_check"),
                "main",
                parameters={
                    "csv": "clean_sample.csv:latest",
                    "ref": "clean_sample.csv:reference",
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
                env_manager="local",
            )

        # -------- data_split -------
        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input_artifact": "clean_sample.csv:latest",
                    "artifact_root": "data",
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify": config["modeling"]["stratify"],  # "" if none
                },
                env_manager="local",
            )

        # --- train_random_forest ---
        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                os.path.join(to_absolute_path("src"), "train_random_forest"),
                "main",
                parameters={
                    "rf_config": rf_config,
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify"],  
                    "max_tfidf_features": config["modeling"].get("max_tfidf_features", 10),
                    "output_artifact": "random_forest_export",
                },
                env_manager="local",
            )

        # (optional) test_regression_model
        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(to_absolute_path("src"), "test_regression_model"),
                "main",
                parameters={
                    "model_export": "random_forest_export:prod",
                    "test_data": "data_test.csv:latest",
                    "target": config["modeling"]["target"],
                },
                env_manager="local",
            )

if __name__ == "__main__":
    go()
