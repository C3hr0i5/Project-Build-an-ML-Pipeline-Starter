import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
#    "test_regression_model"
]


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                version='main',
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        # basic_cleaning
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

# data_check
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

        if "data_split" in active_steps:
            _ = mlflow.run(
        f"{config['main']['components_repository']}/train_val_test_split",
        "main",
        parameters={
            "input_artifact": "clean_sample.csv:latest",
            "artifact_root": "data",
            "test_size": config["modeling"]["test_size"],
            "random_seed": config["modeling"]["random_seed"],
            "stratify": config["modeling"]["stratify"], 
        },
        env_manager="local",
    )

        if "train_random_forest" in active_steps:
            rf_config = os.path.abspath("rf_config.json")
    with open(rf_config, "w+") as fp:
        json.dump(dict(config["modeling"]["random_forest"].items()), fp)

    _ = mlflow.run(
        os.path.join(to_absolute_path("src"), "train_random_forest"),
        "main",
        parameters={
            "rf_config": rf_config,
            "train_data": "data_train.csv:latest",
            "val_data": "data_val.csv:latest",
            "target": config["modeling"]["target"],   
        },
        env_manager="local",
    )

if "test_regression_model" in active_steps:
    _ = mlflow.run(
        os.path.join(to_absolute_path("src"), "test_regression_model"),
        "main",
        parameters={
            # Uses the model artifact you promoted to "prod"
            "model_export": "model_export:prod",
            # Hold-out split produced by data_split
            "test_data": "data_test.csv:latest",
            # Same target you used during training
            "target": config["modeling"]["target"],
        },
        env_manager="local",
    )


if __name__ == "__main__":
    go()
