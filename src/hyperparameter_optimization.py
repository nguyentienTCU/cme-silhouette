
"""Hyperparameter optimization for CME segmentation models."""

from copy import deepcopy
from pathlib import Path

import optuna

from .configuration import PROJECT_ROOT, load_config
from .trainer import run_training


def run_optimization(config):
    base_config = load_config(
        PROJECT_ROOT / config["base_config"]
    )

    model_name = config["model_name"]
    study_name = config["study_name"]
    search_space = config["search_space"]

    study_dir = PROJECT_ROOT / "outputs" / model_name / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)

    storage_path = (study_dir / "study.db").resolve()

    pruner = optuna.pruners.MedianPruner(**config.get("pruning", {}))
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path.as_posix()}",
        load_if_exists=True,
        pruner=pruner,
    )

    def objective(trial):
        trial_config = deepcopy(base_config)

        lr_space = search_space["learning_rate"]
        batch_space = search_space["batch_size"]
        alpha_space = search_space["alpha_loss"]

        trial_config["learning_rate"] = trial.suggest_float(
            "learning_rate",
            lr_space["low"],
            lr_space["high"],
            log=True,
        )

        trial_config["batch_size"] = trial.suggest_int(
            "batch_size",
            batch_space["low"],
            batch_space["high"],
            step=batch_space["step"],
)

        trial_config["alpha_loss"] = trial.suggest_float(
            "alpha_loss",
            alpha_space["low"],
            alpha_space["high"],
            step=alpha_space["step"],
        )

        trial_config["run_name"] = (
            f"{study_name}_trial_{trial.number}"
        )
        trial_config["experiment_type"] = "experiments"

        result = run_training(
            trial_config,
            output_dir=study_dir / "trials" / trial_config["run_name"],
            save_checkpoints=False,
            visualize=False,
            show=False,
            trial=trial,
        )

        score = result["metrics"]["best_val_iou"]
        trial.set_user_attr("output_dir", result["output_dir"])

        return score

    remaining_trials = max(
        0,
        config["n_trials"] - len(study.trials),
    )

    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)

    print("\nOptimization completed.")
    print(f"Best validation IoU: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    return study
