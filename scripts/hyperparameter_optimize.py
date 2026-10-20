
"""Run Optuna hyperparameter optimization."""

import argparse
from pathlib import Path
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.configuration import PROJECT_ROOT
from src.hyperparameter_optimization import run_optimization


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/optuna/manet.yaml",
    )
    args = parser.parse_args(argv)

    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    run_optimization(config)


if __name__ == "__main__":
    main()
