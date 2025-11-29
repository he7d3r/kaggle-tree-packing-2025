import argparse
import random

import mlflow
import pandas as pd

from metric import score
from plotter import Plotter
from solver import solve_all
from submission import make_submission_df

OUTPUT_FILE = "submission.csv"
TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Christmas Tree Packing"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow", action="store_true", help="Enable MLflow logging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mlflow:
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run = mlflow.start_run()
    else:
        run = None  # no MLflow

    try:
        rng = random.Random(42)
        plotter = Plotter()
        tree_data = solve_all(rng, plotter)
        df = make_submission_df(tree_data)
        df.to_csv(OUTPUT_FILE)
        print(f"Submission saved to {OUTPUT_FILE}")

        df = pd.read_csv(
            OUTPUT_FILE,
            dtype={"x": "string", "y": "string", "deg": "string"},
        )
        submission_score = score(df)
        print(f"Submission score: {submission_score}")

        if args.mlflow:
            mlflow.log_metric("submission_score", submission_score)

    except Exception as e:
        if args.mlflow:
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if args.mlflow and run is not None:
            mlflow.end_run()


if __name__ == "__main__":
    main()
