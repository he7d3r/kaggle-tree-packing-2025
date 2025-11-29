import argparse
import logging
import random
import sys

import mlflow
import pandas as pd

from metric import DataFrameScorer, SolutionScorer
from plotter import Plotter
from solver import BaselineIncrementalSolver

OUTPUT_FILE = "submission.csv"
TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Christmas Tree Packing"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATEFMT))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow", action="store_true", help="Enable MLflow logging"
    )
    parser.add_argument(
        "--draft", action="store_true", help="Skip output file creation"
    )
    return parser.parse_args()


def start_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run()


def main() -> None:
    args = parse_args()

    plotter = Plotter()
    solver = BaselineIncrementalSolver(rng=random.Random(42))

    run = start_mlflow() if args.mlflow else None

    try:
        solution = solver.solve_all(plotter)

        if args.draft:
            logger.info("Skipped submission file creation (draft mode).")
            score = SolutionScorer().score(solution)
        else:
            solution.to_dataframe().to_csv(OUTPUT_FILE)
            logger.info("Submission saved to %s.", OUTPUT_FILE)
            submission_df = pd.read_csv(
                OUTPUT_FILE,
                dtype={"x": "string", "y": "string", "deg": "string"},
                index_col="id",
            )
            logger.info("Submission reloaded from %s.", OUTPUT_FILE)
            score = DataFrameScorer().score(submission_df)

        if args.mlflow:
            mlflow.log_metric("submission_score", score)
        else:
            logger.info("Skipped MLflow logging.")

        logger.info("Submission score: %s.", score)

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
