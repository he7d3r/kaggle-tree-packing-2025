import argparse
import logging
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


# Configure logging early in the module
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow", action="store_true", help="Enable MLflow logging"
    )
    parser.add_argument(
        "--draft", action="store_true", help="Skip output file creation"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def start_mlflow():
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return mlflow.start_run()

    run = start_mlflow() if args.mlflow else None

    try:
        rng = random.Random(42)
        plotter = Plotter()
        tree_data = solve_all(rng, plotter)
        df = make_submission_df(tree_data)

        if not args.draft:
            df.to_csv(OUTPUT_FILE)
            logger.info("Submission saved to %s", OUTPUT_FILE)

            df = pd.read_csv(
                OUTPUT_FILE,
                dtype={"x": "string", "y": "string", "deg": "string"},
                index_col="id",
            )
            logger.info("Submission reloaded from %s", OUTPUT_FILE)
        else:
            logger.info("Skipped submission file creation (draft mode)")

        submission_score = score(df)

        if args.mlflow:
            mlflow.log_metric("submission_score", submission_score)
        else:
            logger.info("Skipped MLflow logging")

        logger.info("Submission score: %s", submission_score)

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
