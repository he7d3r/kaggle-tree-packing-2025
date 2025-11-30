import argparse
import logging
import sys

import pandas as pd

from metric import DataFrameScorer, SolutionScorer
from plotter import Plotter
from solver import BaseSolver, GridLayoutSolver, IncrementalSolver

DEFAULT_MAX_TREE_COUNT = 200
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
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    parser.add_argument(
        "--max",
        type=int,
        default=DEFAULT_MAX_TREE_COUNT,
        help=f"Maximum number of trees to solve (default: {DEFAULT_MAX_TREE_COUNT})",
    )

    try:
        return parser.parse_known_args()[0]
    except SystemExit:
        return argparse.Namespace(
            mlflow=False, draft=False, no_plot=False, max=DEFAULT_MAX_TREE_COUNT
        )


def start_mlflow(solver: BaseSolver | IncrementalSolver):
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(run_name=solver.__class__.__name__)


def main() -> None:
    args = parse_args()

    plotter = Plotter()
    solver = GridLayoutSolver()

    run = start_mlflow(solver) if args.mlflow else None

    try:
        solution = solver.solve(problem_sizes=range(1, args.max + 1))

        if not args.no_plot:
            plotter.plot(
                solution,
                filter_fn=lambda n_tree: n_tree.tree_count % 10 == 0,
            )

        if args.draft:
            logger.info("Skipped submission file creation (draft mode).")
            score = SolutionScorer(solution).score()
        else:
            solution.to_dataframe().to_csv(OUTPUT_FILE)
            logger.info("Submission saved to %s.", OUTPUT_FILE)

            submission_df = pd.read_csv(
                OUTPUT_FILE,
                dtype={"x": "string", "y": "string", "deg": "string"},
                index_col="id",
            )
            logger.info("Submission reloaded from %s.", OUTPUT_FILE)
            score = DataFrameScorer(submission_df).score()

        if args.mlflow:
            import mlflow

            mlflow.log_metric("submission_score", score)
        else:
            logger.info("Skipped MLflow logging.")

        logger.info("Submission score: %s.", score)

    except Exception as e:
        if args.mlflow:
            import mlflow

            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if args.mlflow and run is not None:
            import mlflow

            mlflow.end_run()


if __name__ == "__main__":
    main()
