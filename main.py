import argparse
import logging
import sys

import pandas as pd

from metric import DataFrameScorer, SolutionScorer
from plotter import Plotter
from solver import Solver, get_default_solver

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

    parser.add_argument(
        "--plot-every",
        type=int,
        default=10,
        help=(
            "Plot every N trees (e.g. 10 = plot each 10th tree). "
            "Use 0 to disable plotting entirely. Default: 10."
        ),
    )

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
            mlflow=False, draft=False, plot_every=10, max=DEFAULT_MAX_TREE_COUNT
        )


def start_mlflow(solver: Solver):
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(run_name=solver.name)


def in_notebook():
    """Check if we're running in a notebook environment."""
    try:
        from IPython.core.getipython import get_ipython

        return get_ipython() is not None
    except (ImportError, NameError):
        return False


def display_notebook_images():
    """Display saved plot images in notebook environment."""
    try:
        import glob

        from IPython.display import Image, display

        pattern = "images/*.png"

        for image_file in sorted(glob.glob(pattern)):
            print(f"Displaying: {image_file}")
            display(Image(filename=image_file, width=400))

    except ImportError:
        logger.info("IPython not available for image display")
    except Exception as e:
        logger.warning(f"Could not display images: {e}")


def main() -> None:
    args = parse_args()

    plotter = Plotter()
    solver = get_default_solver()

    run = start_mlflow(solver) if args.mlflow else None

    try:
        solution = solver.solve(problem_sizes=range(1, args.max + 1))

        if args.plot_every > 0:
            plotter.plot(
                solution,
                filter_fn=lambda n_tree: (n_tree.tree_count % args.plot_every)
                == 0,
            )
        else:
            logger.info("Plotting disabled (--plot-every=0)")

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

        # Display images in notebook
        if in_notebook() and args.plot_every > 0:
            display_notebook_images()

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
