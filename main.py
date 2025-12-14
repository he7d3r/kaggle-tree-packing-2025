import glob
import os
import subprocess
import sys
from datetime import datetime

# Force matplotlib to use a non-GUI backend BEFORE any imports
os.environ["MPLBACKEND"] = "Agg"  # Non-interactive backend
os.environ["MATPLOTLIB_BACKEND"] = "Agg"

import argparse
import logging

import pandas as pd

from metric import DataFrameScorer, SolutionScorer
from plotter import Plotter
from solver import get_default_solver

DEFAULT_MAX_TREE_COUNT = 200
OUTPUT_FILE = "submission.csv"
TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Christmas Tree Packing"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_git_run_name() -> str:
    """
    Get a run name based on git commit information.
    Falls back to timestamp if git is not available.
    """
    try:
        # Get the latest commit message (first line)
        commit_message = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()

        # Clean up the commit message for use as a run name
        # Remove special characters that might cause issues
        clean_message = " ".join(commit_message.splitlines())
        clean_message = clean_message[:50]  # Truncate if too long

        return clean_message

    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git is not available or not in a git repo
        # Fall back to timestamp-based name
        timestamp = datetime.now().isoformat(timespec="seconds")
        return f"local-run-{timestamp}"


DEFAULT_RUN_NAME = get_git_run_name()


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
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=(
            f"MLflow run name (default: {DEFAULT_RUN_NAME[:50]}..."
            if len(DEFAULT_RUN_NAME) > 50
            else f"MLflow run name (default: {DEFAULT_RUN_NAME})"
        ),
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
        help=(
            "Maximum number of trees to solve "
            f"(default: {DEFAULT_MAX_TREE_COUNT})"
        ),
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multiprocessing (solver + score). Useful for profiling.",
    )
    parser.add_argument(
        "--analyze",
        nargs="?",
        const="submission.csv",  # Default when --analyze is used without argument
        help=(
            "Run score analysis and create plots from CSV file(s). "
            "Accepts a single file path or a glob pattern (e.g., '*.csv'). "
            "Default: 'submission.csv' when flag is used without argument."
        ),
    )

    try:
        return parser.parse_known_args()[0]
    except SystemExit:
        return argparse.Namespace(
            mlflow=False,
            run_name=DEFAULT_RUN_NAME,
            draft=False,
            plot_every=10,
            max=DEFAULT_MAX_TREE_COUNT,
            no_parallel=False,
            analyze=False,
        )


def start_mlflow(run_name: str):
    """Start an MLflow run with the given name."""
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow.start_run(run_name=run_name)


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

    # If analyze flag is set, run analysis and exit
    if args.analyze:
        logger.info(f"Starting score analysis with pattern: {args.analyze}")
        plotter = Plotter(parallel=not args.no_parallel)

        # Check if pattern contains wildcards
        if "*" in args.analyze or "?" in args.analyze or "[" in args.analyze:
            # Use glob to find matching files
            csv_files = sorted(glob.glob(args.analyze))
            if not csv_files:
                logger.warning(
                    f"No files found matching pattern: {args.analyze}"
                )
                return
            logger.info(f"Found {len(csv_files)} file(s): {csv_files}")
            plotter.plot_scores_analysis_multiple(csv_files)
        else:
            # Single file case
            plotter.plot_scores_analysis(args.analyze)

        # Display analysis plot in notebook if applicable
        if in_notebook():
            try:
                from IPython.display import Image, display

                analysis_path = "images/score_analysis.png"
                if os.path.exists(analysis_path):
                    display(Image(filename=analysis_path, width=800))
            except Exception as e:
                logger.warning(f"Could not display analysis plot: {e}")

        return

    # Original main logic
    parallel = not args.no_parallel
    plotter = Plotter(parallel=parallel)
    solver = get_default_solver(parallel=parallel)

    run = start_mlflow(args.run_name) if args.mlflow else None

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
            score = SolutionScorer(solution, parallel=parallel).score()
        else:
            solution.to_dataframe().to_csv(OUTPUT_FILE)
            logger.info("Submission saved to %s.", OUTPUT_FILE)

            submission_df = pd.read_csv(
                OUTPUT_FILE,
                dtype={"x": "string", "y": "string", "deg": "string"},
                index_col="id",
            )
            logger.info("Submission reloaded from %s.", OUTPUT_FILE)
            score = DataFrameScorer(submission_df, parallel=parallel).score()

        if args.mlflow:
            import mlflow

            mlflow.log_param("run_name", args.run_name)
            mlflow.log_metric("submission_score", score)
            logger.info("MLflow run name: %s", args.run_name)
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
            mlflow.log_param("run_name", args.run_name)
            mlflow.end_run(status="FAILED")
        raise
    finally:
        if args.mlflow and run is not None:
            import mlflow

            mlflow.end_run()


if __name__ == "__main__":
    main()
