from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from tqdm import tqdm

from solution import Solution
from trees import ChristmasTree, NTree

T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")


class _SizedContainer(Protocol[T_co]):
    """Protocol for sized iterable containers."""

    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


def _for_each(
    fn: Callable[..., None],
    items: _SizedContainer[tuple],
    *,
    parallel: bool,
    desc: str,
    max_workers: int | None = None,
) -> None:
    """Execute function on items with optional parallel processing."""
    if not parallel:
        for item in tqdm(items, desc=desc):
            fn(*item)
        return

    # Use ProcessPoolExecutor for CPU-bound tasks (matplotlib rendering)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fn, *item) for item in items]
        # Process results as they complete with progress bar
        for future in tqdm(as_completed(futures), total=len(items), desc=desc):
            future.result()


def _plot_single_n_tree_task(
    plotter: "Plotter",
    n_tree: NTree,
    output_path: Path,
) -> None:
    """Task wrapper for parallel plotting."""
    plotter._plot_single_n_tree(n_tree, output_path)


class Plotter:
    """
    Plotter for visualizing tree placement solutions.

    Supports both sequential and parallel plotting of multiple configurations.
    """

    def __init__(
        self,
        output_dir: Path = Path("images"),
        filename_format: str = "{:03d}_trees.png",
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> None:
        """
        Initialize the Plotter with optional parallel processing.

        Parameters
        ----------
        output_dir : Path
            Directory where plots should be saved. Default is 'images'.
        filename_format : str
            Format string for filenames using tree count.
        parallel : bool
            Whether to use parallel processing. Default is True.
        max_workers : Optional[int]
            Maximum number of worker processes. If None, uses os.cpu_count().
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename_format = filename_format
        self.parallel = parallel
        self.max_workers = max_workers

    def plot(
        self, solution: Solution, *, filter_fn: Optional[Callable] = None
    ) -> None:
        """
        Plot all n_tree objects that satisfy filter_fn.

        Parameters
        ----------
        solution : Solution
            The solution containing n_trees to plot.
        filter_fn : callable, optional
            A function taking a single n_tree and returning True/False.
            If None, defaults to plotting all n-trees.
        """
        # Filter n_trees if needed
        n_trees_to_plot = [
            n_tree
            for n_tree in solution.n_trees
            if filter_fn is None or filter_fn(n_tree)
        ]

        def make_task(n_tree: NTree):
            output_path = self.output_dir / self.filename_format.format(
                n_tree.tree_count
            )
            return (self, n_tree, output_path)

        tasks = [make_task(n_tree) for n_tree in n_trees_to_plot]

        _for_each(
            partial(_plot_single_n_tree_task),
            tasks,
            parallel=self.parallel,
            desc="Plotting n-trees",
            max_workers=self.max_workers,
        )

    def plot_scores_analysis(
        self, submission_file: str = "submission.csv"
    ) -> None:
        """
        Create a line plot showing scores for each n-tree configuration.

        Parameters
        ----------
        submission_file : str
            Path to the submission CSV file to analyze.
        """
        solution = Solution.from_csv(submission_file)

        # Compute scores
        n_trees = []
        scores = []
        for n_tree_obj in solution.n_trees:
            n_trees.append(n_tree_obj.tree_count)
            try:
                scores.append(n_tree_obj.score)
            except Exception as e:
                print(f"Error computing score: {e}")
                scores.append(np.nan)

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            n_trees,
            scores,
            "b-",
            linewidth=1,
            marker="o",
            markersize=3,
            label=f"Score per n-tree - {Path(submission_file).name}",
        )

        # Configure
        ax.set_xlabel("Number of Trees (n)")
        ax.set_xlim(left=0, right=201)
        ax.set_ylabel("Score per n-tree")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Score Analysis - {Path(submission_file).name}")
        ax.legend()
        plt.tight_layout()

        # Save the plot
        output_path = self.output_dir / "score_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_scores_analysis_multiple(
        self, submission_files: list[str]
    ) -> None:
        """
        Create a multi-line plot showing scores for multiple submission files.

        Parameters
        ----------
        submission_files : list[str]
            List of paths to submission CSV files to analyze.
        """
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Define a colormap for different lines
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(submission_files), 10)))  # type: ignore

        for idx, submission_file in enumerate(submission_files):
            try:
                solution = Solution.from_csv(submission_file)

                # Compute scores
                n_trees = []
                scores = []
                for n_tree_obj in solution.n_trees:
                    n_trees.append(n_tree_obj.tree_count)
                    try:
                        scores.append(n_tree_obj.score)
                    except Exception as e:
                        print(f"Error computing score: {e}")
                        scores.append(np.nan)

                # Plot the line for this file
                color = (
                    colors[idx % len(colors)]
                    if len(submission_files) <= 10
                    else None
                )
                label = Path(submission_file).name
                ax.plot(
                    n_trees,
                    scores,
                    "-",
                    linewidth=1,
                    marker="o",
                    markersize=3,
                    label=label,
                    color=color,
                )

            except Exception as e:
                print(f"Error processing {submission_file}: {e}")
                continue

        # Configure
        ax.set_xlabel("Number of Trees (n)")
        ax.set_xlim(left=0, right=201)
        ax.set_ylabel("Score per n-tree")
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)

        # Set title based on number of files
        if len(submission_files) == 1:
            ax.set_title(f"Score Analysis - {Path(submission_files[0]).name}")
        else:
            ax.set_title(f"Score Analysis - {len(submission_files)} files")

        ax.legend(loc="best", fontsize="small")
        plt.tight_layout()

        # Save the plot
        output_path = self.output_dir / "score_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _plot_single_n_tree(self, n_tree: NTree, output_path: Path) -> None:
        """
        Plot a single n_tree and save to output_path.

        This method is designed to be pickleable for parallel execution.
        Creates a new figure for each call since matplotlib figures
        are not thread/process safe.
        """
        num_trees = n_tree.tree_count
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.viridis(  # type: ignore
            [i / num_trees for i in range(num_trees)]
        )

        for i, tree in enumerate(n_tree.trees):
            self._plot_tree(tree, ax, color=colors[i])

        side_length = n_tree.side_length
        minx, miny, _, _ = n_tree.bounds

        bounding_square = Rectangle(
            (minx, miny),
            side_length,
            side_length,
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(bounding_square)

        padding = 0.5
        ax.set_xlim(minx - padding, minx + side_length + padding)
        ax.set_ylim(miny - padding, miny + side_length + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        plt.title(f"{num_trees} Trees: {side_length:.12f}")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_tree(
        self, tree: ChristmasTree, ax: Axes, color: NDArray[np.float64]
    ) -> None:
        """Plot a single tree on the given axes."""
        x, y = tree.polygon.exterior.xy
        ax.plot(x, y, color=color)
        ax.fill(x, y, alpha=0.5, color=color)

    def plot_summary(self) -> None:
        df = pd.read_csv("summary.csv")
        df.plot(
            kind="line",
            x="n",
            figsize=(12, 12),
            marker=".",
            subplots=True,
            grid=True,
            title="Summary Statistics",
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_plots.png", dpi=150)
