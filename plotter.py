from concurrent.futures import ProcessPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree, from_scale
from solution import Solution


def _plot_single_n_tree_helper(args):
    """Helper function for parallel plotting of a single n_tree."""
    plotter, n_tree, output_path = args
    return plotter._plot_single_n_tree(n_tree, output_path)


class Plotter:
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
            Directory where the plot should be saved. Default is 'images'.
        filename_format : str
            Format string for filenames.
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
        Supports both sequential and parallel execution.

        Parameters
        ----------
        solution : Solution
            The solution containing n_trees.
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

        if not self.parallel:
            # Sequential version - useful for debugging/profiling
            for n_tree in tqdm(n_trees_to_plot, desc="Plotting n-trees (seq)"):
                output_path = self.output_dir / self.filename_format.format(
                    n_tree.tree_count
                )
                self._plot_single_n_tree(n_tree, output_path)
            return

        # Parallel version
        self._plot_parallel(n_trees_to_plot)

    def _plot_parallel(self, n_trees: list[NTree]) -> None:
        """Parallel plotting implementation."""
        # Prepare arguments for each task
        tasks = [
            (
                self,
                n_tree,
                self.output_dir
                / self.filename_format.format(n_tree.tree_count),
            )
            for n_tree in n_trees
        ]

        # Use ProcessPoolExecutor for CPU-bound tasks (matplotlib rendering)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(_plot_single_n_tree_helper, task): n_tree
                for task, n_tree in zip(tasks, n_trees)
            }

            # Process results as they complete with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Plotting n-trees (parallel)",
            ):
                n_tree = futures[future]
                try:
                    future.result()  # Raises exception if any
                except Exception as e:
                    print(
                        f"Error plotting n-tree with {n_tree.tree_count} "
                        f"trees: {e}"
                    )
                    raise

    def _plot_single_n_tree(self, n_tree: NTree, output_path: Path) -> None:
        """
        Plot a single n_tree and save to output_path.
        This method is designed to be pickleable for parallel execution.

        Note: We need to create a new figure for each call since matplotlib
        figures are not thread/process safe.
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
            (float(minx), float(miny)),
            float(side_length),
            float(side_length),
            fill=False,
            edgecolor="red",
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(bounding_square)

        padding = Decimal("0.5")
        ax.set_xlim(float(minx - padding), float(minx + side_length + padding))
        ax.set_ylim(float(miny - padding), float(miny + side_length + padding))
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        plt.title(f"{num_trees} Trees: {side_length:.12f}")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def _plot_tree(
        self, tree: ChristmasTree, ax: Axes, color: NDArray[np.float64]
    ) -> None:
        # Rescale for plotting
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [from_scale(val) for val in x_scaled]
        y = [from_scale(val) for val in y_scaled]
        ax.plot(x, y, color=color)
        ax.fill(x, y, alpha=0.5, color=color)
