from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from christmas_tree import SCALE_FACTOR, NTree


class Plotter:
    def __init__(
        self,
        output_dir: Path = Path("images"),
        filename_format: str = "{:03d}_trees.png",
    ) -> None:
        """
        Initialize the Plotter.

        Parameters
        ----------
        output_dir : Path
            Directory where the plot should be saved. Default is 'images'.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.filename_format = filename_format

    def plot(self, n_tree: NTree) -> None:
        """
        Plot the arrangement of trees and the bounding square.
        Optionally save the image to a directory.

        Parameters
        ----------
        n_tree : NTree
            The NTree object containing the trees to plot.
        """
        num_trees = n_tree.tree_count
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])  # type: ignore

        for i, tree in enumerate(n_tree.trees):
            # Rescale for plotting
            x_scaled, y_scaled = tree.polygon.exterior.xy
            x = [Decimal(val) / SCALE_FACTOR for val in x_scaled]
            y = [Decimal(val) / SCALE_FACTOR for val in y_scaled]
            ax.plot(x, y, color=colors[i])
            ax.fill(x, y, alpha=0.5, color=colors[i])

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

        filename = self.filename_format.format(num_trees)
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
