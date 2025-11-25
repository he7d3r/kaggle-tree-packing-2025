from decimal import Decimal
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.ops import unary_union

from christmas_tree import SCALE_FACTOR, ChristmasTree


def plot_results(
    side_length: Decimal,
    placed_trees: list[ChristmasTree],
    num_trees: int,
    output_dir: str | Path | None = "images",
    filename: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot the arrangement of trees and the bounding square.
    Optionally save the image to a directory.

    Parameters
    ----------
    side_length : Decimal
        Side length of the bounding square.
    placed_trees : list[ChristmasTree]
        Trees that were placed.
    num_trees : int
        Number of trees.
    output_dir : str | Path | None
        Directory where the plot should be saved. If None, saving is skipped.
    filename : str | None
        Optional filename. If not provided, a default is generated.
    show : bool
        Whether to display the image on screen.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])  # type: ignore

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        # Rescale for plotting
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / SCALE_FACTOR for val in x_scaled]
        y = [Decimal(val) / SCALE_FACTOR for val in y_scaled]
        ax.plot(x, y, color=colors[i])
        ax.fill(x, y, alpha=0.5, color=colors[i])

    minx = Decimal(bounds[0]) / SCALE_FACTOR
    miny = Decimal(bounds[1]) / SCALE_FACTOR
    maxx = Decimal(bounds[2]) / SCALE_FACTOR
    maxy = Decimal(bounds[3]) / SCALE_FACTOR

    width = maxx - minx
    height = maxy - miny

    square_x = minx if width >= height else minx - (side_length - width) / 2
    square_y = miny if height >= width else miny - (side_length - height) / 2

    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(bounding_square)

    padding = Decimal("0.5")
    ax.set_xlim(
        float(square_x - padding), float(square_x + side_length + padding)
    )
    ax.set_ylim(
        float(square_y - padding), float(square_y + side_length + padding)
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.title(f"{num_trees} Trees: {side_length:.12f}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{num_trees:03d}_trees.png"

        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
