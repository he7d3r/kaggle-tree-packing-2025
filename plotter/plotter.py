from decimal import Decimal

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from shapely.ops import unary_union

from christmas_tree.christmas_tree import SCALE_FACTOR, ChristmasTree


def plot_results(
    side_length: Decimal, placed_trees: list[ChristmasTree], num_trees: int
) -> None:
    """Plots the arrangement of trees and the bounding square."""
    _, ax = plt.subplots(figsize=(6, 6))
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

    padding = 0.5
    ax.set_xlim(
        float(square_x - Decimal(str(padding))),
        float(square_x + side_length + Decimal(str(padding))),
    )
    ax.set_ylim(
        float(square_y - Decimal(str(padding))),
        float(square_y + side_length + Decimal(str(padding))),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.title(f"{num_trees} Trees: {side_length:.12f}")
    plt.show()
    plt.close()
