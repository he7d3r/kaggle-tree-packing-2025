# %%
"""
Santa 2025 Metric
For each N-tree configuration, calculate the bounding square divided by N.
Final score is the sum of the scores across all configurations.

A scaling factor is used to maintain reasonably precise floating point
calculations in the shapely (v 2.1.2) library.
"""

from decimal import Decimal, getcontext

import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree

from christmas_tree.christmas_tree import ChristmasTree

# Decimal precision and scaling factor
getcontext().prec = 25
scale_factor = Decimal("1e18")


class ParticipantVisibleError(Exception):
    pass


def score(submission: pd.DataFrame) -> float:
    """
    For each n-tree configuration, the metric calculates the bounding square
    volume divided by n, summed across all configurations.

    This metric uses shapely v2.1.2.

    Examples
    -------
    >>> import pandas as pd
    >>> row_id_column_name = 'id'
    >>> data = [['002_0', 's-0.2', 's-0.3', 's335'], ['002_1', 's0.49', 's0.21', 's155']]
    >>> submission = pd.DataFrame(columns=['id', 'x', 'y', 'deg'], data=data)
    >>> solution = submission[['id']].copy()
    >>> score(solution, submission, row_id_column_name)
    0.877038143325...
    """

    # remove the leading 's' from submissions
    data_cols = ["x", "y", "deg"]
    submission = submission.astype(str)
    for c in data_cols:
        if not submission[c].str.startswith("s").all():
            raise ParticipantVisibleError(
                f"Value(s) in column {c} found without `s` prefix."
            )
        submission[c] = submission[c].str[1:]

    # enforce value limits
    limit = 100
    bad_x = (submission["x"].astype(float) < -limit).any() or (
        submission["x"].astype(float) > limit
    ).any()
    bad_y = (submission["y"].astype(float) < -limit).any() or (
        submission["y"].astype(float) > limit
    ).any()
    if bad_x or bad_y:
        raise ParticipantVisibleError(
            "x and/or y values outside the bounds of -100 to 100."
        )

    # grouping puzzles to score
    submission["tree_count_group"] = submission["id"].str.split("_").str[0]

    total_score = Decimal("0.0")
    for group, df_group in submission.groupby("tree_count_group"):
        num_trees = len(df_group)

        # Create tree objects from the submission values
        placed_trees = []
        for _, row in df_group.iterrows():
            placed_trees.append(ChristmasTree(row["x"], row["y"], row["deg"]))

        # Check for collisions using neighborhood search
        all_polygons = [p.polygon for p in placed_trees]
        r_tree = STRtree(all_polygons)

        # Checking for collisions
        for i, poly in enumerate(all_polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(all_polygons[index]) and not poly.touches(
                    all_polygons[index]
                ):
                    raise ParticipantVisibleError(
                        f"Overlapping trees in group {group}"
                    )

        # Calculate score for the group
        bounds = unary_union(all_polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square bounding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        group_score = (
            (Decimal(side_length_scaled) ** 2)
            / (scale_factor**2)
            / Decimal(num_trees)
        )
        total_score += group_score

    return float(total_score)
