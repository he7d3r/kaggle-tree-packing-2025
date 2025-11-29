# %%
"""
Santa 2025 Metric
For each N-tree configuration, calculate the bounding square divided by N.
Final score is the sum of the scores across all configurations.

A scaling factor is used to maintain reasonably precise floating point
calculations in the shapely (v 2.1.2) library.
"""

from decimal import Decimal

import pandas as pd
from shapely import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import SCALE_FACTOR, TreePacking
from submission import Submission


class ParticipantVisibleError(Exception):
    pass


class Scorer:
    def score_df(self, submission: pd.DataFrame) -> float:
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
        submission = self._remove_leading_s_prefix(submission)
        self._validate_limits(submission)

        # grouping puzzles to score
        grouped = Submission.groups(submission)
        total_score = Decimal("0.0")
        for group, df_group in tqdm(list(grouped), desc="Scoring groups"):
            name = str(group)
            polygons = TreePacking.from_dataframe(df_group).polygons
            total_score += self._score_group(name, polygons)

        return float(total_score)

    def score_submission(self, submission: Submission) -> float:
        """Scores a Submission object."""
        total_score = Decimal("0.0")
        for pack in tqdm(submission.packs, desc="Scoring packs"):
            name = f"{pack.tree_count:03d}"
            polygons = pack.polygons
            total_score += self._score_group(name, polygons)
        return float(total_score)

    def _score_group(self, name: str, polygons: list[Polygon]) -> Decimal:
        # Create tree objects from the submission values and
        # check for collisions using neighborhood search
        r_tree = STRtree(polygons)

        # Checking for collisions
        for i, poly in enumerate(polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if poly.intersects(polygons[index]) and not poly.touches(
                    polygons[index]
                ):
                    raise ParticipantVisibleError(
                        f"Overlapping trees in group {name}"
                    )

        # Calculate score for the group
        bounds = unary_union(polygons).bounds
        # Use the largest edge of the bounding rectangle to make a square bounding box
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        num_trees = len(polygons)
        group_score = (
            (Decimal(side_length_scaled) ** 2)
            / (SCALE_FACTOR**2)
            / Decimal(num_trees)
        )
        return group_score

    def _remove_leading_s_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        data_cols = ["x", "y", "deg"]
        df = df.astype(str)
        for c in data_cols:
            if not df[c].str.startswith("s").all():
                raise ParticipantVisibleError(
                    f"Value(s) in column {c} found without `s` prefix."
                )
            df[c] = df[c].str[1:]
        return df

    def _validate_limits(self, df) -> None:
        limit = 100
        if (df[["x", "y"]].astype(float).abs() > limit).any().any():
            raise ParticipantVisibleError(
                f"x and/or y values outside the bounds of -{limit} to {limit}."
            )
