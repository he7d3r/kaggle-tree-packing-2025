"""
Santa 2025 Metric
For each N-tree configuration, calculate the bounding square divided by N.
Final score is the sum of the scores across all configurations.

A scaling factor is used to maintain reasonably precise floating point
calculations in the shapely (v 2.1.2) library.
"""

from decimal import Decimal
from typing import Generator, Iterable

import pandas as pd
from shapely import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import SCALE_FACTOR, NTree
from solution import Solution


class ParticipantVisibleError(Exception):
    pass


class BaseScorer:
    """Template method pattern: subclasses override only small pieces."""

    def preprocess(self):
        """Optional hook for subclasses."""
        return

    def generate_groups(self) -> Iterable[tuple[str, list[Polygon]]]:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def score(self) -> float:
        self.preprocess()

        total_score = Decimal("0")
        for name, polygons in tqdm(self.generate_groups(), desc="Scoring"):
            total_score += self._score_group(name, polygons)

        return float(total_score)

    def _score_group(self, name: str, polygons: list[Polygon]) -> Decimal:
        # Create tree objects from the solution values and
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

        # bounding square score
        bounds = unary_union(polygons).bounds
        side_length_scaled = max(bounds[2] - bounds[0], bounds[3] - bounds[1])

        return (
            (Decimal(side_length_scaled) ** 2)
            / (SCALE_FACTOR**2)
            / Decimal(len(polygons))
        )


class DataFrameScorer(BaseScorer):
    def __init__(self, submission_df: pd.DataFrame):
        self.submission_df = submission_df

    def preprocess(self):
        df = self._remove_leading_s_prefix(self.submission_df)
        self._validate_limits(df)
        self.submission_df = df

    def generate_groups(
        self,
    ) -> Generator[tuple[str, list[Polygon]], None, None]:
        grouped = Solution.groups(self.submission_df)

        for group, df_group in grouped:
            name = str(group)
            n_tree = NTree.from_dataframe(df_group)
            yield name, n_tree.polygons

    def _remove_leading_s_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.astype(str)
        for c in ["x", "y", "deg"]:
            if not df[c].str.startswith("s").all():
                raise ParticipantVisibleError(
                    f"Value(s) in column {c} found without `s` prefix."
                )
            df[c] = df[c].str[1:]
        return df

    def _validate_limits(self, df: pd.DataFrame) -> None:
        limit = 100
        if (df[["x", "y"]].astype(float).abs() > limit).any().any():
            raise ParticipantVisibleError(
                f"x and/or y values outside the bounds of -{limit} to {limit}."
            )


class SolutionScorer(BaseScorer):
    def __init__(self, solution: Solution):
        self.solution = solution

    def generate_groups(
        self,
    ) -> Generator[tuple[str, list[Polygon]], None, None]:
        for n_tree in self.solution.n_trees:
            name = f"{n_tree.tree_count:03d}"
            yield name, n_tree.polygons
