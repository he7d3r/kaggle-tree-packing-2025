from decimal import Decimal
from typing import Generator, Iterable

import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import SCALE_FACTOR, NTree
from solution import Solution


class ParticipantVisibleError(Exception):
    pass


class BaseScorer:
    """
    Santa 2025 Metric
    For each N-tree configuration, calculate the bounding square divided by N.
    Final score is the sum of the scores across all configurations.

    A scaling factor is used to maintain reasonably precise floating point
    calculations in the shapely (v 2.1.2) library.
    """

    def score(self) -> float:
        self.preprocess()

        total_score = Decimal("0")
        for n_tree in tqdm(self.generate_n_trees(), desc="Scoring"):
            total_score += self._score_n_tree(n_tree)

        return float(total_score)

    def preprocess(self):
        """Optional hook for subclasses."""
        return

    def generate_n_trees(self) -> Iterable[NTree]:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    def _score_n_tree(self, n_tree: NTree) -> Decimal:
        # Create tree objects from the solution values and
        # check for collisions using neighborhood search
        polygons = n_tree.polygons
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
                        f"Overlapping trees in n-tree {n_tree.name}"
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

    def generate_n_trees(self) -> Generator[NTree, None, None]:
        for _, n_tree_df in Solution.n_tree_dfs(self.submission_df):
            yield NTree.from_dataframe(n_tree_df)

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

    def generate_n_trees(self) -> Generator[NTree, None, None]:
        for n_tree in self.solution.n_trees:
            yield n_tree
