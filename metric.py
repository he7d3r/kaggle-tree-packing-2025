from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal

import pandas as pd
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import NTree, from_scale
from solution import Solution


class ParticipantVisibleError(Exception):
    pass


def _score_n_tree_helper(args):
    scorer, n_tree = args
    return scorer._score_n_tree(n_tree)


class BaseScorer:
    """
    Santa 2025 Metric
    For each N-tree configuration, calculate the bounding square divided by N.
    Final score is the sum of the scores across all configurations.

    A scaling factor is used to maintain reasonably precise floating point
    calculations in the shapely (v 2.1.2) library.
    """

    def __init__(self, parallel: bool = True) -> None:
        self.parallel = parallel

    def score(self) -> float:
        self.preprocess()

        n_tree_list = self.n_trees()
        total_score = Decimal("0")

        if not self.parallel:
            # Sequential – profiling-friendly
            for n_tree in tqdm(n_tree_list, desc="Scoring (seq)"):
                total_score += self._score_n_tree(n_tree)
            return float(total_score)

        # Parallel version
        with ProcessPoolExecutor() as executor:
            results = tqdm(
                executor.map(
                    _score_n_tree_helper,
                    ((self, n_tree) for n_tree in n_tree_list),
                ),
                total=len(n_tree_list),
                desc="Scoring (parallel)",
            )
            for val in results:
                total_score += val

        return float(total_score)

    def preprocess(self):
        """Optional hook for subclasses."""
        return

    def n_trees(self) -> list[NTree]:
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

        return (from_scale(side_length_scaled) ** 2) / Decimal(len(polygons))


class SolutionScorer(BaseScorer):
    def __init__(self, solution: Solution, parallel: bool = True):
        super().__init__(parallel=parallel)
        self.solution = solution

    def n_trees(self) -> tuple[NTree, ...]:
        return self.solution.n_trees


class DataFrameScorer(BaseScorer):
    def __init__(self, submission_df: pd.DataFrame, parallel: bool = True):
        super().__init__(parallel=parallel)
        self.submission_df = submission_df

    def preprocess(self):
        df = self._remove_leading_s_prefix(self.submission_df)
        self._validate_limits(df)
        self.submission_df = df

    def n_trees(self) -> list[NTree]:
        return [
            NTree.from_dataframe(n_tree_df)
            for _, n_tree_df in Solution.n_tree_dfs(self.submission_df)
        ]

    def _remove_leading_s_prefix(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.astype(str)
        if not (df.apply(lambda col: col.str.startswith("s")).all().all()):
            raise ParticipantVisibleError(
                "Value(s) in columns x, y, deg found without `s` prefix."
            )
        df = df.apply(lambda col: col.str.slice(1))
        return df

    def _validate_limits(self, df: pd.DataFrame) -> None:
        limit = 100
        if (df[["x", "y"]].astype(float).abs() > limit).any().any():
            raise ParticipantVisibleError(
                f"x and/or y values outside the bounds of -{limit} to {limit}."
            )
