from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal

import pandas as pd
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import NTree, detect_overlap
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

    def __init__(self, parallel: bool = True) -> None:
        self.parallel = parallel

    def score(self) -> float:
        self.preprocess()
        n_tree_list = self.n_trees()

        if not self.parallel:
            total_score = sum(
                self.score_n_tree(n_tree)
                for n_tree in tqdm(n_tree_list, desc="Scoring (seq)")
            )
            return float(total_score)

        with ProcessPoolExecutor() as executor:
            total_score = sum(
                tqdm(
                    executor.map(self.score_n_tree, n_tree_list),
                    total=len(n_tree_list),
                    desc="Scoring (parallel)",
                )
            )

        return float(total_score)

    def preprocess(self):
        """Optional hook for subclasses."""
        return

    def n_trees(self) -> list[NTree]:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def score_n_tree(n_tree: NTree) -> Decimal:
        BaseScorer.check_collisions(n_tree)
        return n_tree.score

    @staticmethod
    def check_collisions(n_tree: NTree) -> None:
        # check for collisions using neighborhood search
        polygons = n_tree.polygons
        r_tree = STRtree(polygons)
        # Checking for collisions
        for i, poly in enumerate(polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if detect_overlap(poly, polygons[index]):
                    raise ParticipantVisibleError(
                        f"Overlapping trees in n-tree {n_tree.name}"
                    )


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

    def n_trees(self) -> tuple[NTree, ...]:
        return tuple(
            NTree.from_dataframe(n_tree_df)
            for _, n_tree_df in Solution.n_tree_dfs(self.submission_df)
        )

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
