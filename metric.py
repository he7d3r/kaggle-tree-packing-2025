from decimal import Decimal

import pandas as pd

from christmas_tree import NTree
from solution import Solution


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
        return float(sum(n_tree.score for n_tree in self.n_trees()))

    def preprocess(self):
        """Optional hook for subclasses."""
        return

    def n_trees(self) -> list[NTree]:
        """Must be implemented by subclasses."""
        raise NotImplementedError

    @staticmethod
    def score_n_tree(n_tree: NTree) -> Decimal:
        return n_tree.score


class SolutionScorer(BaseScorer):
    def __init__(self, solution: Solution):
        self.solution = solution

    def n_trees(self) -> tuple[NTree, ...]:
        return self.solution.n_trees


class DataFrameScorer(BaseScorer):
    def __init__(self, submission_df: pd.DataFrame):
        self.submission_df = submission_df

    def n_trees(self) -> tuple[NTree, ...]:
        return Solution.n_trees_from_dataframe(self.submission_df)
