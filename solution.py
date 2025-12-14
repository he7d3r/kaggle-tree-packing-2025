from dataclasses import dataclass

import pandas as pd

from christmas_tree import NTree, ParticipantVisibleError


@dataclass(frozen=True)
class Solution:
    """Immutable solution consisting of multiple NTree instances."""

    n_trees: tuple[NTree, ...] = ()

    def score(self) -> float:
        return float(sum(n_tree.score for n_tree in self.n_trees))

    def to_dataframe(self) -> pd.DataFrame:
        """Creates a submission DataFrame from the Solution's n_trees data."""
        data = [
            {
                # Build the index of the submission, in the format:
                #  <trees_in_problem>_<tree_index>
                "id": f"{n_tree.tree_count:03d}_{t}",
                "x": tree.center_x,
                "y": tree.center_y,
                "deg": tree.angle,
            }
            for n_tree in self.n_trees
            for t, tree in enumerate(n_tree.trees)
        ]
        df = pd.DataFrame(data).set_index("id").astype(float).round(decimals=6)

        # To ensure everything is kept as a string, prepend an 's'
        for col in df.columns:
            df[col] = "s" + df[col].astype("string")
        return df

    @staticmethod
    def from_csv(submission_file: str = "submission.csv") -> "Solution":
        submission_df = pd.read_csv(
            submission_file,
            dtype={"x": "string", "y": "string", "deg": "string"},
            index_col="id",
        )
        return Solution.from_dataframe(submission_df)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "Solution":
        """Populates the Solution object from a DataFrame."""
        Solution.validate_dataframe(df)
        df = df.apply(lambda col: col.str.slice(1))
        n_trees = Solution.n_trees_from_dataframe(df)
        return Solution(n_trees=n_trees)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> None:
        if df.apply(lambda col: col.str.startswith("s")).all().all():
            return
        raise ParticipantVisibleError(
            "Value(s) in columns x, y, deg found without `s` prefix."
        )

    @staticmethod
    def n_trees_from_dataframe(df: pd.DataFrame):
        return tuple(
            NTree.from_dataframe(n_tree_df)
            for _, n_tree_df in Solution.n_tree_dfs(df)
        )

    @staticmethod
    def n_tree_dfs(df: pd.DataFrame):
        """Extracts n-tree data-frames from the DataFrame."""
        return df.groupby(df.index.astype(str).str.split("_").str[0])
