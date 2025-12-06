from dataclasses import dataclass

import pandas as pd

from christmas_tree import NTree


@dataclass(frozen=True)
class Solution:
    """Immutable solution consisting of multiple NTree instances."""

    n_trees: tuple[NTree, ...] = ()

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
    def from_dataframe(df: pd.DataFrame) -> "Solution":
        """Populates the Solution object from a DataFrame."""
        df = df.apply(lambda col: col.str.lstrip("s"))
        n_trees = tuple(
            NTree.from_dataframe(n_tree_df)
            for _, n_tree_df in Solution.n_tree_dfs(df)
        )
        return Solution(n_trees=n_trees)

    @staticmethod
    def n_tree_dfs(df: pd.DataFrame):
        """Extracts n-tree data-frames from the DataFrame."""
        return df.groupby(df.index.astype(str).str.split("_").str[0])
