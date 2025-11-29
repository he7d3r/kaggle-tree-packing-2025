from typing import Any

import pandas as pd

from christmas_tree import NTree


class Solution:
    def __init__(self, n_trees: list[NTree] | None = None):
        self.n_trees = n_trees if n_trees else []

    def add(self, n_tree: NTree) -> None:
        self.n_trees.append(n_tree)

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
        n_trees = [
            NTree.from_dataframe(group_df)
            for _, group_df in Solution.groups(df)
        ]
        return Solution(n_trees)

    @staticmethod
    def groups(df: pd.DataFrame) -> Any:
        """Extracts groups of trees from the DataFrame."""
        return df.groupby(df.index.astype(str).str.split("_").str[0])
