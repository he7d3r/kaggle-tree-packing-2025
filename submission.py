import pandas as pd

from christmas_tree import ChristmasTree, TreePacking


class Submission:
    def __init__(self, packs: list[TreePacking] | None = None):
        self.packs = packs if packs else []

    def add(self, pack: TreePacking) -> None:
        self.packs.append(pack)

    def to_dataframe(self) -> pd.DataFrame:
        """Creates a submission DataFrame from the tree data."""
        data = [
            {
                # Build the index of the submission, in the format:
                #  <trees_in_problem>_<tree_index>
                "id": f"{pack.tree_count:03d}_{t}",
                "x": tree.center_x,
                "y": tree.center_y,
                "deg": tree.angle,
            }
            for pack in self.packs
            for t, tree in enumerate(pack.trees)
        ]
        df = pd.DataFrame(data).set_index("id").astype(float).round(decimals=6)

        # To ensure everything is kept as a string, prepend an 's'
        for col in df.columns:
            df[col] = "s" + df[col].astype("string")
        return df

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "Submission":
        """Populates the Submission object from a DataFrame."""
        df = df.apply(lambda col: col.str.lstrip("s"))
        grouped = df.groupby(df.index.str.split("_").str[0])
        packs = [
            TreePacking.from_dataframe(group_df) for _, group_df in grouped
        ]
        return Submission(packs)
