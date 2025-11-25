import pandas as pd


def make_submission_df(tree_data: list[list[float]]) -> pd.DataFrame:
    """Creates a submission DataFrame from the tree data."""

    # Build the index of the submission, in the format:
    #  <trees_in_problem>_<tree_index>
    index = [f"{n:03d}_{t}" for n in range(1, 201) for t in range(n)]

    columns = ["x", "y", "deg"]
    df = pd.DataFrame(tree_data, index, columns).rename_axis("id")

    for col in columns:
        df[col] = df[col].astype(float).round(decimals=6)

    # To ensure everything is kept as a string, prepend an 's'
    for col in df.columns:
        df[col] = "s" + df[col].astype("string")
    return df
