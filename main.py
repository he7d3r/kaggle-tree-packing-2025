import random

import pandas as pd

from metric import score
from solver import solve_all
from submission import make_submission_df

OUTPUT_FILE = "submission.csv"


def main() -> None:
    rng = random.Random(42)
    tree_data = solve_all(rng)
    df = make_submission_df(tree_data)
    df.to_csv(OUTPUT_FILE)
    print(f"Submission saved to {OUTPUT_FILE}")

    df = pd.read_csv(
        OUTPUT_FILE, dtype={"x": "string", "y": "string", "deg": "string"}
    )
    submission_score = score(df)
    print(f"Submission score: {submission_score}")


if __name__ == "__main__":
    main()
