import random

import mlflow
import pandas as pd

from metric import score
from solver import solve_all
from submission import make_submission_df

OUTPUT_FILE = "submission.csv"


def main() -> None:
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Christmas Tree Packing")
    with mlflow.start_run():
        try:
            rng = random.Random(42)
            tree_data = solve_all(rng)
            df = make_submission_df(tree_data)
            df.to_csv(OUTPUT_FILE)
            print(f"Submission saved to {OUTPUT_FILE}")

            df = pd.read_csv(
                OUTPUT_FILE,
                dtype={"x": "string", "y": "string", "deg": "string"},
            )
            submission_score = score(df)
            mlflow.log_metric("submission_score", submission_score)
            print(f"Submission score: {submission_score}")
        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.end_run(status="FAILED")
            raise


if __name__ == "__main__":
    main()
