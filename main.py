import random

from metric import score
from solver import solve_all
from submission import make_submission_df


def main() -> None:
    rng = random.Random(42)
    tree_data = solve_all(rng)
    output_file = "submission.csv"
    df = make_submission_df(tree_data)
    submission_score = score(df.reset_index())
    print(f"Submission score: {submission_score}")
    df.to_csv(output_file)
    print(f"Submission saved to {output_file}")


if __name__ == "__main__":
    main()
