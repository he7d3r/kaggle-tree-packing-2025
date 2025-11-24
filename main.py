import random

from solver.solver import solve_all
from submission.submission import make_submission_df


def main() -> None:
    rng = random.Random(42)
    tree_data = solve_all(rng)
    output_file = "submission.csv"
    df = make_submission_df(tree_data)
    df.to_csv(output_file)
    print(f"Submission saved to {output_file}")


if __name__ == "__main__":
    main()
