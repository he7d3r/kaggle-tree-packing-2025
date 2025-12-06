import math
from decimal import Decimal
from typing import Sequence

from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree
from solution import Solution


def get_default_solver() -> "Solver":
    return Solver(name="GridWithRotationSolver")


class Solver:
    ANGLES: tuple[int, ...] = (0, 90)
    WIDTH_INCREMENTS: tuple[int, ...] = (-1, 0)

    def __init__(self, name: str):
        self.name = name

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem the specified n-tree sizes."""
        solution = Solution()
        for tree_count in tqdm(problem_sizes, desc="Placing trees"):
            n_tree = self._solve_single(tree_count)
            solution.add(n_tree)
        return solution

    def _solve_single(self, tree_count: int) -> NTree:
        best = NTree()
        best_length = math.inf
        for angle in self.ANGLES:
            tree = ChristmasTree(angle=Decimal(angle))
            base_n_cols = self.estimate_n_cols(tree_count, tree)
            for increment in self.WIDTH_INCREMENTS:
                n_cols = base_n_cols + increment
                if n_cols < 1:
                    continue
                n_tree = self.grid_n_tree(tree, tree_count, n_cols)
                side_length = n_tree.side_length
                if side_length < best_length:
                    best = n_tree
                    best_length = side_length
        return best

    def estimate_n_cols(self, tree_count: int, tree: ChristmasTree) -> int:
        side = self.ideal_square_side(tree, tree_count)
        width = tree.sides[0]
        # Near-square estimate
        return math.ceil(side / width)

    def ideal_square_side(self, tree: ChristmasTree, n: int) -> Decimal:
        tree_area = math.prod(tree.sides)
        total_area = tree_area * n
        return Decimal(total_area).sqrt()

    def grid_n_tree(
        self, base_tree: ChristmasTree, n_trees: int, n_cols: int
    ) -> NTree:
        """Arrange `tree_count` Christmas trees in a near-square grid."""
        width, height = base_tree.sides
        n_rows = math.ceil(n_trees / n_cols)

        # Create a new NTree and a new ChristmasTree for each grid cell
        n_tree = NTree()
        for row in range(n_rows):
            for col in range(n_cols):
                x = col * width
                y = row * height
                positioned = ChristmasTree(
                    center_x=x, center_y=y, angle=base_tree.angle
                )
                n_tree.add_tree(positioned)

                if n_tree.tree_count == n_trees:
                    return n_tree

        return n_tree
