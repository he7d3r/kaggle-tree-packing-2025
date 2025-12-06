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
        n_trees = tuple(
            self._solve_single(tree_count)
            for tree_count in tqdm(problem_sizes, desc="Placing trees")
        )
        return Solution(n_trees=n_trees)

    def _solve_single(self, tree_count: int) -> NTree:
        best: NTree = NTree()
        best_length = math.inf
        for angle in self.ANGLES:
            tree = ChristmasTree(angle=Decimal(angle))
            dx, dy = self._compute_dx_dy(tree)
            side = self._ideal_square_side(tree, tree_count)
            base_n_cols = self._estimate_n_cols(side, dx)
            for increment in self.WIDTH_INCREMENTS:
                n_cols = base_n_cols + increment
                if n_cols < 1:
                    continue
                n_tree = self._grid_n_tree(tree, tree_count, n_cols, dx, dy)
                side_length = n_tree.side_length
                if side_length < best_length:
                    best = n_tree
                    best_length = side_length
        return best

    def _compute_dx_dy(self, tree):
        dx, dy = tree.sides
        return dx, dy

    def _estimate_n_cols(self, side: Decimal, dx: Decimal) -> int:
        # Near-square estimate
        return math.ceil(side / dx)

    def _ideal_square_side(self, tree: ChristmasTree, n: int) -> Decimal:
        total_area = tree.bounding_rectangle_area * n
        return Decimal(total_area).sqrt()

    def _grid_n_tree(
        self,
        base_tree: ChristmasTree,
        n_trees: int,
        n_cols: int,
        dx: Decimal,
        dy: Decimal,
    ) -> NTree:
        """Arrange `n_trees` Christmas trees in a near-square grid."""
        trees = tuple(
            ChristmasTree(
                center_x=(t % n_cols) * dx,
                center_y=(t // n_cols) * dy,
                angle=base_tree.angle,
            )
            for t in range(n_trees)
        )
        return NTree(trees=trees)
