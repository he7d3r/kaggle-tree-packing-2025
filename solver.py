import math
from decimal import Decimal
from typing import Sequence

from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree, to_scale
from solution import Solution


def get_default_solver() -> "Solver":
    return Solver(name="Tighter Grid With Smaller Rotation")


class Solver:
    ANGLES: tuple[int, ...] = (0, 30, 60, 90)
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
        """
        Compute minimal horizontal and vertical offsets such that a translated
        copy of the polygon into either direction by the corresponding offset
        does not overlap (touching is allowed).
        """
        width, height = tree.sides
        dx_step_in = width / 32
        dy_step_in = height / 32
        dx = width
        dy = height
        polygon = tree.polygon
        while dx > 0:
            moved = affinity.translate(
                polygon, xoff=float(to_scale(dx)), yoff=0.0
            )
            if self._relevant_collision(moved, polygon):
                break
            dx -= dx_step_in
        # FIXME: The tree at (0, 1) could collide with (1, 0) but not (0, 0)
        while dy > 0:
            moved = affinity.translate(
                polygon, xoff=0.0, yoff=float(to_scale(dy))
            )
            if self._relevant_collision(moved, polygon):
                break
            dy -= dy_step_in
        return dx + dx_step_in, dy + dy_step_in

    def _relevant_collision(self, a: Polygon, b: Polygon) -> bool:
        return a.intersects(b) and not a.touches(b)

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
