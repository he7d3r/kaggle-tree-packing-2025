import math
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from typing import Sequence

from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree, to_scale
from solution import Solution


def get_default_solver(parallel: bool = True) -> "Solver":
    return Solver(
        name="Tighter grid with 5 by 5 degree rotations + ProcessPoolExecutor",
        parallel=parallel,
    )


def _solve_single_helper(args):
    solver, tree_count = args
    return solver._solve_single(tree_count)


class Solver:
    ANGLES: tuple[Decimal, ...] = tuple(Decimal(a) for a in range(0, 95, 5))
    WIDTH_INCREMENTS: tuple[int, ...] = (-1, 0)

    def __init__(self, name: str, parallel: bool = True):
        self.name = name
        self.parallel = parallel

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem for the specified n-tree sizes."""

        if not self.parallel:
            # Sequential – simple & profiler-friendly
            n_trees = [
                self._solve_single(tree_count)
                for tree_count in tqdm(
                    problem_sizes, desc="Placing trees (seq)"
                )
            ]
            return Solution(n_trees=tuple(n_trees))

        # Parallel version
        with ProcessPoolExecutor() as executor:
            n_trees = list(
                tqdm(
                    executor.map(
                        _solve_single_helper,
                        ((self, tree_count) for tree_count in problem_sizes),
                    ),
                    total=len(problem_sizes),
                    desc="Placing trees (parallel)",
                )
            )

        return Solution(n_trees=tuple(n_trees))

    def _solve_single(self, tree_count: int) -> NTree:
        best: NTree = NTree()
        best_length = math.inf
        for angle in self.ANGLES:
            tree = ChristmasTree(angle=angle)
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

    def _compute_dx_dy(self, tree: ChristmasTree) -> tuple[Decimal, Decimal]:
        """
        Compute minimal horizontal and vertical grid offsets for a given tree.

        Returns dx, dy such that trees placed at grid positions (i*dx, j*dy)
        do not overlap (touching is allowed). Results are cached.
        """
        width, height = tree.sides
        dx_step_in = width / 32
        dy_step_in = height / 32
        dx = width
        dy = height
        polygon = tree.polygon

        # Find minimal horizontal offset (dx)
        while dx > 0:
            moved_x = affinity.translate(
                polygon, xoff=float(to_scale(dx)), yoff=0.0
            )
            if self._relevant_collision(moved_x, polygon):
                break
            dx -= dx_step_in
        dx += dx_step_in

        # Find minimal vertical offset (dy), checking diagonal collisions
        while dy > 0:
            moved_y = affinity.translate(
                polygon, xoff=0.0, yoff=float(to_scale(dy))
            )
            if self._relevant_collision(moved_y, polygon):
                break
            # Check if horizontally adjacent trees collide with
            # vertically adjacent trees
            moved_x = affinity.translate(
                polygon, xoff=float(to_scale(dx)), yoff=0.0
            )
            if self._relevant_collision(moved_x, moved_y):
                break
            # Check if diagonally adjacent trees collide with origin tree
            moved_xy = affinity.translate(
                polygon, xoff=float(to_scale(dx)), yoff=float(to_scale(dy))
            )
            if self._relevant_collision(moved_xy, polygon):
                break
            dy -= dy_step_in
        dy += dy_step_in

        return dx, dy

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
