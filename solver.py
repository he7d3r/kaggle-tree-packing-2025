import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Callable, ClassVar, Sequence, Tuple

from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree, to_scale
from solution import Solution

BISECTION_TOLERANCE = Decimal("0.0001")


def _relevant_collision(a: Polygon, b: Polygon) -> bool:
    """Check for intersection without touching."""
    return a.intersects(b) and not a.touches(b)


def _valid_h_offset(polygon: Polygon, offset: Decimal) -> bool:
    """Test if horizontal offset causes collision."""
    moved_x = affinity.translate(polygon, xoff=float(to_scale(offset)))
    return _relevant_collision(moved_x, polygon)


def _valid_v_offset(polygon: Polygon, dx: Decimal, offset: Decimal) -> bool:
    """Test vertical offset collision considering horizontal neighbors."""
    moved_y = affinity.translate(polygon, yoff=float(to_scale(offset)))
    if _relevant_collision(moved_y, polygon):
        return True

    moved_x = affinity.translate(polygon, xoff=float(to_scale(dx)))
    if _relevant_collision(moved_x, moved_y):
        return True

    moved_xy = affinity.translate(
        polygon, xoff=float(to_scale(dx)), yoff=float(to_scale(offset))
    )
    if _relevant_collision(moved_xy, polygon):
        return True

    return False


def _bisect_offset(
    lower_bound: Decimal,
    upper_bound: Decimal,
    collision_fn: Callable[[Decimal], bool],
    tolerance: Decimal,
) -> Decimal:
    """Bisection search for minimal offset without collisions."""
    low = lower_bound
    high = upper_bound

    while high - low > tolerance:
        mid = (low + high) / 2
        if collision_fn(mid):
            low = mid  # desired offset is in (mid, high]
        else:
            high = mid  # desired offset is in (low, mid]

    return high


@dataclass(frozen=True)
class RotatedTreeGridParams:
    """Hashable/frozen class to store pre-computed grid parameters."""

    angle: Decimal
    dx: Decimal
    dy: Decimal

    @classmethod
    def from_angle(cls, angle: Decimal) -> "RotatedTreeGridParams":
        """
        Computes dx and dy for a given angle and returns a new instance.
        This method replaces the functionality of Solver._compute_dx_dy.
        """
        tree = ChristmasTree(angle=angle)
        width, height = tree.sides
        polygon = tree.polygon

        # Find minimal horizontal offset (dx)
        dx = _bisect_offset(
            lower_bound=Decimal("0.0"),
            upper_bound=width,
            collision_fn=partial(_valid_h_offset, polygon),
            tolerance=BISECTION_TOLERANCE,
        )

        # Find minimal vertical offset (dy)
        dy = _bisect_offset(
            lower_bound=Decimal("0.0"),
            upper_bound=height,
            collision_fn=partial(_valid_v_offset, polygon, dx),
            tolerance=BISECTION_TOLERANCE,
        )

        return cls(angle=angle, dx=dx, dy=dy)


def get_default_solver(parallel: bool = True) -> "Solver":
    return Solver(
        name="Bisection-search grid packing: 0-90° rotations at 1° resolution",
        parallel=parallel,
    )


def _solve_single_helper(args):
    """Helper for parallel execution to unpack arguments."""
    solver, tree_count = args
    return solver._solve_single(tree_count)


class Solver:
    ANGLES: ClassVar[Tuple[Decimal, ...]] = tuple(
        Decimal(a) for a in range(0, 91, 1)
    )
    WIDTH_INCREMENTS: ClassVar[Tuple[int, ...]] = (-1, 0)

    # Store the pre-computed parameters
    _GRID_PARAMS: Tuple[RotatedTreeGridParams, ...]

    def __init__(self, name: str, parallel: bool = True):
        self.name = name
        self.parallel = parallel
        self._GRID_PARAMS = self._precompute_grid_params()

    def _precompute_grid_params(self) -> Tuple[RotatedTreeGridParams, ...]:
        """
        Computes RotatedTreeGridParams for all angles by calling the
        classmethod constructor on RotatedTreeGridParams.
        """
        return tuple(
            RotatedTreeGridParams.from_angle(angle) for angle in self.ANGLES
        )

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
        """
        Solves the placement for a single tree count, iterating over
        the pre-computed grid parameters.
        """
        best: NTree = NTree()
        best_length = math.inf

        for params in self._GRID_PARAMS:
            angle, dx, dy = params.angle, params.dx, params.dy
            tree = ChristmasTree(angle=angle)
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

    def _estimate_n_cols(self, side: Decimal, dx: Decimal) -> int:
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
