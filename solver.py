import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import partial
from itertools import product
from typing import Callable, ClassVar, Sequence, Tuple

from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree, to_scale
from solution import Solution

BISECTION_TOLERANCE = Decimal("0.000025")


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
    """Hashable/frozen class to store pre-computed grid and area parameters."""

    angle: Decimal
    width: Decimal
    height: Decimal
    dx: Decimal
    dy: Decimal
    bounding_rectangle_area: Decimal

    @classmethod
    def from_angle(cls, angle: Decimal) -> "RotatedTreeGridParams":
        """
        Computes dx, dy, and bounding_rectangle_area for a given angle.
        """
        tree = ChristmasTree(angle=angle)
        width, height = tree.sides
        polygon = tree.polygon
        bounding_rectangle_area = tree.bounding_rectangle_area

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

        return cls(
            angle=angle,
            width=width,
            height=height,
            dx=dx,
            dy=dy,
            bounding_rectangle_area=bounding_rectangle_area,
        )


def get_default_solver(parallel: bool = True) -> "Solver":
    return Solver(
        name="Optimized Bisection-search grid packing: 0-90° rotations at 1° resolution",
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
            n_trees = [
                self._solve_single(tree_count)
                for tree_count in tqdm(
                    problem_sizes, desc="Placing trees (seq)"
                )
            ]
            return Solution(n_trees=tuple(n_trees))

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
        best_length = math.inf
        best_n_tree_args = {}

        all_params = {
            "grid_params": self._GRID_PARAMS,
            "width_increments": self.WIDTH_INCREMENTS,
        }
        for param_combination in product(*all_params.values()):
            params, increment = param_combination
            side = self._ideal_square_side(
                params.bounding_rectangle_area, tree_count
            )
            base_n_cols = self._estimate_n_cols(side, params.dx)
            n_cols = base_n_cols + increment
            if n_cols < 1 or tree_count < n_cols:
                continue
            side_length = compute_side_length(
                tree_count,
                n_cols,
                params.dx,
                params.width,
                params.dy,
                params.height,
            )
            if side_length < best_length:
                best_length = side_length
                best_n_tree_args["angle"] = params.angle
                best_n_tree_args["n_trees"] = tree_count
                best_n_tree_args["n_cols"] = n_cols
                best_n_tree_args["dx"] = params.dx
                best_n_tree_args["dy"] = params.dy
        return self._grid_n_tree(**best_n_tree_args)

    def _estimate_n_cols(self, side: Decimal, dx: Decimal) -> int:
        # Near-square estimate
        return math.ceil(side / dx)

    def _ideal_square_side(
        self, bounding_rectangle_area: Decimal, n: int
    ) -> Decimal:
        """Calculates ideal square side using pre-computed area factor."""
        total_area = bounding_rectangle_area * n
        return Decimal(total_area).sqrt()

    def _grid_n_tree(
        self,
        angle: Decimal,
        n_trees: int,
        n_cols: int,
        dx: Decimal,
        dy: Decimal,
    ) -> NTree:
        """Arrange `n_trees` rotated Christmas trees in a near-square grid."""
        trees = tuple(
            ChristmasTree(
                center_x=(t % n_cols) * dx,
                center_y=(t // n_cols) * dy,
                angle=angle,
            )
            for t in range(n_trees)
        )
        return NTree(trees=trees)


def compute_side_length(
    tree_count: int,
    n_cols: int,
    dx: Decimal,
    width: Decimal,
    dy: Decimal,
    height: Decimal,
) -> Decimal:
    n_rows = (Decimal(tree_count) / Decimal(n_cols)).to_integral_exact(
        rounding=ROUND_CEILING
    )
    width = dx * Decimal(n_cols - 1) + width
    height = dy * Decimal(n_rows - 1) + height
    return max(width, height)
