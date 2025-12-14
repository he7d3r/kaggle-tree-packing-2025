from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import partial
from typing import Callable, ClassVar, Sequence, Tuple

from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree, detect_overlap, to_scale
from solution import Solution

BISECTION_TOLERANCE = Decimal("0.000001")


def _valid_h_offset(polygon: Polygon, offset: Decimal) -> bool:
    """Test if horizontal offset causes collision."""
    moved_x = affinity.translate(polygon, xoff=float(to_scale(offset)))
    return detect_overlap(moved_x, polygon)


def _valid_v_offset(polygon: Polygon, dx: Decimal, offset: Decimal) -> bool:
    """Test vertical offset collision considering horizontal neighbors."""
    moved_y = affinity.translate(polygon, yoff=float(to_scale(offset)))
    if detect_overlap(moved_y, polygon):
        return True

    moved_x = affinity.translate(polygon, xoff=float(to_scale(dx)))
    if detect_overlap(moved_x, moved_y):
        return True

    moved_xy = affinity.translate(
        polygon, xoff=float(to_scale(dx)), yoff=float(to_scale(offset))
    )
    if detect_overlap(moved_xy, polygon):
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
        ).quantize(BISECTION_TOLERANCE, rounding=ROUND_CEILING)

        # Find minimal vertical offset (dy)
        dy = _bisect_offset(
            lower_bound=Decimal("0.0"),
            upper_bound=height,
            collision_fn=partial(_valid_v_offset, polygon, dx),
            tolerance=BISECTION_TOLERANCE,
        ).quantize(BISECTION_TOLERANCE, rounding=ROUND_CEILING)

        return cls(
            angle=angle,
            width=width,
            height=height,
            dx=dx,
            dy=dy,
            bounding_rectangle_area=bounding_rectangle_area,
        )


def get_default_solver(parallel: bool = True) -> "Solver":
    return Solver(parallel=parallel)


def _solve_single_helper(args):
    """Helper for parallel execution to unpack arguments."""
    solver, tree_count = args
    return solver._solve_single(tree_count)


def _precompute_grid_params_helper(angle: Decimal) -> RotatedTreeGridParams:
    """Helper for parallel grid pre-computation."""
    return RotatedTreeGridParams.from_angle(angle)


class Solver:
    ANGLES: ClassVar[Tuple[Decimal, ...]] = tuple(
        Decimal(a / 64) for a in range(0, 1 + 90 * 64)
    )

    # Store the pre-computed parameters
    _GRID_PARAMS: Tuple[RotatedTreeGridParams, ...]

    def __init__(self, parallel: bool = True):
        self.parallel = parallel
        self._GRID_PARAMS = self._precompute_grid_params()

    def _precompute_grid_params(self) -> Tuple[RotatedTreeGridParams, ...]:
        """
        Computes RotatedTreeGridParams for all angles.
        Runs in parallel if self.parallel is True.
        """
        if not self.parallel:
            return tuple(
                RotatedTreeGridParams.from_angle(angle)
                for angle in tqdm(
                    self.ANGLES, desc="Pre-computing params (seq)"
                )
            )

        with ProcessPoolExecutor() as executor:
            params = tuple(
                tqdm(
                    executor.map(
                        _precompute_grid_params_helper,
                        self.ANGLES,
                    ),
                    total=len(self.ANGLES),
                    desc="Pre-computing params (parallel)",
                )
            )

        return params

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem for the specified n-tree sizes."""

        if not self.parallel:
            n_trees = tuple(
                self._solve_single(tree_count)
                for tree_count in tqdm(
                    problem_sizes, desc="Placing trees (seq)"
                )
            )
            return Solution(n_trees=n_trees)

        with ProcessPoolExecutor() as executor:
            n_trees = tuple(
                tqdm(
                    executor.map(
                        _solve_single_helper,
                        ((self, tree_count) for tree_count in problem_sizes),
                    ),
                    total=len(problem_sizes),
                    desc="Placing trees (parallel)",
                )
            )

        return Solution(n_trees=n_trees)

    def _solve_single(self, tree_count: int) -> NTree:
        """
        Solves the placement for a single tree count, iterating over
        the pre-computed grid parameters.
        """
        best_length = Decimal("Infinity")
        for params in self._GRID_PARAMS:
            coord_fn = partial(to_coordinates, params=params)
            side_fn = partial(bounding_square_side, params=params)
            positions, length = self._solve_single_params(tree_count, side_fn)
            if length < best_length:
                best_length = length
                best_positions = positions
                best_angle = params.angle
                best_coord_fn = coord_fn
        coords = tuple(best_coord_fn(*position) for position in best_positions)
        trees = tuple(ChristmasTree(*xy, angle=best_angle) for xy in coords)
        return NTree(trees=trees)

    def _solve_single_params(
        self,
        tree_count: int,
        compute_side: Callable[[int, int], Decimal],
    ) -> tuple[list[tuple[int, int]], Decimal]:
        positions = [(0, 0)]
        prev_row = 0
        prev_col = 0
        max_row = 0
        max_col = 0
        while len(positions) < tree_count:
            if prev_row == max_row and prev_col == max_col:
                # The previous tree was at the corner of a rectangle.
                # Start a new row or new column (whichever is best)
                side_adding_col = compute_side(max_col + 1, max_row)
                side_adding_row = compute_side(max_col, max_row + 1)
                if side_adding_col <= side_adding_row:
                    row = 0
                    col = max_col + 1
                    max_col += 1
                else:
                    row = max_row + 1
                    col = 0
                    max_row += 1
            elif prev_row == max_row:
                # Continue adding to the previous row until it is full.
                # This does not change max_row and max_col
                row = max_row
                col = prev_col + 1
            elif prev_col == max_col:
                # Continue adding to the previous column until it is full.
                # This does not change max_row and max_col
                row = prev_row + 1
                col = max_col
            else:
                raise Exception("This should not happen.")
            positions.append((col, row))
            prev_row = row
            prev_col = col
        length = compute_side(max_col, max_row)
        return positions, length


def to_coordinates(
    col: int, row: int, params: RotatedTreeGridParams
) -> tuple[Decimal, Decimal]:
    x = Decimal(col) * params.dx
    y = Decimal(row) * params.dy
    return x, y


def bounding_square_side(
    n_cols: int, n_rows: int, params: RotatedTreeGridParams
) -> Decimal:
    x, y = to_coordinates(n_cols, n_rows, params)
    x += params.width
    y += params.height
    return max(x, y)
