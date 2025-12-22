from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import partial
from itertools import product
from typing import (
    Any,
    Callable,
    ClassVar,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

from shapely import unary_union
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from christmas_tree import ChristmasTree, GeometryAdapter, NTree, detect_overlap
from solution import Solution

T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")


class _SizedContainer(Protocol[T_co]):
    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


def _map(
    fn: Callable[[T_co], R],
    items: _SizedContainer[T_co],
    *,
    parallel: bool,
    desc: str,
) -> tuple[R, ...]:
    if not parallel:
        return tuple(
            tqdm((fn(item) for item in items), total=len(items), desc=desc)
        )

    with ProcessPoolExecutor() as executor:
        return tuple(tqdm(executor.map(fn, items), total=len(items), desc=desc))


BISECTION_TOLERANCE = Decimal("0.000001")


def _has_horizontal_collision(geometry: BaseGeometry, offset: Decimal) -> bool:
    """Test if horizontal offset causes collision."""
    moved_x = GeometryAdapter.translate(geometry, dx=offset)
    return detect_overlap(moved_x, geometry)


def _has_vertical_collision(
    geometry: BaseGeometry, dx: Decimal, offset: Decimal
) -> bool:
    """Test vertical offset collision considering horizontal neighbors."""
    moved_y = GeometryAdapter.translate(geometry, dy=offset)
    if detect_overlap(moved_y, geometry):
        return True

    moved_x = GeometryAdapter.translate(geometry, dx=dx)
    if detect_overlap(moved_x, moved_y):
        return True

    moved_xy = GeometryAdapter.translate(geometry, dx=dx, dy=offset)
    return detect_overlap(moved_xy, geometry)


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


# ---------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TileConfig:
    """
    Immutable configuration describing a tiling prototype.
    angle: Decimal
        Rotation angle (in degrees) of the only tree in the Tile pattern.
    fake_param: int
        Placeholder for future parameters.
    """

    angle: Decimal
    fake_param: int

    def build_n_tree(self) -> NTree:
        return NTree.leaf(ChristmasTree(angle=self.angle))

    def build_tree(self, x: Decimal, y: Decimal) -> ChristmasTree:
        return ChristmasTree(x, y, angle=self.angle)


@dataclass(frozen=True)
class TileMetrics:
    """Hashable/frozen class to store pre-computed tile and area parameters."""

    width: Decimal
    height: Decimal
    dx: Decimal
    dy: Decimal
    bounding_rectangle_area: Decimal

    @classmethod
    def from_n_tree(cls, n_tree: NTree) -> "TileMetrics":
        assert n_tree.tree is not None, "Composite NTrees not supported yet."

        width, height = n_tree.sides
        geometry = unary_union(n_tree.polygons)

        # Find minimal horizontal offset (dx)
        dx = _bisect_offset(
            lower_bound=Decimal("0.0"),
            upper_bound=width,
            collision_fn=partial(_has_horizontal_collision, geometry),
            tolerance=BISECTION_TOLERANCE,
        ).quantize(BISECTION_TOLERANCE, rounding=ROUND_CEILING)

        # Find minimal vertical offset (dy)
        dy = _bisect_offset(
            lower_bound=Decimal("0.0"),
            upper_bound=height,
            collision_fn=partial(_has_vertical_collision, geometry, dx),
            tolerance=BISECTION_TOLERANCE,
        ).quantize(BISECTION_TOLERANCE, rounding=ROUND_CEILING)

        return cls(
            width=width,
            height=height,
            dx=dx,
            dy=dy,
            bounding_rectangle_area=n_tree.bounding_rectangle_area,
        )

    def coordinates(self, col: int, row: int) -> tuple[Decimal, Decimal]:
        return Decimal(col) * self.dx, Decimal(row) * self.dy

    def bounding_square_side(self, n_cols: int, n_rows: int) -> Decimal:
        x, y = self.coordinates(n_cols, n_rows)
        return max(x + self.width, y + self.height)


@dataclass(frozen=True)
class TilePattern:
    config: TileConfig
    metrics: TileMetrics

    @classmethod
    def from_config(cls, config: TileConfig) -> "TilePattern":
        n_tree = config.build_n_tree()
        metrics = TileMetrics.from_n_tree(n_tree)
        return cls(config=config, metrics=metrics)

    def build_n_tree(self, positions: Iterable[tuple[int, int]]) -> NTree:
        trees = tuple(
            self.config.build_tree(*self.metrics.coordinates(col, row))
            for col, row in positions
        )
        return NTree.from_trees(trees)


@dataclass(frozen=True)
class Tiling:
    pattern: TilePattern
    positions: tuple[tuple[int, int], ...]
    side: Decimal


# ---------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------


def expand_param_grid(
    grid: Mapping[str, tuple[Any, ...]],
) -> Iterable[dict[str, Any]]:
    """
    Expand a discrete parameter grid into dictionaries representing
    all combinations (Cartesian product).
    """
    keys = tuple(grid.keys())
    values = tuple(grid[k] for k in keys)
    for combo in product(*values):
        yield dict(zip(keys, combo))


def get_default_solver(parallel: bool = True) -> "Solver":
    return Solver(parallel=parallel)


class Solver:
    PARAM_GRID: ClassVar[dict[str, tuple[Any, ...]]] = {
        "angle": tuple(Decimal(a / 64) for a in range(0, 1 + 90 * 64)),
        "fake_param": (0,),  # placeholder
    }

    _patterns: tuple[TilePattern, ...]

    def __init__(self, parallel: bool = True):
        self.parallel = parallel
        self._patterns = self._precompute_patterns()

    def _precompute_patterns(self) -> tuple[TilePattern, ...]:
        configs = self._build_configs()
        return _map(
            TilePattern.from_config,
            configs,
            parallel=self.parallel,
            desc="Pre-computing tile patterns",
        )

    @classmethod
    def _build_configs(cls) -> tuple[TileConfig, ...]:
        return tuple(
            TileConfig(**params) for params in expand_param_grid(cls.PARAM_GRID)
        )

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem for the specified n-tree sizes."""
        n_trees = _map(
            self._solve_for_tree_count,
            problem_sizes,
            parallel=self.parallel,
            desc="Placing trees",
        )
        return Solution(n_trees=n_trees)

    def _solve_for_tree_count(self, tree_count: int) -> NTree:
        """
        Solves the placement for a single tree count, iterating over
        the pre-computed tile parameters.
        """
        best: Tiling | None = None

        for pattern in self._patterns:
            candidate = self._construct_tiling(tree_count, pattern)
            if best is None or candidate.side < best.side:
                best = candidate

        assert best is not None
        return best.pattern.build_n_tree(best.positions)

    def _construct_tiling(
        self, tree_count: int, pattern: TilePattern
    ) -> Tiling:
        metrics = pattern.metrics

        positions = [(0, 0)]
        prev_row = 0
        prev_col = 0
        max_row = 0
        max_col = 0
        while len(positions) < tree_count:
            if prev_row == max_row and prev_col == max_col:
                # The previous tree was at the corner of a rectangle.
                # Start a new row or new column (whichever is best)
                side_col = metrics.bounding_square_side(max_col + 1, max_row)
                side_row = metrics.bounding_square_side(max_col, max_row + 1)
                if side_col <= side_row:
                    row, col = 0, max_col + 1
                    max_col += 1
                else:
                    row, col = max_row + 1, 0
                    max_row += 1
            elif prev_row == max_row:
                # Continue adding to the previous row until it is full.
                # This does not change max_row and max_col
                row, col = max_row, prev_col + 1
            elif prev_col == max_col:
                # Continue adding to the previous column until it is full.
                # This does not change max_row and max_col
                row, col = prev_row + 1, max_col
            else:
                raise RuntimeError("Invalid tiling state")

            positions.append((col, row))
            prev_row, prev_col = row, col

        side = metrics.bounding_square_side(max_col, max_row)
        return Tiling(pattern, tuple(positions), side)
