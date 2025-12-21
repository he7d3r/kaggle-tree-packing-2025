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
    Tuple,
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
    if detect_overlap(moved_xy, geometry):
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
class TileConfig:
    """
    Immutable configuration describing a tiling prototype.
    angle: Decimal
        Rotation angle (in degrees) of the only tree in the Tile pattern.
    """

    angle: Decimal
    fake_param: int  # placeholder for future extension

    def build_n_tree(self) -> NTree:
        tree = ChristmasTree(angle=self.angle)
        return NTree.leaf(tree)


@dataclass(frozen=True)
class PackingTile:
    """Hashable/frozen class to store pre-computed tile and area parameters."""

    angle: Decimal
    width: Decimal
    height: Decimal
    dx: Decimal
    dy: Decimal
    bounding_rectangle_area: Decimal

    @staticmethod
    def from_config(config: TileConfig) -> "PackingTile":
        """
        Stage 1: build a PackingTile from a TileConfig describing
        a single rotated tree.
        """
        n_tree = config.build_n_tree()
        return PackingTile.from_n_tree(n_tree)

    @classmethod
    def from_n_tree(cls, n_tree: NTree) -> "PackingTile":
        """
        Computes dx, dy, and bounding rectangle for a tiling prototype/pattern.
        """
        assert n_tree.tree is not None, "Composite NTrees not supported yet."

        width, height = n_tree.sides
        geometry = unary_union(n_tree.polygons)
        bounding_rectangle_area = width * height

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
            angle=n_tree.tree.angle,
            width=width,
            height=height,
            dx=dx,
            dy=dy,
            bounding_rectangle_area=bounding_rectangle_area,
        )

    def coordinates(self, col: int, row: int) -> tuple[Decimal, Decimal]:
        return Decimal(col) * self.dx, Decimal(row) * self.dy

    def bounding_square_side(self, n_cols: int, n_rows: int) -> Decimal:
        x, y = self.coordinates(n_cols, n_rows)
        x += self.width
        y += self.height
        return max(x, y)


def get_default_solver(parallel: bool = True) -> "Solver":
    return Solver(parallel=parallel)


@dataclass(frozen=True)
class Tiling:
    tile: PackingTile
    positions: tuple[tuple[int, int], ...]
    side: Decimal


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


class Solver:
    PARAM_GRID: ClassVar[dict[str, tuple[Any, ...]]] = {
        "angle": tuple(Decimal(a / 64) for a in range(0, 1 + 90 * 64)),
        "fake_param": (0,),  # single-value placeholder
    }

    # Store the pre-computed tiles
    _tiles: Tuple[PackingTile, ...]

    def __init__(self, parallel: bool = True):
        self.parallel = parallel
        self._tiles = self._precompute_tiles()

    def _precompute_tiles(self) -> tuple[PackingTile, ...]:
        """
        Computes PackingTile for all parameter combinations.
        """
        configs = self._build_configs()
        return _map(
            PackingTile.from_config,
            configs,
            parallel=self.parallel,
            desc="Pre-computing tiles",
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

        for tile in self._tiles:
            candidate = self._construct_tiling(tree_count, tile)
            if best is None or candidate.side < best.side:
                best = candidate

        assert best is not None

        coords = tuple(
            best.tile.coordinates(col, row) for col, row in best.positions
        )
        trees = tuple(
            ChristmasTree(x, y, angle=best.tile.angle) for x, y in coords
        )
        return NTree.from_trees(trees)

    def _construct_tiling(self, tree_count: int, tile: PackingTile) -> Tiling:
        positions = [(0, 0)]
        prev_row = 0
        prev_col = 0
        max_row = 0
        max_col = 0
        while len(positions) < tree_count:
            if prev_row == max_row and prev_col == max_col:
                # The previous tree was at the corner of a rectangle.
                # Start a new row or new column (whichever is best)
                side_adding_col = tile.bounding_square_side(
                    max_col + 1, max_row
                )
                side_adding_row = tile.bounding_square_side(
                    max_col, max_row + 1
                )
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
        length = tile.bounding_square_side(max_col, max_row)
        return Tiling(tile, tuple(positions), length)
