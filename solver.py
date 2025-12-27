import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import partial
from itertools import product
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
)

import optuna
from shapely import unary_union
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from solution import Solution
from tree import ChristmasTree, GeometryAdapter, NTree, detect_overlap

optuna.logging.set_verbosity(optuna.logging.WARNING)

BISECTION_TOLERANCE = Decimal("0.000005")
T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")


@dataclass(frozen=True)
class DecimalRange:
    start: Decimal
    end: Decimal
    step: Decimal = Decimal("1")


TRIALS = 100
PARAM_GRID = {
    "angle_1": DecimalRange(Decimal("0"), Decimal("90"), Decimal("5")),
    "angle_2": DecimalRange(Decimal("0"), Decimal("180"), Decimal("5")),
    "direction_12": DecimalRange(Decimal("0"), Decimal("90"), Decimal("5")),
}


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
class TreeSpec:
    """Immutable specification of a single tree in a Tile pattern."""

    angle: Decimal  # degrees


@dataclass(frozen=True)
class RelativeTransform:
    """Immutable relative transformation between two trees in a Tile pattern."""

    from_idx: int
    to_idx: int
    direction: Decimal  # degrees


@dataclass(frozen=True)
class TileConfig:
    """Immutable tile configuration consisting of tree specs and relations."""

    tree_specs: tuple[TreeSpec, ...]
    relations: tuple[RelativeTransform, ...]

    def build_composite_n_tree(self) -> NTree:
        """
        Build a composite NTree using TreeSpec and RelativeTransform.

        - Tree 0 is anchored at the origin.
        - Other trees are positioned relative to it using bisection
        along a direction ray.
        """

        if not self.tree_specs:
            raise ValueError("TileConfig must contain at least one TreeSpec")

        # Base tree at the origin
        base_tree = ChristmasTree(
            center_x=Decimal(0),
            center_y=Decimal(0),
            angle=self.tree_specs[0].angle,
        )

        trees = [base_tree]

        for rel in self.relations:
            ref_tree = trees[rel.from_idx]

            # Create target tree at origin (rotation only)
            target = ChristmasTree(angle=self.tree_specs[rel.to_idx].angle)

            # --- direction unit vector ---
            theta = float(rel.direction * Decimal(math.pi) / Decimal(180))
            ux = Decimal(math.cos(theta))
            uy = Decimal(math.sin(theta))

            tgt_geom = target.polygon

            placed_geometries = [t.polygon for t in trees]

            def collision_fn(offset: Decimal) -> bool:
                moved = GeometryAdapter.translate(
                    tgt_geom, dx=offset * ux, dy=offset * uy
                )
                return any(detect_overlap(moved, g) for g in placed_geometries)

            offset = _bisect_offset(
                lower_bound=Decimal("0"),
                upper_bound=max(ref_tree.side_length, target.side_length) * 2,
                collision_fn=collision_fn,
                tolerance=BISECTION_TOLERANCE,
            )

            trees.append(
                ChristmasTree(
                    center_x=ref_tree.center_x + offset * ux,
                    center_y=ref_tree.center_y + offset * uy,
                    angle=target.angle,
                )
            )

        return NTree.from_trees(tuple(trees))


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


@dataclass(frozen=True)
class TileMetrics:
    """Hashable/frozen class to store pre-computed tile and area parameters."""

    width: Decimal
    height: Decimal
    dx: Decimal
    dy: Decimal

    @classmethod
    def from_n_tree(cls, n_tree: NTree) -> "TileMetrics":
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

        return cls(width=width, height=height, dx=dx, dy=dy)

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
        base_n_tree = config.build_composite_n_tree()
        metrics = TileMetrics.from_n_tree(base_n_tree)
        return cls(config=config, metrics=metrics)

    def build_n_tree(self, positions: Iterable[tuple[int, int]]) -> NTree:
        """
        Build a global NTree by placing translated copies of the composite
        tile NTree at each grid position.
        """
        base_tile = self.config.build_composite_n_tree()
        result_trees: list[ChristmasTree] = []

        for col, row in positions:
            dx, dy = self.metrics.coordinates(col, row)

            for tree in base_tile.trees:
                result_trees.append(
                    ChristmasTree(
                        center_x=tree.center_x + dx,
                        center_y=tree.center_y + dy,
                        angle=tree.angle,
                    )
                )

        return NTree.from_trees(tuple(result_trees))


@dataclass(frozen=True)
class Tiling:
    pattern: TilePattern
    positions: tuple[tuple[int, int], ...]
    side: Decimal


class PatternEvaluator:
    """Strategy interface for selecting the best tiling among patterns."""

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling: ...


class BruteForceEvaluator(PatternEvaluator):
    def __init__(self, parallel: bool = True) -> None:
        patterns = self.precompute_patterns(parallel=parallel)
        self.patterns: tuple[TilePattern, ...] = patterns

    @classmethod
    def precompute_patterns(
        cls, *, parallel: bool = True
    ) -> tuple[TilePattern, ...]:
        configs = cls._build_configs()
        return _map(
            TilePattern.from_config,
            configs,
            parallel=parallel,
            desc="Pre-computing tile patterns",
        )

    @classmethod
    def _build_configs(cls) -> tuple[TileConfig, ...]:
        param_grid = {
            p: tuple(
                Decimal(v)
                for v in range(int(r.start), int(r.end) + 1, int(r.step))
            )
            for p, r in PARAM_GRID.items()
        }
        return tuple(
            tile_config_factory(
                angle_1=params["angle_1"],
                angle_2=params["angle_2"],
                direction=params["direction_12"],
            )
            for params in expand_param_grid(param_grid)
        )

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        best: Tiling | None = None
        for pattern in self.patterns:
            candidate = construct(tree_count, pattern)
            if best is None or candidate.side < best.side:
                best = candidate

        assert best is not None
        return best


def tile_config_factory(
    angle_1: Decimal, angle_2: Decimal, direction: Decimal
) -> TileConfig:
    """
    Construct a TileConfig on demand from Decimal parameters.
    """
    return TileConfig(
        tree_specs=(TreeSpec(angle=angle_1), TreeSpec(angle=angle_2)),
        relations=(
            RelativeTransform(from_idx=0, to_idx=1, direction=direction),
        ),
    )


def tile_factory(
    angle_1: Decimal, angle_2: Decimal, direction: Decimal
) -> TilePattern:
    """
    Construct a TilePattern on demand from Decimal parameters.
    """
    return TilePattern.from_config(
        tile_config_factory(angle_1, angle_2, direction)
    )


class OptunaContinuousEvaluator(PatternEvaluator):
    """
    Optuna evaluator using a continuous parameter space.
    Patterns are generated on demand from floating-point parameters.
    """

    def __init__(
        self,
        *,
        param_grid: Mapping[str, DecimalRange] = PARAM_GRID,
        tile_factory: Callable[[Decimal, Decimal, Decimal], TilePattern],
        n_trials: int,
        seed: int,
        warm_start: Sequence[TilePattern] = (),
    ):
        self.param_grid = param_grid
        self.tile_factory = tile_factory
        self.n_trials = n_trials
        self.seed = seed
        self._warm_start = tuple(warm_start)

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        best: Tiling | None = None

        def objective(trial: optuna.Trial) -> float:
            nonlocal best

            angle_1 = Decimal(
                trial.suggest_float(
                    "angle_1",
                    float(self.param_grid["angle_1"].start),
                    float(self.param_grid["angle_1"].end),
                )
            )
            angle_2 = Decimal(
                trial.suggest_float(
                    "angle_2",
                    float(self.param_grid["angle_2"].start),
                    float(self.param_grid["angle_2"].end),
                )
            )
            direction_12 = Decimal(
                trial.suggest_float(
                    "direction_12",
                    float(self.param_grid["direction_12"].start),
                    float(self.param_grid["direction_12"].end),
                )
            )

            # Generate TilePattern on demand
            pattern = self.tile_factory(angle_1, angle_2, direction_12)
            tiling = construct(tree_count, pattern)

            if best is None or tiling.side < best.side:
                best = tiling

            return float(tiling.side)

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            direction="minimize",
        )

        for pattern in self._warm_start:
            cfg = pattern.config
            study.enqueue_trial(
                {
                    "angle_1": cfg.tree_specs[0].angle,
                    "angle_2": cfg.tree_specs[1].angle,
                    "direction_12": cfg.relations[0].direction,
                }
            )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=False,
            n_jobs=-1,
        )

        assert best is not None
        return best


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


def get_default_solver(
    *, strategy: str = "brute", parallel: bool = True, seed: int = 42
) -> "Solver":
    """
    Create solver with specified strategy.

    Args:
        strategy: "brute" (default) or "optuna"
        parallel: Enable multiprocessing
        seed: Random seed for Optuna
    """
    if strategy == "brute":
        evaluator = BruteForceEvaluator()
    elif strategy == "optuna":
        evaluator = OptunaContinuousEvaluator(
            param_grid=PARAM_GRID,
            tile_factory=tile_factory,
            n_trials=TRIALS,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return Solver(evaluator=evaluator, parallel=parallel)


class Solver:
    def __init__(
        self, *, evaluator: PatternEvaluator, parallel: bool = True
    ) -> None:
        self.parallel = parallel
        self._evaluator = evaluator

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """
        Solves the tree placement problem for the specified n-tree sizes.

        Warm-starts Optuna across increasing N.
        """
        n_trees: list[NTree] = []

        warm_patterns: list[TilePattern] = []

        for tree_count in tqdm(
            problem_sizes, total=len(problem_sizes), desc="Placing trees"
        ):
            if isinstance(self._evaluator, OptunaContinuousEvaluator):
                evaluator = OptunaContinuousEvaluator(
                    param_grid=self._evaluator.param_grid,
                    tile_factory=self._evaluator.tile_factory,
                    n_trials=self._evaluator.n_trials,
                    seed=self._evaluator.seed,
                    warm_start=warm_patterns,
                )
            else:
                evaluator = self._evaluator
            tiling = evaluator.evaluate(
                tree_count=tree_count, construct=self._construct_tiling
            )

            warm_patterns = [tiling.pattern]

            n_trees.append(
                tiling.pattern.build_n_tree(tiling.positions).take_first(
                    tree_count
                )
            )

        return Solution(n_trees=tuple(n_trees))

    def _construct_tiling(
        self, tree_count: int, pattern: TilePattern
    ) -> Tiling:
        metrics = pattern.metrics

        positions = [(0, 0)]
        prev_row = 0
        prev_col = 0
        max_row = 0
        max_col = 0
        trees_per_tile = len(pattern.config.tree_specs)
        while len(positions) * trees_per_tile < tree_count:
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
