import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from functools import lru_cache, partial
from itertools import product
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import optuna
from shapely import unary_union
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from solution import Solution
from trees import ChristmasTree, GeometryAdapter, NTree, detect_overlap

optuna.logging.set_verbosity(optuna.logging.WARNING)

BISECTION_TOLERANCE = Decimal("0.00005")
T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")


@dataclass(frozen=True)
class DecimalRange:
    start: Decimal
    end: Decimal
    step: Decimal = Decimal("1")


PARAM_GRID = {
    "angle_1": DecimalRange(Decimal("0"), Decimal("180"), Decimal("5")),
    "angle_2": DecimalRange(Decimal("0"), Decimal("180"), Decimal("5")),
    "direction_12": DecimalRange(Decimal("0"), Decimal("180"), Decimal("5")),
}
OPTUNA_N_TRIALS = 200


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

    @classmethod
    def from_params(
        cls, angle_1: Decimal, angle_2: Decimal, direction: Decimal
    ) -> "TileConfig":
        """
        Construct a TileConfig on demand from Decimal parameters.
        """
        return cls(
            tree_specs=(TreeSpec(angle=angle_1), TreeSpec(angle=angle_2)),
            relations=(
                RelativeTransform(from_idx=0, to_idx=1, direction=direction),
            ),
        )

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
    base_n_tree: NTree

    @classmethod
    @lru_cache(maxsize=20_000)
    def from_tile_config(cls, config: TileConfig) -> "TilePattern":
        base_n_tree = config.build_composite_n_tree()
        metrics = TileMetrics.from_n_tree(base_n_tree)
        return cls(config=config, metrics=metrics, base_n_tree=base_n_tree)

    def build_n_tree(self, positions: Iterable[tuple[int, int]]) -> NTree:
        """
        Build a global NTree by placing translated copies of the composite
        tile NTree at each grid position.
        """
        result_trees: list[ChristmasTree] = []

        for col, row in positions:
            dx, dy = self.metrics.coordinates(col, row)

            for tree in self.base_n_tree.trees:
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


@runtime_checkable
class WarmStartCapable(Protocol):
    def warm_start(self, patterns: Sequence[TilePattern]) -> None: ...


class PatternEvaluator:
    """Strategy interface for selecting the best tiling among patterns."""

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling: ...

    def elite_patterns(self) -> Sequence[TilePattern]:
        """Patterns recommended for warm-starting the next solve step."""
        return ()

    @property
    def execution_mode(self) -> Literal["sequential", "parallel"]: ...


class BruteForceEvaluator(PatternEvaluator):
    def __init__(self, parallel: bool = True, top_k: int = 5) -> None:
        self.patterns = self.precompute_patterns(parallel=parallel)
        self.top_k = top_k
        self._elite_patterns: tuple[TilePattern, ...] = ()

    @classmethod
    def precompute_patterns(
        cls, *, parallel: bool = True
    ) -> tuple[TilePattern, ...]:
        tile_configs = cls._build_tile_configs()
        return _map(
            TilePattern.from_tile_config,
            tile_configs,
            parallel=parallel,
            desc="Pre-computing tile patterns",
        )

    @classmethod
    def _build_tile_configs(cls) -> tuple[TileConfig, ...]:
        param_grid = {
            param_name: tuple(
                Decimal(v)
                for v in range(
                    int(decimal_range.start),
                    int(decimal_range.end) + 1,
                    int(decimal_range.step),
                )
            )
            for param_name, decimal_range in PARAM_GRID.items()
        }
        return tuple(
            TileConfig.from_params(
                params["angle_1"], params["angle_2"], params["direction_12"]
            )
            for params in expand_param_grid(param_grid)
        )

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        best_k: list[Tiling] = []

        for pattern in self.patterns:
            tiling = construct(tree_count, pattern)
            best_k.append(tiling)
            best_k.sort(key=lambda t: t.side)
            del best_k[self.top_k :]

        self._elite_patterns = tuple(t.pattern for t in best_k)
        return best_k[0]

    def elite_patterns(self) -> Sequence[TilePattern]:
        return self._elite_patterns


class OptunaContinuousEvaluator(PatternEvaluator):
    """
    Optuna evaluator using a continuous parameter space.
    Patterns are generated on demand from floating-point parameters.
    """

    def __init__(
        self,
        *,
        param_grid: Mapping[str, DecimalRange],
        n_trials: int,
        seed: int,
        warm_start: Sequence[TilePattern] = (),
        top_k: int = 5,
    ) -> None:
        self.param_grid = param_grid
        self.n_trials = n_trials
        self.seed = seed
        self.top_k = top_k

        # patterns injected by Solver before evaluate()
        self._warm_patterns: tuple[TilePattern, ...] = tuple(warm_start)

        # patterns discovered in the last run
        self._elite_patterns: tuple[TilePattern, ...] = ()

    # -------- public API --------

    @property
    def execution_mode(self) -> Literal["sequential"]:
        return "sequential"

    def warm_start(self, patterns: Sequence[TilePattern]) -> None:
        self._warm_patterns = tuple(patterns)

    def elite_patterns(self) -> Sequence[TilePattern]:
        return self._elite_patterns

    def evaluate(
        self,
        tree_count: int,
        construct: Callable[[int, TilePattern], Tiling],
    ) -> Tiling:
        """
        Public entry point. Creates a fresh evaluator to ensure a clean
        Optuna study per tree_count, while propagating warm-start patterns.
        """
        evaluator = OptunaContinuousEvaluator(
            param_grid=self.param_grid,
            n_trials=self.n_trials,
            seed=self.seed,
            warm_start=self._warm_patterns,
            top_k=self.top_k,
        )

        best = evaluator._run(tree_count, construct)

        # propagate elite patterns back to this instance
        self._elite_patterns = evaluator._elite_patterns

        return best

    # -------- internal implementation --------

    def _run(
        self,
        tree_count: int,
        construct: Callable[[int, TilePattern], Tiling],
    ) -> Tiling:
        best: Tiling | None = None
        best_k: list[Tiling] = []

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

            tile_config = TileConfig.from_params(angle_1, angle_2, direction_12)
            tile_pattern = TilePattern.from_tile_config(tile_config)
            tiling = construct(tree_count, tile_pattern)

            if best is None or tiling.side < best.side:
                best = tiling

            best_k.append(tiling)
            best_k.sort(key=lambda t: t.side)
            del best_k[self.top_k :]

            return float(tiling.side)

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            direction="minimize",
        )

        # enqueue warm-start trials
        for pattern in self._warm_patterns:
            cfg = pattern.config
            study.enqueue_trial(
                {
                    "angle_1": cfg.tree_specs[0].angle,
                    "angle_2": cfg.tree_specs[1].angle,
                    "direction_12": cfg.relations[0].direction,
                }
            )

        study.optimize(
            objective, n_trials=self.n_trials, show_progress_bar=False
        )

        self._elite_patterns = tuple(t.pattern for t in best_k)
        assert best is not None
        return best


class HybridEvaluator(PatternEvaluator):
    def __init__(
        self, brute: BruteForceEvaluator, optuna: PatternEvaluator
    ) -> None:
        self.brute = brute
        self.optuna = optuna

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        # 1. Generate elite patterns
        self.brute.evaluate(tree_count, construct)
        warm_patterns = list(self.brute.elite_patterns())

        # 2. Warm-start if supported
        if isinstance(self.optuna, WarmStartCapable):
            self.optuna.warm_start(warm_patterns)

        # 3. Delegate to optuna
        return self.optuna.evaluate(tree_count, construct)

    def elite_patterns(self) -> Sequence[TilePattern]:
        return self.optuna.elite_patterns()


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
    *, strategy: str = "hybrid", parallel: bool = True, seed: int = 42
) -> "Solver":
    """
    Create solver with specified strategy.

    Args:
        strategy: "brute", "optuna" or "hybrid" (default)
        parallel: Enable multiprocessing
        seed: Random seed for Optuna
    """
    if strategy == "hybrid":
        evaluator = HybridEvaluator(
            brute=BruteForceEvaluator(parallel=parallel, top_k=5),
            optuna=OptunaContinuousEvaluator(
                param_grid=PARAM_GRID,
                n_trials=OPTUNA_N_TRIALS,
                seed=seed,
            ),
        )
    elif strategy == "brute":
        evaluator = BruteForceEvaluator(parallel=parallel)
    elif strategy == "optuna":
        evaluator = OptunaContinuousEvaluator(
            param_grid=PARAM_GRID, n_trials=OPTUNA_N_TRIALS, seed=seed
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return Solver(evaluator=evaluator, parallel=parallel)


class Solver:
    def __init__(self, *, evaluator: PatternEvaluator, parallel: bool = True):
        self._evaluator = evaluator
        self.parallel = parallel

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """
        Solves the tree placement problem for the specified n-tree sizes.
        """

        if self._evaluator.execution_mode == "parallel":
            n_trees = list(
                _map(
                    self._solve_for_tree_count,
                    problem_sizes,
                    parallel=self.parallel,
                    desc="Placing trees",
                )
            )
            return Solution(n_trees=tuple(n_trees))

        # --- sequential (warm-start capable) ---
        n_trees: list[NTree] = []
        warm_patterns: Sequence[TilePattern] = ()

        evaluator = self._evaluator
        warm_capable = isinstance(evaluator, WarmStartCapable)

        for tree_count in tqdm(
            problem_sizes, total=len(problem_sizes), desc="Placing trees"
        ):
            if warm_capable:
                evaluator.warm_start(warm_patterns)

            tiling = evaluator.evaluate(
                tree_count=tree_count,
                construct=self._construct_tiling,
            )

            if warm_capable:
                warm_patterns = evaluator.elite_patterns() or (tiling.pattern,)

            n_trees.append(
                tiling.pattern.build_n_tree(tiling.positions).take_first(
                    tree_count
                )
            )

        return Solution(n_trees=tuple(n_trees))

    def _solve_for_tree_count(self, tree_count: int) -> NTree:
        """
        Solves the placement for a single tree count using the configured
        pattern evaluation strategy.
        """
        best = self._evaluator.evaluate(
            tree_count=tree_count, construct=self._construct_tiling
        )
        return best.pattern.build_n_tree(best.positions).take_first(tree_count)

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
