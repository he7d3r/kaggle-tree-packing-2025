import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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
import pandas as pd
from tqdm import tqdm

from solution import Solution
from trees import DECIMAL_PLACES, ChristmasTree, NTree, ParticipantVisibleError

optuna.logging.set_verbosity(optuna.logging.WARNING)


TOP_K = 20
BISECTION_TOLERANCE = 10 ** (-DECIMAL_PLACES)
BISECTION_MIN_CLEARANCE = 30 * BISECTION_TOLERANCE

T_co = TypeVar("T_co", covariant=True)
R = TypeVar("R")
T = TypeVar("T")


@dataclass(frozen=True)
class FloatRange:
    """Range specification for float parameters."""

    start: float
    end: float
    step: float = 1.0

    def __iter__(self) -> Iterator[float]:
        """Generate values without accumulating rounding errors."""
        if self.step <= 0:
            raise ValueError("step must be positive")

        n_steps = int((self.end - self.start) / self.step) + 1
        for i in range(n_steps):
            value = self.start + i * self.step
            if value > self.end:
                break
            yield value


class SummaryCollector:
    """Collect per-n solution summary rows."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame()

        return pd.DataFrame.from_records(self._rows).set_index("n").sort_index()

    def to_csv(self, path: str, **kwargs) -> None:
        self.to_dataframe().to_csv(path, **kwargs)


class _SizedContainer(Protocol[T_co]):
    """Protocol for sized iterable containers."""

    def __iter__(self) -> Iterator[T_co]: ...
    def __len__(self) -> int: ...


def _map(
    fn: Callable[[T_co], R],
    items: _SizedContainer[T_co],
    *,
    parallel: bool,
    desc: str,
) -> tuple[R, ...]:
    """Map function over items with optional parallelization."""
    if not parallel:
        return tuple(
            tqdm((fn(item) for item in items), total=len(items), desc=desc)
        )

    with ProcessPoolExecutor() as executor:
        return tuple(tqdm(executor.map(fn, items), total=len(items), desc=desc))


def _bisect_minimal_valid(
    lower_bound: float,
    upper_bound: float,
    validate: Callable[[float], T | None],
    tolerance: float,
    preferred_clearance: float = 0.0,
) -> tuple[float, T]:
    """
    Find minimal value in [lower_bound, upper_bound] where validate succeeds.

    Uses bisection search to find the smallest offset where validation
    succeeds, preferably with at least `preferred_clearance` distance from the
    collision boundary.

    Args:
        lower_bound: Minimum offset to search (typically 0.0)
        upper_bound: Maximum offset to search (must be valid)
        validate: Function returning validated object if offset is valid,
                 None if invalid (e.g., causes collision)
        tolerance: Search precision (stop when range < tolerance)
        preferred_clearance: Preferable distance the returned offset must be
                      from the exact collision boundary, to guard against floating-point issues. Default is 0.0.

    Returns:
        (safe_offset, validated_object)
        The offset is guaranteed to be at least `preferred_clearance` away from
        the exact collision boundary.

    Raises:
        ValueError: If upper_bound is invalid or if the offset with
                   preferred_clearance is invalid.
    """
    # Ensure we start with a valid upper bound
    upper = upper_bound
    upper_result = validate(upper)

    for _ in range(2):  # hard cap
        if upper_result is not None:
            break
        upper *= 2.0
        upper_result = validate(upper)
    else:
        raise ValueError(
            f"Could not find valid upper bound starting from {upper_bound}."
        )

    low = lower_bound
    high = upper
    high_result = upper_result

    # Bisection
    while high - low > tolerance:
        mid = (low + high) / 2
        mid_result = validate(mid)
        if mid_result is None:
            low = mid
        else:
            high = mid
            high_result = mid_result

    # Clearance handling
    if preferred_clearance > 0.0:
        safe_offset = high + preferred_clearance
        safe_result = validate(safe_offset)

        if safe_result is not None:
            return safe_offset, safe_result

    # Fallback: closest valid boundary solution
    return high, high_result


def _validate_trees(trees: tuple[ChristmasTree, ...]) -> NTree | None:
    """
    Validate a collection of trees for collisions.

    Args:
        trees: Tuple of ChristmasTree instances to validate

    Returns:
        NTree if all trees are collision-free, None if collision detected
    """
    try:
        return NTree.from_trees(trees)
    except ParticipantVisibleError:
        return None


# ---------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TileConfig:
    """
    Immutable star topology tile configuration.

    Tree 0 is the anchor (placed at origin).
    Every other tree i is placed relative to tree 0
    along direction directions[i-1].
    """

    angles: tuple[float, ...]  # length = k
    directions: tuple[float, ...]  # length = k-1

    def __post_init__(self) -> None:
        if len(self.angles) < 1:
            raise ValueError("TileConfig must contain at least one tree")
        if len(self.directions) != len(self.angles) - 1:
            raise ValueError("directions must have length len(angles) - 1")

    def build_composite_n_tree(self) -> NTree:
        """
        Build a composite NTree using star topology placement.

        Tree 0 is anchored at the origin. Other trees are positioned
        relative to it using bisection along a direction ray.

        Each placement validates collision-free status by constructing
        a partial NTree, whose __post_init__ validates no overlaps.
        """
        # Anchor tree
        base = ChristmasTree(
            center_x=0.0,
            center_y=0.0,
            angle=self.angles[0],
        )

        current_n_tree = NTree.leaf(base)

        for i, direction in enumerate(self.directions, start=1):
            # Angle for the target tree
            angle = self.angles[i]
            # Direction unit vector
            theta = math.radians(direction)
            ux, uy = math.cos(theta), math.sin(theta)

            def validate(offset: float) -> NTree | None:
                """
                Validate tree placement at given offset.

                Returns NTree if placement is valid, None if collision occurs.
                """
                candidate = ChristmasTree(
                    center_x=base.center_x + offset * ux,
                    center_y=base.center_y + offset * uy,
                    angle=angle,
                )
                return _validate_trees(current_n_tree.trees + (candidate,))

            # Bounding radius of already placed trees relative to anchor
            max_existing_radius = max(
                math.hypot(
                    tree.center_x - base.center_x,
                    tree.center_y - base.center_y,
                )
                + tree.side_length
                for tree in current_n_tree.trees
            )
            candidate_radius = ChristmasTree(angle=angle).side_length
            safe_upper = 2.0 * (max_existing_radius + candidate_radius)

            # Use the validated NTree directly - no need to reconstruct
            _, current_n_tree = _bisect_minimal_valid(
                lower_bound=0.0,
                upper_bound=safe_upper,
                validate=validate,
                tolerance=BISECTION_TOLERANCE,
                preferred_clearance=BISECTION_MIN_CLEARANCE,
            )

        return current_n_tree


@dataclass(frozen=True)
class TileFamily:
    """
    Defines a family of star-topology tiles with fixed arity k.
    """

    k: int
    angle_range: FloatRange
    direction_range: FloatRange

    @property
    def param_names(self) -> tuple[str, ...]:
        return (
            *(f"angle_{i}" for i in range(self.k)),
            *(f"dir_{i}" for i in range(1, self.k)),
        )

    def build_config(self, params: Mapping[str, float]) -> TileConfig:
        angles = tuple(params[f"angle_{i}"] for i in range(self.k))
        directions = tuple(params[f"dir_{i}"] for i in range(1, self.k))
        return TileConfig(angles=angles, directions=directions)


TILE_FAMILIES = (
    TileFamily(
        k=2,
        angle_range=FloatRange(0.0, 180.0, 5.0),
        direction_range=FloatRange(0.0, 180.0, 5.0),
    ),
    TileFamily(
        k=3,
        angle_range=FloatRange(0.0, 180.0, 20.0),
        direction_range=FloatRange(0.0, 180.0, 20.0),
    ),
)


def _validate_horizontal_offset(n_tree: NTree, offset: float) -> NTree | None:
    """
    Validate horizontal offset for tiling.

    Returns translated NTree if offset is valid, None if collision occurs.
    """
    translated_trees = tuple(
        ChristmasTree(
            center_x=tree.center_x + offset,
            center_y=tree.center_y,
            angle=tree.angle,
        )
        for tree in n_tree.trees
    )
    return _validate_trees(n_tree.trees + translated_trees)


def _validate_vertical_offset(
    n_tree: NTree, horizontal_n_tree: NTree, offset: float
) -> NTree | None:
    """
    Validate vertical offset for tiling considering horizontal neighbors.

    Tests collision pairs introduced by the vertical dimension:
    1. Original ↔ Vertical (direct vertical neighbor)
    2. Horizontal ↔ Vertical (secondary diagonal)
    3. Original ↔ Diagonal (main diagonal)

    Returns NTree of vertical neighbors if valid, None if collision occurs.
    """
    original_trees = n_tree.trees

    vertical_trees = tuple(
        ChristmasTree(
            center_x=tree.center_x,
            center_y=tree.center_y + offset,
            angle=tree.angle,
        )
        for tree in original_trees
    )

    if _validate_trees(original_trees + vertical_trees) is None:
        return None

    if _validate_trees(horizontal_n_tree.trees + vertical_trees) is None:
        return None

    diagonal_trees = tuple(
        ChristmasTree(
            center_x=tree.center_x,
            center_y=tree.center_y + offset,
            angle=tree.angle,
        )
        for tree in horizontal_n_tree.trees
    )

    if _validate_trees(original_trees + diagonal_trees) is None:
        return None

    return NTree.from_trees(vertical_trees)


@dataclass(frozen=True)
class TileMetrics:
    """Hashable/frozen class to store pre-computed expensive metrics."""

    width: float
    height: float
    dx: float
    dy: float

    @classmethod
    def from_n_tree(cls, n_tree: NTree) -> "TileMetrics":
        """
        Compute tile metrics from an NTree using NTree-based collision detection.

        Finds minimal horizontal (dx) and vertical (dy) offsets that allow
        tiling without collisions by testing with actual NTree validation.
        """
        width, height = n_tree.sides
        safe_upper_bound = max(width, height) * 2

        # Find minimal horizontal offset (dx)
        dx, horizontal_n_tree = _bisect_minimal_valid(
            lower_bound=0.0,
            upper_bound=safe_upper_bound,
            validate=partial(_validate_horizontal_offset, n_tree),
            tolerance=BISECTION_TOLERANCE,
            preferred_clearance=BISECTION_MIN_CLEARANCE,
        )

        # Find minimal vertical offset (dy)
        dy, _ = _bisect_minimal_valid(
            lower_bound=0.0,
            upper_bound=safe_upper_bound,
            validate=partial(
                _validate_vertical_offset, n_tree, horizontal_n_tree
            ),
            tolerance=BISECTION_TOLERANCE,
            preferred_clearance=BISECTION_MIN_CLEARANCE,
        )

        return cls(width=width, height=height, dx=dx, dy=dy)

    def coordinates(self, col: int, row: int) -> tuple[float, float]:
        """Return coordinates for a given grid position."""
        return col * self.dx, row * self.dy

    def bounding_square_side(self, n_cols: int, n_rows: int) -> float:
        """Compute bounding square side for given grid dimensions."""
        x, y = self.coordinates(n_cols, n_rows)
        return max(x + self.width, y + self.height)


@dataclass(frozen=True)
class TilePattern:
    """Complete tile pattern with configuration, metrics, and base NTree."""

    config: TileConfig
    metrics: TileMetrics
    base_n_tree: NTree

    @classmethod
    @lru_cache(maxsize=20_000)
    def from_tile_config(cls, config: TileConfig) -> "TilePattern":
        """Create a TilePattern from a TileConfig."""
        base_n_tree = config.build_composite_n_tree()
        metrics = TileMetrics.from_n_tree(base_n_tree)
        return cls(config=config, metrics=metrics, base_n_tree=base_n_tree)

    def build_n_tree(self, positions: tuple[tuple[int, int], ...]) -> NTree:
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
    """Complete tiling with pattern, positions, and bounding side."""

    pattern: TilePattern
    positions: tuple[tuple[int, int], ...]
    side: float


@runtime_checkable
class WarmStartCapable(Protocol):
    """Protocol for evaluators that support warm-starting."""

    def warm_start(self, patterns: Sequence[TilePattern]) -> None: ...


class PatternEvaluator:
    """Strategy interface for selecting the best tiling among patterns."""

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        """Evaluate patterns and return best tiling."""
        ...

    def elite_patterns(self) -> Sequence[TilePattern]:
        """Return patterns recommended for warm-starting the next solve step."""
        return ()

    @property
    def execution_mode(self) -> Literal["sequential", "parallel"]:
        """Return execution mode for this evaluator."""
        ...


class BruteForceEvaluator(PatternEvaluator):
    """Brute force evaluator trying all discrete parameter combinations."""

    def __init__(self, parallel: bool = True, top_k: int = 1) -> None:
        self.patterns = self.precompute_patterns(parallel=parallel)
        self.top_k = top_k
        self._elite_patterns: tuple[TilePattern, ...] = ()

    @property
    def execution_mode(self) -> Literal["parallel"]:
        return "parallel"

    @classmethod
    def precompute_patterns(
        cls, *, parallel: bool = True
    ) -> tuple[TilePattern, ...]:
        """Precompute all tile patterns from TILE_FAMILIES parameter grid."""
        patterns: list[TilePattern] = []

        for family in TILE_FAMILIES:
            grid = {
                name: tuple(
                    family.angle_range
                    if "angle" in name
                    else family.direction_range
                )
                for name in family.param_names
            }

            configs = (
                family.build_config(params)
                for params in expand_param_grid(grid)
            )

            patterns.extend(
                _map(
                    TilePattern.from_tile_config,
                    tuple(configs),
                    parallel=parallel,
                    desc=f"Pre-computing k={family.k} tile patterns",
                )
            )

        return tuple(patterns)

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        """Evaluate all patterns and return the best one."""
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
        n_trials: int,
        seed: int,
        warm_start: Sequence[TilePattern] = (),
        top_k: int = 1,
    ) -> None:
        self.n_trials = n_trials
        self.seed = seed
        self.top_k = top_k

        # Patterns injected by Solver before evaluate()
        self._warm_patterns: tuple[TilePattern, ...] = tuple(warm_start)

        # Patterns discovered in the last run
        self._elite_patterns: tuple[TilePattern, ...] = ()

    @property
    def execution_mode(self) -> Literal["sequential"]:
        return "sequential"

    def warm_start(self, patterns: Sequence[TilePattern]) -> None:
        """Set warm-start patterns for the next evaluation."""
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
            n_trials=self.n_trials,
            seed=self.seed,
            warm_start=self._warm_patterns,
            top_k=self.top_k,
        )

        best = evaluator._run(tree_count, construct)

        # Propagate elite patterns back to this instance
        self._elite_patterns = evaluator._elite_patterns

        return best

    def _run(
        self,
        tree_count: int,
        construct: Callable[[int, TilePattern], Tiling],
    ) -> Tiling:
        """Internal implementation of evaluation."""
        best: Tiling | None = None
        best_k: list[Tiling] = []

        def objective(trial: optuna.Trial) -> float:
            nonlocal best
            k = trial.suggest_categorical(
                "k", [family.k for family in TILE_FAMILIES]
            )

            family = next(f for f in TILE_FAMILIES if f.k == k)

            params = {}

            for i in range(k):
                params[f"angle_{i}"] = trial.suggest_float(
                    f"angle_{i}",
                    family.angle_range.start,
                    family.angle_range.end,
                )

            for i in range(1, k):
                params[f"dir_{i}"] = trial.suggest_float(
                    f"dir_{i}",
                    family.direction_range.start,
                    family.direction_range.end,
                )

            tile_config = family.build_config(params)
            tile_pattern = TilePattern.from_tile_config(tile_config)
            tiling = construct(tree_count, tile_pattern)

            if best is None or tiling.side < best.side:
                best = tiling

            best_k.append(tiling)
            best_k.sort(key=lambda t: t.side)
            del best_k[self.top_k :]

            return tiling.side

        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            direction="minimize",
        )

        # Enqueue warm-start trials
        for pattern in self._warm_patterns:
            cfg = pattern.config
            k = len(cfg.angles)

            trial_params: dict[str, float | int] = {"k": k}

            trial_params |= {f"angle_{i}": a for i, a in enumerate(cfg.angles)}
            trial_params |= {
                f"dir_{i}": d for i, d in enumerate(cfg.directions, start=1)
            }

            study.enqueue_trial(trial_params)

        study.optimize(
            objective, n_trials=self.n_trials, show_progress_bar=False
        )

        self._elite_patterns = tuple(t.pattern for t in best_k)
        assert best is not None
        return best


class HybridEvaluator(PatternEvaluator):
    """Hybrid evaluator combining brute force and Optuna optimization."""

    def __init__(
        self, brute: BruteForceEvaluator, optuna: PatternEvaluator
    ) -> None:
        self.brute = brute
        self.optuna = optuna

    @property
    def execution_mode(self) -> Literal["sequential"]:
        return "sequential"

    def evaluate(
        self, tree_count: int, construct: Callable[[int, TilePattern], Tiling]
    ) -> Tiling:
        """Evaluate using brute force then refine with Optuna."""
        # Generate elite patterns
        self.brute.evaluate(tree_count, construct)
        warm_patterns = list(self.brute.elite_patterns())

        # Warm-start if supported
        if isinstance(self.optuna, WarmStartCapable):
            self.optuna.warm_start(warm_patterns)

        # Delegate to optuna
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
    *,
    strategy: str = "hybrid",
    parallel: bool = True,
    seed: int = 42,
    n_trials: int = 2,
) -> "Solver":
    """
    Create solver with specified strategy.

    Args:
        strategy: "brute", "optuna" or "hybrid" (default)
        parallel: Enable multiprocessing
        seed: Random seed for Optuna
        n_trials: Number of Optuna trials (default: 2)
    """
    if strategy == "hybrid":
        evaluator = HybridEvaluator(
            brute=BruteForceEvaluator(parallel=parallel, top_k=TOP_K),
            optuna=OptunaContinuousEvaluator(
                n_trials=n_trials, top_k=TOP_K, seed=seed
            ),
        )
    elif strategy == "brute":
        evaluator = BruteForceEvaluator(parallel=parallel, top_k=TOP_K)
    elif strategy == "optuna":
        evaluator = OptunaContinuousEvaluator(
            n_trials=n_trials, top_k=TOP_K, seed=seed
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return Solver(evaluator=evaluator, parallel=parallel)


class Solver:
    """Main solver for tree placement optimization."""

    def __init__(self, *, evaluator: PatternEvaluator, parallel: bool = True):
        self._evaluator = evaluator
        self.parallel = parallel

    def solve(
        self,
        problem_sizes: Sequence[int],
        *,
        summary: SummaryCollector | None = None,
    ) -> Solution:
        """Solve the tree placement problem for specified tree counts."""
        if self._evaluator.execution_mode == "parallel":
            results = _map(
                self._solve_for_tree_count,
                problem_sizes,
                parallel=self.parallel,
                desc="Placing trees",
            )

            n_trees: list[NTree] = []

            for n_tree, row in results:
                n_trees.append(n_tree)
                if summary is not None:
                    summary.add(row)

            return Solution(n_trees=tuple(n_trees))

        # Sequential (warm-start capable)
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

            n_tree = tiling.pattern.build_n_tree(tiling.positions).take_first(
                tree_count
            )

            if summary is not None:
                cfg = tiling.pattern.config
                summary_row = {
                    "n": tree_count,
                    "score": n_tree.score,
                    "k": len(cfg.angles),
                    **{f"angle_{i}": a for i, a in enumerate(cfg.angles)},
                    **{
                        f"dir_{i}": d
                        for i, d in enumerate(cfg.directions, start=1)
                    },
                }
                summary.add(summary_row)

            n_trees.append(n_tree)

        return Solution(n_trees=tuple(n_trees))

    def _solve_for_tree_count(
        self, tree_count: int
    ) -> tuple[NTree, dict[str, object]]:
        """Solve placement for a single tree count."""
        tiling = self._evaluator.evaluate(
            tree_count=tree_count, construct=self._construct_tiling
        )

        n_tree = tiling.pattern.build_n_tree(tiling.positions).take_first(
            tree_count
        )

        cfg = tiling.pattern.config
        summary_row = {
            "n": tree_count,
            "score": n_tree.score,
            "k": len(cfg.angles),
            **{f"angle_{i}": a for i, a in enumerate(cfg.angles)},
            **{f"dir_{i}": d for i, d in enumerate(cfg.directions, start=1)},
        }

        return n_tree, summary_row

    def _construct_tiling(
        self, tree_count: int, pattern: TilePattern
    ) -> Tiling:
        """Construct a tiling for given tree count and pattern."""
        metrics = pattern.metrics

        positions = [(0, 0)]
        prev_row = 0
        prev_col = 0
        max_row = 0
        max_col = 0
        trees_per_tile = len(pattern.config.angles)

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
