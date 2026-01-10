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
from tqdm import tqdm

from solution import Solution
from trees import DECIMAL_PLACES, ChristmasTree, NTree, ParticipantVisibleError

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


PARAM_GRID = {
    "angle_1": FloatRange(0.0, 180.0, 5.0),
    "angle_2": FloatRange(0.0, 180.0, 5.0),
    "direction_12": FloatRange(0.0, 180.0, 5.0),
}
OPTUNA_N_TRIALS = 200
BISECTION_TOLERANCE = 10 ** (-DECIMAL_PLACES)
BISECTION_MIN_CLEARANCE = 10 * BISECTION_TOLERANCE


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
    min_clearance: float = 0.0,
) -> tuple[float, T]:
    """
    Find minimal value in [lower_bound, upper_bound] where validate succeeds.

    Uses bisection search to find the smallest offset where validation
    succeeds, ensuring a minimum clearance from the collision boundary.

    Args:
        lower_bound: Minimum offset to search (typically 0.0)
        upper_bound: Maximum offset to search (must be valid)
        validate: Function returning validated object if offset is valid,
                 None if invalid (e.g., causes collision)
        tolerance: Search precision (stop when range < tolerance)
        min_clearance: Minimum distance the returned offset must be from the
                      exact collision boundary, to guard against floating-point issues. Default is 0.0.

    Returns:
        (safe_offset, validated_object)
        The offset is guaranteed to be at least `min_clearance` away from
        the exact collision boundary.

    Raises:
        ValueError: If upper_bound is invalid or if the offset with
                   min_clearance is invalid.
    """
    upper_result = validate(upper_bound)
    if upper_result is None:
        raise ValueError(f"Upper bound {upper_bound} is invalid.")

    low = lower_bound
    high = upper_bound

    while high - low > tolerance:
        mid = (low + high) / 2
        if validate(mid) is None:
            low = mid
        else:
            high = mid

    # Ensure minimum clearance from boundary
    safe_offset = high + min_clearance
    safe_result = validate(safe_offset)

    if safe_result is None:
        raise ValueError(
            f"Offset {safe_offset} (bisection result {high} + clearance {min_clearance}) "
            f"is invalid. Upper bound {upper_bound} may be insufficient."
        )

    return safe_offset, safe_result


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
class TreeSpec:
    """Immutable specification of a single tree in a tile pattern."""

    angle: float  # degrees


@dataclass(frozen=True)
class RelativeTransform:
    """Immutable relative transformation between two trees in a tile pattern."""

    from_idx: int
    to_idx: int
    direction: float  # degrees


@dataclass(frozen=True)
class TileConfig:
    """Immutable tile configuration consisting of tree specs and relations."""

    tree_specs: tuple[TreeSpec, ...]
    relations: tuple[RelativeTransform, ...]

    @classmethod
    def from_params(
        cls, angle_1: float, angle_2: float, direction: float
    ) -> "TileConfig":
        """Construct a TileConfig from float parameters."""
        return cls(
            tree_specs=(TreeSpec(angle=angle_1), TreeSpec(angle=angle_2)),
            relations=(
                RelativeTransform(from_idx=0, to_idx=1, direction=direction),
            ),
        )

    def build_composite_n_tree(self) -> NTree:
        """
        Build a composite NTree using TreeSpec and RelativeTransform.

        Tree 0 is anchored at the origin. Other trees are positioned
        relative to it using bisection along a direction ray.

        Each placement validates collision-free status by constructing
        a partial NTree, whose __post_init__ validates no overlaps.
        """
        if not self.tree_specs:
            raise ValueError("TileConfig must contain at least one TreeSpec")

        # Base tree at the origin
        base_tree = ChristmasTree(
            center_x=0.0,
            center_y=0.0,
            angle=self.tree_specs[0].angle,
        )

        current_n_tree = NTree.leaf(base_tree)

        for rel in self.relations:
            ref_tree = current_n_tree.trees[rel.from_idx]

            # Create target tree at origin (rotation only)
            target = ChristmasTree(angle=self.tree_specs[rel.to_idx].angle)

            # Direction unit vector
            theta = math.radians(rel.direction)
            ux = math.cos(theta)
            uy = math.sin(theta)

            def validate_placement(offset: float) -> NTree | None:
                """
                Validate tree placement at given offset.

                Returns NTree if placement is valid, None if collision occurs.
                """
                candidate_tree = ChristmasTree(
                    center_x=ref_tree.center_x + offset * ux,
                    center_y=ref_tree.center_y + offset * uy,
                    angle=target.angle,
                )
                test_trees = current_n_tree.trees + (candidate_tree,)
                return _validate_trees(test_trees)

            safe_upper_bound = max(ref_tree.side_length, target.side_length) * 2

            _, validated_n_tree = _bisect_minimal_valid(
                lower_bound=0.0,
                upper_bound=safe_upper_bound,
                validate=validate_placement,
                tolerance=BISECTION_TOLERANCE,
                min_clearance=BISECTION_MIN_CLEARANCE,
            )

            # Use the validated NTree directly - no need to reconstruct
            current_n_tree = validated_n_tree

        return current_n_tree


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
            min_clearance=BISECTION_MIN_CLEARANCE,
        )

        # Find minimal vertical offset (dy)
        dy, _ = _bisect_minimal_valid(
            lower_bound=0.0,
            upper_bound=safe_upper_bound,
            validate=partial(
                _validate_vertical_offset, n_tree, horizontal_n_tree
            ),
            tolerance=BISECTION_TOLERANCE,
            min_clearance=BISECTION_MIN_CLEARANCE,
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

    def __init__(self, parallel: bool = True, top_k: int = 5) -> None:
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
        """Precompute all tile patterns from parameter grid."""
        tile_configs = cls._build_tile_configs()
        return _map(
            TilePattern.from_tile_config,
            tile_configs,
            parallel=parallel,
            desc="Pre-computing tile patterns",
        )

    @classmethod
    def _build_tile_configs(cls) -> tuple[TileConfig, ...]:
        """Build all tile configurations from parameter grid."""
        param_grid = {
            param_name: tuple(float_range)
            for param_name, float_range in PARAM_GRID.items()
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
        param_grid: Mapping[str, FloatRange],
        n_trials: int,
        seed: int,
        warm_start: Sequence[TilePattern] = (),
        top_k: int = 5,
    ) -> None:
        self.param_grid = param_grid
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
            param_grid=self.param_grid,
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

            angle_1 = trial.suggest_float(
                "angle_1",
                self.param_grid["angle_1"].start,
                self.param_grid["angle_1"].end,
            )
            angle_2 = trial.suggest_float(
                "angle_2",
                self.param_grid["angle_2"].start,
                self.param_grid["angle_2"].end,
            )
            direction_12 = trial.suggest_float(
                "direction_12",
                self.param_grid["direction_12"].start,
                self.param_grid["direction_12"].end,
            )

            tile_config = TileConfig.from_params(angle_1, angle_2, direction_12)
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
    """Main solver for tree placement optimization."""

    def __init__(self, *, evaluator: PatternEvaluator, parallel: bool = True):
        self._evaluator = evaluator
        self.parallel = parallel

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solve the tree placement problem for specified tree counts."""
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

            n_trees.append(
                tiling.pattern.build_n_tree(tiling.positions).take_first(
                    tree_count
                )
            )

        return Solution(n_trees=tuple(n_trees))

    def _solve_for_tree_count(self, tree_count: int) -> NTree:
        """Solve placement for a single tree count."""
        best = self._evaluator.evaluate(
            tree_count=tree_count, construct=self._construct_tiling
        )
        return best.pattern.build_n_tree(best.positions).take_first(tree_count)

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
