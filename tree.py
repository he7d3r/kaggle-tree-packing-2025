import math
from abc import abstractmethod
from dataclasses import dataclass
from decimal import Decimal, getcontext
from functools import cached_property
from typing import TypeVar

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

G = TypeVar("G", bound=BaseGeometry)

# Set precision for Decimal
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e15")


def to_float(d: Decimal) -> float:
    return float(d * SCALE_FACTOR)


def from_float(f: float) -> Decimal:
    return Decimal(f) / SCALE_FACTOR


class GeometryAdapter:
    @staticmethod
    def translate(
        geometry: G, dx: Decimal = Decimal(0), dy: Decimal = Decimal(0)
    ) -> G:
        return affinity.translate(
            geometry, xoff=to_float(dx), yoff=to_float(dy)
        )

    @staticmethod
    def rotate(polygon: Polygon, angle_deg: Decimal) -> Polygon:
        return affinity.rotate(polygon, float(angle_deg), origin=(0.0, 0.0))

    @staticmethod
    def bounds(
        geometry: BaseGeometry,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        minx, miny, maxx, maxy = geometry.bounds
        return (
            from_float(minx),
            from_float(miny),
            from_float(maxx),
            from_float(maxy),
        )


class ParticipantVisibleError(Exception):
    pass


def detect_overlap(a: BaseGeometry, b: BaseGeometry) -> bool:
    """Check for intersection without touching."""
    return a.intersects(b) and not a.touches(b)


def _create_initial_tree_polygon() -> Polygon:
    """Create the base tree polygon."""
    trunk_w = Decimal("0.15")
    trunk_h = Decimal("0.2")
    base_w = Decimal("0.7")
    mid_w = Decimal("0.4")
    top_w = Decimal("0.25")
    tip_y = Decimal("0.8")
    tier_1_y = Decimal("0.5")
    tier_2_y = Decimal("0.25")
    base_y = Decimal("0.0")
    trunk_bottom_y = -trunk_h

    points: tuple[tuple[Decimal, Decimal], ...] = (
        (Decimal("0.0"), tip_y),
        (top_w / 2, tier_1_y),
        (top_w / 4, tier_1_y),
        (mid_w / 2, tier_2_y),
        (mid_w / 4, tier_2_y),
        (base_w / 2, base_y),
        (trunk_w / 2, base_y),
        (trunk_w / 2, trunk_bottom_y),
        (-(trunk_w / 2), trunk_bottom_y),
        (-(trunk_w / 2), base_y),
        (-(base_w / 2), base_y),
        (-(mid_w / 4), tier_2_y),
        (-(mid_w / 2), tier_2_y),
        (-(top_w / 4), tier_1_y),
        (-(top_w / 2), tier_1_y),
    )
    return Polygon([(to_float(x), to_float(y)) for x, y in points])


BASE_TREE: Polygon = _create_initial_tree_polygon()


class BoundedGeometryMixin:
    """Mixin providing common geometry properties."""

    @property
    @abstractmethod
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]: ...

    @cached_property
    def sides(self) -> tuple[Decimal, Decimal]:
        minx, miny, maxx, maxy = self.bounds
        return maxx - minx, maxy - miny

    @cached_property
    def side_length(self) -> Decimal:
        return max(self.sides)

    @cached_property
    def bounding_rectangle_area(self) -> Decimal:
        return Decimal(math.prod(self.sides))


@dataclass(frozen=True)
class ChristmasTree(BoundedGeometryMixin):
    """Immutable single Christmas tree with fixed size and rotation/position."""

    center_x: Decimal = Decimal("0")
    center_y: Decimal = Decimal("0")
    angle: Decimal = Decimal("0")

    def __post_init__(self):
        """Validate tree position and angle upon initialization."""
        self._validate_limits()
        self._validate_angle()

    def _validate_limits(self) -> None:
        """Validate that coordinates are within bounds."""
        limit = Decimal("100")
        if abs(self.center_x) > limit or abs(self.center_y) > limit:
            raise ParticipantVisibleError(
                f"Tree coordinates ({self.center_x}, {self.center_y}) "
                f"outside bounds of -{limit} to {limit}."
            )

    def _validate_angle(self) -> None:
        """Validate that angle is within [0, 360) range."""
        if self.angle < Decimal("0") or self.angle >= Decimal("360"):
            raise ParticipantVisibleError(
                f"Tree angle {self.angle} outside valid range [0, 360)."
            )

    @cached_property
    def polygon(self) -> Polygon:
        poly = GeometryAdapter.rotate(BASE_TREE, self.angle)
        return GeometryAdapter.translate(
            poly, dx=self.center_x, dy=self.center_y
        )

    @cached_property
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return GeometryAdapter.bounds(self.polygon)

    @cached_property
    def half_diagonal(self) -> Decimal:
        w, h = self.sides
        return (w * w + h * h).sqrt() / 2

    def to_string(self, decimals: int = 1) -> str:
        fmt = f".{decimals}f"
        return (
            f"ChristmasTree(center_x={format(float(self.center_x), fmt)}, "
            f"center_y={format(float(self.center_y), fmt)}, "
            f"angle={format(float(self.angle), fmt)})"
        )


@dataclass(frozen=True)
class NTree(BoundedGeometryMixin):
    """
    Immutable recursive tree structure.

    - Leaf node: tree is not None, n_trees is empty.
    - Internal node: tree is None, n_trees is non-empty.
    """

    tree: ChristmasTree | None = None
    n_trees: tuple["NTree", ...] = ()

    def __post_init__(self) -> None:
        self._validate_structure()
        self._validate_no_overlaps()

    def _validate_structure(self) -> None:
        if self.tree is None and not self.n_trees:
            raise ValueError("NTree must have either a tree or n_trees.")

        if self.tree is not None and self.n_trees:
            raise ValueError("NTree cannot have both a tree and n_trees.")

    def _validate_no_overlaps(self) -> None:
        """Check for collisions using neighborhood search."""
        polygons = self.polygons
        if len(polygons) <= 1:
            return

        r_tree = STRtree(polygons)
        # Checking for collisions
        for i, poly in enumerate(polygons):
            for j in r_tree.query(poly):
                if i >= j:
                    continue
                if detect_overlap(poly, polygons[j]):
                    trees = "\n".join(
                        tree.to_string(decimals=2) for tree in self.trees
                    )
                    raise ParticipantVisibleError(
                        "Overlapping trees in NTree with "
                        f"{self.tree_count} trees.\n"
                        f"All trees:\n{trees}"
                    )

    @cached_property
    def trees(self) -> tuple[ChristmasTree, ...]:
        """Flattened tuple of all leaf ChristmasTree objects in this NTree."""
        if self.tree is not None:
            return (self.tree,)
        return tuple(t for n_tree in self.n_trees for t in n_tree.trees)

    @cached_property
    def tree_count(self) -> int:
        if self.tree is not None:
            return 1
        return sum(n_tree.tree_count for n_tree in self.n_trees)

    @cached_property
    def polygons(self) -> tuple[Polygon, ...]:
        if self.tree is not None:
            return (self.tree.polygon,)
        return tuple(p for n_tree in self.n_trees for p in n_tree.polygons)

    @cached_property
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        if self.tree is not None:
            return self.tree.bounds
        return GeometryAdapter.bounds(unary_union(self.polygons))

    @cached_property
    def score(self) -> Decimal:
        return (self.side_length**2) / Decimal(self.tree_count)

    @property
    def name(self) -> str:
        return f"{self.tree_count:03d}"

    @staticmethod
    def leaf(tree: ChristmasTree) -> "NTree":
        return NTree(tree=tree)

    @staticmethod
    def combine(*n_trees: "NTree") -> "NTree":
        return NTree(n_trees=n_trees)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "NTree":
        """
        Build an NTree from a DataFrame with columns ['x', 'y', 'deg'].

        Each row becomes a leaf NTree; the result is a single composite NTree.
        """
        leaves = tuple(
            NTree.leaf(
                ChristmasTree(
                    center_x=Decimal(x),
                    center_y=Decimal(y),
                    angle=Decimal(deg),
                )
            )
            for x, y, deg in zip(df["x"], df["y"], df["deg"])
        )
        return NTree._from_leaves(leaves)

    @staticmethod
    def from_trees(trees: tuple[ChristmasTree, ...]) -> "NTree":
        leaves = tuple(NTree.leaf(t) for t in trees)
        return NTree._from_leaves(leaves)

    @staticmethod
    def _from_leaves(leaves: tuple["NTree", ...]) -> "NTree":
        if not leaves:
            raise ValueError("Cannot build NTree from empty leaves")
        if len(leaves) == 1:
            return leaves[0]
        return NTree.combine(*leaves)

    def take_first(self, n: int) -> "NTree":
        return NTree.from_trees(self.trees[:n])
