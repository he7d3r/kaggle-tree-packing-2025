import math
from dataclasses import dataclass
from decimal import Decimal, getcontext
from functools import cached_property
from typing import cast

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree

# Set precision for Decimal
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e15")


class ParticipantVisibleError(Exception):
    pass


def detect_overlap(a: Polygon, b: Polygon) -> bool:
    """Check for intersection without touching."""
    return a.intersects(b) and not a.touches(b)


def to_scale(value: Decimal) -> Decimal:
    return value * SCALE_FACTOR


def from_scale(value: float) -> Decimal:
    return Decimal(value) / SCALE_FACTOR


def unscaled_bounds(
    bounds: tuple[float, float, float, float],
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    return cast(
        tuple[Decimal, Decimal, Decimal, Decimal],
        tuple(from_scale(v) for v in bounds),
    )


def scaled_points(
    points: tuple[tuple[Decimal, Decimal], ...],
) -> tuple[tuple[Decimal, Decimal], ...]:
    return tuple((to_scale(x), to_scale(y)) for x, y in points)


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

    unscaled_points: tuple[tuple[Decimal, Decimal], ...] = (
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
    return Polygon(scaled_points(unscaled_points))


BASE_TREE: Polygon = _create_initial_tree_polygon()


@dataclass(frozen=True)
class ChristmasTree:
    """Immutable single Christmas tree with fixed size and rotation/position."""

    center_x: Decimal = Decimal("0")
    center_y: Decimal = Decimal("0")
    angle: Decimal = Decimal("0")

    @cached_property
    def polygon(self) -> Polygon:
        rotated = affinity.rotate(BASE_TREE, float(self.angle), origin=(0, 0))
        translated = affinity.translate(
            rotated,
            xoff=float(to_scale(self.center_x)),
            yoff=float(to_scale(self.center_y)),
        )
        return translated

    @cached_property
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return unscaled_bounds(self.polygon.bounds)

    @cached_property
    def sides(self) -> tuple[Decimal, Decimal]:
        minx, miny, maxx, maxy = self.bounds
        return maxx - minx, maxy - miny

    @cached_property
    def bounding_rectangle_area(self) -> Decimal:
        return Decimal(math.prod(self.sides))


@dataclass(frozen=True)
class NTree:
    """Immutable collection of ChristmasTree objects."""

    trees: tuple[ChristmasTree, ...] = ()

    @cached_property
    def tree_count(self) -> int:
        return len(self.trees)

    @cached_property
    def polygons(self) -> tuple[Polygon, ...]:
        return tuple(t.polygon for t in self.trees)

    @cached_property
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        return unscaled_bounds(unary_union(self.polygons).bounds)

    @cached_property
    def sides(self) -> tuple[Decimal, Decimal]:
        minx, miny, maxx, maxy = self.bounds
        return maxx - minx, maxy - miny

    @cached_property
    def side_length(self) -> Decimal:
        return max(self.sides)

    @cached_property
    def score(self) -> Decimal:
        return (self.side_length**2) / Decimal(self.tree_count)

    @property
    def name(self) -> str:
        return f"{self.tree_count:03d}"

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "NTree":
        trees = tuple(
            ChristmasTree(
                center_x=Decimal(row["x"]),
                center_y=Decimal(row["y"]),
                angle=Decimal(row["deg"]),
            )
            for _, row in df.iterrows()
        )
        return NTree(trees=trees)

    def validate(self) -> None:
        """Check for collisions using neighborhood search"""
        polygons = self.polygons
        r_tree = STRtree(polygons)
        # Checking for collisions
        for i, poly in enumerate(polygons):
            indices = r_tree.query(poly)
            for index in indices:
                if index == i:  # don't check against self
                    continue
                if detect_overlap(poly, polygons[index]):
                    raise ParticipantVisibleError(
                        f"Overlapping trees in n-tree {self.name}"
                    )
