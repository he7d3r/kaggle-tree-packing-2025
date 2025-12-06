from dataclasses import dataclass, field
from decimal import Decimal, getcontext
from functools import cached_property

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Set precision for Decimal
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e15")


@dataclass(frozen=True)
class ChristmasTree:
    """Immutable single Christmas tree with fixed size and rotation/position."""

    center_x: Decimal = Decimal("0")
    center_y: Decimal = Decimal("0")
    angle: Decimal = Decimal("0")
    polygon: Polygon = field(init=False)

    def __post_init__(self):
        poly = self.initial_tree_polygon()
        rotated = affinity.rotate(poly, float(self.angle), origin=(0, 0))
        translated = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR),
        )
        # Assign to frozen dataclass using object.__setattr__
        object.__setattr__(self, "polygon", translated)

    def initial_tree_polygon(self) -> Polygon:
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
        return Polygon(
            [
                (Decimal("0.0") * SCALE_FACTOR, tip_y * SCALE_FACTOR),
                (top_w / Decimal("2") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (top_w / Decimal("4") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (mid_w / Decimal("2") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (mid_w / Decimal("4") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (base_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                (trunk_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                (
                    trunk_w / Decimal("2") * SCALE_FACTOR,
                    trunk_bottom_y * SCALE_FACTOR,
                ),
                (
                    -(trunk_w / Decimal("2")) * SCALE_FACTOR,
                    trunk_bottom_y * SCALE_FACTOR,
                ),
                (
                    -(trunk_w / Decimal("2")) * SCALE_FACTOR,
                    base_y * SCALE_FACTOR,
                ),
                (
                    -(base_w / Decimal("2")) * SCALE_FACTOR,
                    base_y * SCALE_FACTOR,
                ),
                (
                    -(mid_w / Decimal("4")) * SCALE_FACTOR,
                    tier_2_y * SCALE_FACTOR,
                ),
                (
                    -(mid_w / Decimal("2")) * SCALE_FACTOR,
                    tier_2_y * SCALE_FACTOR,
                ),
                (
                    -(top_w / Decimal("4")) * SCALE_FACTOR,
                    tier_1_y * SCALE_FACTOR,
                ),
                (
                    -(top_w / Decimal("2")) * SCALE_FACTOR,
                    tier_1_y * SCALE_FACTOR,
                ),
            ]
        )

    @cached_property
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        minx, miny, maxx, maxy = self.polygon.bounds
        return (
            Decimal(minx) / SCALE_FACTOR,
            Decimal(miny) / SCALE_FACTOR,
            Decimal(maxx) / SCALE_FACTOR,
            Decimal(maxy) / SCALE_FACTOR,
        )

    @cached_property
    def sides(self) -> tuple[Decimal, Decimal]:
        minx, miny, maxx, maxy = self.bounds
        return maxx - minx, maxy - miny


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
        minx, miny, maxx, maxy = unary_union(self.polygons).bounds
        return (
            Decimal(minx) / SCALE_FACTOR,
            Decimal(miny) / SCALE_FACTOR,
            Decimal(maxx) / SCALE_FACTOR,
            Decimal(maxy) / SCALE_FACTOR,
        )

    @cached_property
    def sides(self) -> tuple[Decimal, Decimal]:
        minx, miny, maxx, maxy = self.bounds
        return maxx - minx, maxy - miny

    @cached_property
    def side_length(self) -> Decimal:
        return max(self.sides)

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
