from decimal import Decimal, getcontext
from functools import cached_property

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Set precision for Decimal
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e15")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(
        self,
        center_x: Decimal = Decimal("0"),
        center_y: Decimal = Decimal("0"),
        angle: Decimal = Decimal("0"),
    ):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.angle: Decimal = angle
        self.center_x: Decimal = center_x
        self.center_y: Decimal = center_y

        # Build the polygon once, rotate then translate
        poly = self.initial_tree_polygon()
        rotated = affinity.rotate(poly, float(angle), origin=(0, 0))
        self.polygon: Polygon = affinity.translate(
            rotated,
            xoff=float(center_x * SCALE_FACTOR),
            yoff=float(center_y * SCALE_FACTOR),
        )

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
                # Start at Tip
                (Decimal("0.0") * SCALE_FACTOR, tip_y * SCALE_FACTOR),
                # Right side - Top Tier
                (top_w / Decimal("2") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                (top_w / Decimal("4") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
                # Right side - Middle Tier
                (mid_w / Decimal("2") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                (mid_w / Decimal("4") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
                # Right side - Bottom Tier
                (base_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                # Right Trunk
                (trunk_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
                (
                    trunk_w / Decimal("2") * SCALE_FACTOR,
                    trunk_bottom_y * SCALE_FACTOR,
                ),
                # Left Trunk
                (
                    -(trunk_w / Decimal("2")) * SCALE_FACTOR,
                    trunk_bottom_y * SCALE_FACTOR,
                ),
                (
                    -(trunk_w / Decimal("2")) * SCALE_FACTOR,
                    base_y * SCALE_FACTOR,
                ),
                # Left side - Bottom Tier
                (
                    -(base_w / Decimal("2")) * SCALE_FACTOR,
                    base_y * SCALE_FACTOR,
                ),
                # Left side - Middle Tier
                (
                    -(mid_w / Decimal("4")) * SCALE_FACTOR,
                    tier_2_y * SCALE_FACTOR,
                ),
                (
                    -(mid_w / Decimal("2")) * SCALE_FACTOR,
                    tier_2_y * SCALE_FACTOR,
                ),
                # Left side - Top Tier
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

    def __repr__(self) -> str:
        return (
            f"ChristmasTree(center_x={self.center_x}, center_y={self.center_y}, "
            f"angle={self.angle})"
        )


class NTree:
    def __init__(self, trees: list[ChristmasTree] | None = None):
        self.trees = trees or []

    def add_tree(self, tree: ChristmasTree) -> None:
        self.trees.append(tree)
        self._invalidate_cache()

    def _invalidate_cache(self):
        """Invalidate cached geometric properties."""
        for attr in ("_union", "bounds", "sides", "side_length"):
            if attr in self.__dict__:
                del self.__dict__[attr]

    @property
    def name(self) -> str:
        return f"{self.tree_count:03d}"

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
        # Force a square bounding with the largest side
        return max(self.sides)

    @property
    def polygons(self) -> list[Polygon]:
        return [t.polygon for t in self.trees]

    @property
    def tree_count(self) -> int:
        return len(self.trees)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "NTree":
        trees = [
            ChristmasTree(
                center_x=Decimal(row["x"]),
                center_y=Decimal(row["y"]),
                angle=Decimal(row["deg"]),
            )
            for _, row in df.iterrows()
        ]
        return NTree(trees)
