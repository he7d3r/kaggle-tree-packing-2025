import copy
from decimal import Decimal, getcontext

from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Set precision for Decimal
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e15")


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

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

        initial_polygon = Polygon(
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
        rotated = affinity.rotate(
            initial_polygon, float(self.angle), origin=(0, 0)
        )
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * SCALE_FACTOR),
            yoff=float(self.center_y * SCALE_FACTOR),
        )


class TreePacking:
    def __init__(
        self, trees: list[ChristmasTree] | None = None, deepcopy: bool = False
    ):
        self.trees = (
            copy.deepcopy(trees) if deepcopy and trees else (trees or [])
        )

    def add_tree(self, tree: ChristmasTree) -> None:
        self.trees.append(tree)

    @property
    def bounds(self) -> tuple[Decimal, Decimal, Decimal, Decimal]:
        bounds = unary_union(self.polygons).bounds
        minx = Decimal(bounds[0]) / SCALE_FACTOR
        miny = Decimal(bounds[1]) / SCALE_FACTOR
        maxx = Decimal(bounds[2]) / SCALE_FACTOR
        maxy = Decimal(bounds[3]) / SCALE_FACTOR
        return minx, miny, maxx, maxy

    @property
    def sides(self) -> tuple[Decimal, Decimal]:
        minx, miny, maxx, maxy = self.bounds
        width = maxx - minx
        height = maxy - miny
        return width, height

    @property
    def side_length(self) -> Decimal:
        # Force a square bounding with the largest side
        return max(self.sides)

    @property
    def polygons(self) -> list[Polygon]:
        return [t.polygon for t in self.trees]

    @property
    def tree_count(self) -> int:
        return len(self.trees)
