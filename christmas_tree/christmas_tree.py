from decimal import Decimal, getcontext

from shapely import affinity
from shapely.geometry import Polygon

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
