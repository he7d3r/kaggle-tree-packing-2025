import math
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TypeVar

import pandas as pd
from shapely import affinity
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

DECIMAL_PLACES: int = 6
G = TypeVar("G", bound=BaseGeometry)


class ParticipantVisibleError(Exception):
    pass


def detect_overlap(a: BaseGeometry, b: BaseGeometry) -> bool:
    """Check for intersection without touching."""
    return a.intersects(b) and not a.touches(b)


def _create_initial_tree_polygon() -> Polygon:
    """Create the base tree polygon."""
    trunk_w = 0.15
    trunk_h = 0.2
    base_w = 0.7
    mid_w = 0.4
    top_w = 0.25
    tip_y = 0.8
    tier_1_y = 0.5
    tier_2_y = 0.25
    base_y = 0.0
    trunk_bottom_y = -trunk_h

    points = (
        (0.0, tip_y),
        (top_w / 2, tier_1_y),
        (top_w / 4, tier_1_y),
        (mid_w / 2, tier_2_y),
        (mid_w / 4, tier_2_y),
        (base_w / 2, base_y),
        (trunk_w / 2, base_y),
        (trunk_w / 2, trunk_bottom_y),
        (-trunk_w / 2, trunk_bottom_y),
        (-trunk_w / 2, base_y),
        (-base_w / 2, base_y),
        (-mid_w / 4, tier_2_y),
        (-mid_w / 2, tier_2_y),
        (-top_w / 4, tier_1_y),
        (-top_w / 2, tier_1_y),
    )
    return Polygon(points)


BASE_TREE: Polygon = _create_initial_tree_polygon()


@lru_cache(maxsize=128)
def _get_rotation_matrix(angle: float) -> tuple[float, float, float, float]:
    """
    Cache rotation matrix components for a given angle.

    Returns (a, b, d, e) for affine transformation matrix:
        | a  b xoff |   | cos -sin  cx |
        | d  e yoff | = | sin  cos  cy |
        | 0  0   1  |   |  0    0   1  |

    Args:
        angle: Rotation angle in degrees

    Returns:
        Tuple of (cos, -sin, sin, cos) for the rotation component
    """
    theta = math.radians(angle)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return cos_theta, -sin_theta, sin_theta, cos_theta


class BoundedGeometryMixin:
    """Mixin providing common geometry properties."""

    @property
    @abstractmethod
    def bounds(self) -> tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) bounds."""
        ...

    @cached_property
    def sides(self) -> tuple[float, float]:
        """Return (width, height) of bounding box."""
        minx, miny, maxx, maxy = self.bounds
        return maxx - minx, maxy - miny

    @cached_property
    def side_length(self) -> float:
        """Return the side length of the bounding square."""
        return max(self.sides)

    @cached_property
    def bounding_rectangle_area(self) -> float:
        """Return area of bounding rectangle."""
        return math.prod(self.sides)


@dataclass(frozen=True)
class ChristmasTree(BoundedGeometryMixin):
    """Immutable single Christmas tree with fixed size and rotation/position."""

    center_x: float = 0.0
    center_y: float = 0.0
    angle: float = 0.0

    def __post_init__(self):
        """Round coordinates and validate tree upon initialization."""
        object.__setattr__(
            self, "center_x", round(self.center_x, DECIMAL_PLACES)
        )
        object.__setattr__(
            self, "center_y", round(self.center_y, DECIMAL_PLACES)
        )
        object.__setattr__(self, "angle", round(self.angle, DECIMAL_PLACES))
        self._validate_limits()
        self._validate_angle()

    def _validate_limits(self) -> None:
        """Validate that coordinates are within bounds."""
        limit = 100.0
        if abs(self.center_x) > limit or abs(self.center_y) > limit:
            raise ParticipantVisibleError(
                f"Tree coordinates ({self.center_x}, {self.center_y}) "
                f"outside bounds of -{limit} to {limit}."
            )

    def _validate_angle(self) -> None:
        """Validate that angle is within [0, 360) range."""
        if self.angle < 0.0 or self.angle >= 360.0:
            raise ParticipantVisibleError(
                f"Tree angle {self.angle} outside valid range [0, 360)."
            )

    @cached_property
    def polygon(self) -> Polygon:
        """Return the transformed polygon for this tree."""
        a, b, d, e = _get_rotation_matrix(self.angle)
        return affinity.affine_transform(
            BASE_TREE, [a, b, d, e, self.center_x, self.center_y]
        )

    @cached_property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return bounds of the tree polygon."""
        return self.polygon.bounds

    @cached_property
    def half_diagonal(self) -> float:
        """Return half the diagonal of the bounding box."""
        w, h = self.sides
        return math.sqrt(w * w + h * h) / 2

    def to_string(self, decimals: int = 1) -> str:
        """Return formatted string representation."""
        fmt = f".{decimals}f"
        return (
            f"ChristmasTree(center_x={self.center_x:{fmt}}, "
            f"center_y={self.center_y:{fmt}}, "
            f"angle={self.angle:{fmt}})"
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
        """Validate structure and check for overlaps."""
        self._validate_structure()
        self._validate_no_overlaps()

    def _validate_structure(self) -> None:
        """Ensure NTree has either a tree or n_trees, but not both."""
        if self.tree is None and not self.n_trees:
            raise ValueError("NTree must have either a tree or n_trees.")

        if self.tree is not None and self.n_trees:
            raise ValueError("NTree cannot have both a tree and n_trees.")

    def _validate_no_overlaps(self) -> None:
        """Check for collisions using spatial indexing."""
        polygons = self.polygons
        if len(polygons) <= 1:
            return

        r_tree = STRtree(polygons)
        for i, poly in enumerate(polygons):
            for j in r_tree.query(poly):
                if i >= j:
                    continue
                if detect_overlap(poly, polygons[j]):
                    trees_str = "\n".join(
                        tree.to_string(decimals=2) for tree in self.trees
                    )
                    raise ParticipantVisibleError(
                        "Overlapping trees in NTree with "
                        f"{self.tree_count} trees.\n"
                        f"All trees:\n{trees_str}"
                    )

    @cached_property
    def trees(self) -> tuple[ChristmasTree, ...]:
        """Flattened tuple of all leaf ChristmasTree objects in this NTree."""
        if self.tree is not None:
            return (self.tree,)
        return tuple(t for n_tree in self.n_trees for t in n_tree.trees)

    @cached_property
    def tree_count(self) -> int:
        """Return total number of trees in this NTree."""
        if self.tree is not None:
            return 1
        return sum(n_tree.tree_count for n_tree in self.n_trees)

    @cached_property
    def polygons(self) -> tuple[Polygon, ...]:
        """Return all polygons in this NTree."""
        if self.tree is not None:
            return (self.tree.polygon,)
        return tuple(p for n_tree in self.n_trees for p in n_tree.polygons)

    @cached_property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return bounds of all trees in this NTree."""
        if self.tree is not None:
            return self.tree.bounds
        return unary_union(self.polygons).bounds

    @cached_property
    def score(self) -> float:
        """Calculate efficiency score: area per tree."""
        return (self.side_length**2) / self.tree_count

    @property
    def name(self) -> str:
        """Return zero-padded tree count as name."""
        return f"{self.tree_count:03d}"

    @staticmethod
    def leaf(tree: ChristmasTree) -> "NTree":
        """Create a leaf NTree from a single ChristmasTree."""
        return NTree(tree=tree)

    @staticmethod
    def combine(*n_trees: "NTree") -> "NTree":
        """Combine multiple NTrees into a single composite NTree."""
        return NTree(n_trees=n_trees)

    @staticmethod
    def from_dataframe(df: pd.DataFrame) -> "NTree":
        """
        Build an NTree from a DataFrame with columns ['x', 'y', 'deg'].

        Each row becomes a leaf NTree; the result is a single composite NTree.
        """
        leaves = tuple(
            NTree.leaf(ChristmasTree(center_x=x, center_y=y, angle=deg))
            for x, y, deg in zip(df["x"], df["y"], df["deg"])
        )
        return NTree._from_leaves(leaves)

    @staticmethod
    def from_trees(trees: tuple[ChristmasTree, ...]) -> "NTree":
        """Build an NTree from a tuple of ChristmasTrees."""
        leaves = tuple(NTree.leaf(t) for t in trees)
        return NTree._from_leaves(leaves)

    @staticmethod
    def _from_leaves(leaves: tuple["NTree", ...]) -> "NTree":
        """Build an NTree from leaf nodes."""
        if not leaves:
            raise ValueError("Cannot build NTree from empty leaves")
        if len(leaves) == 1:
            return leaves[0]
        return NTree.combine(*leaves)

    def take_first(self, n: int) -> "NTree":
        """Return a new NTree with only the first n trees."""
        return NTree.from_trees(self.trees[:n])
