import math
import random
from decimal import Decimal

from shapely import affinity
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import SCALE_FACTOR, ChristmasTree, TreePacking
from plotter import Plotter


def solve_all(rng: random.Random, plotter: Plotter) -> list[list[float]]:
    """Solves the tree placement problem for 1 to 200 trees."""
    tree_data = []
    current_placed_trees = []  # Initialize an empty list for the first iteration

    for n in tqdm(range(200), desc="Placing trees"):
        # Pass the current_placed_trees to initialize_trees
        tree_packing = initialize_trees(
            n + 1, existing_trees=current_placed_trees, rng=rng
        )
        if (n + 1) % 10 == 0:
            plotter.plot(tree_packing)
        for tree in tree_packing.trees:
            tree_data.append([tree.center_x, tree.center_y, tree.angle])
    return tree_data


def initialize_trees(
    num_trees: int,
    existing_trees: list[ChristmasTree] | None,
    rng: random.Random,
) -> TreePacking:
    """
    This builds a simple, greedy starting configuration, by using the previous n-tree
    placements, and adding more tree for the (n+1)-tree configuration. We place a tree
    fairly far away at a (weighted) random angle, and the bring it closer to the center
    until it overlaps. Then we back it up until it no longer overlaps.

    You can easily modify this code to build each n-tree configuration completely
    from scratch.
    """
    tree_packing = TreePacking(existing_trees)
    if num_trees == 0:
        return tree_packing

    num_to_add = num_trees - len(tree_packing.trees)

    if num_to_add > 0:
        unplaced_trees = [
            ChristmasTree(angle=str(rng.uniform(0, 360)))
            for _ in range(num_to_add)
        ]
        if (
            not tree_packing.trees
        ):  # Only place the first tree at origin if starting from scratch
            tree_packing.add_tree(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            placed_polygons = [p.polygon for p in tree_packing.trees]
            tree_index = STRtree(placed_polygons)

            best_px = Decimal("0")
            best_py = Decimal("0")
            min_radius = Decimal("Infinity")

            # This loop tries 10 random starting attempts and keeps the best one
            for _ in range(10):
                # The new tree starts at a position 20 from the center, at a random vector angle.
                angle = generate_weighted_angle(rng)
                vx = Decimal(str(math.cos(angle)))
                vy = Decimal(str(math.sin(angle)))

                # Move towards center along the vector in steps of 0.5 until collision
                radius = Decimal("20.0")
                step_in = Decimal("0.5")

                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * SCALE_FACTOR),
                        yoff=float(py * SCALE_FACTOR),
                    )

                    # Looking for nearby objects
                    possible_indices = tree_index.query(candidate_poly)
                    # This is the collision detection step
                    if any(
                        (
                            candidate_poly.intersects(placed_polygons[i])
                            and not candidate_poly.touches(placed_polygons[i])
                        )
                        for i in possible_indices
                    ):
                        collision_found = True
                        break
                    radius -= step_in

                # back up in steps of 0.05 until it no longer has a collision.
                if collision_found:
                    step_out = Decimal("0.05")
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy

                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * SCALE_FACTOR),
                            yoff=float(py * SCALE_FACTOR),
                        )

                        possible_indices = tree_index.query(candidate_poly)
                        if not any(
                            (
                                candidate_poly.intersects(placed_polygons[i])
                                and not candidate_poly.touches(
                                    placed_polygons[i]
                                )
                            )
                            for i in possible_indices
                        ):
                            break
                else:
                    # No collision found even at the center. Place it at the center.
                    radius = Decimal("0")
                    px = Decimal("0")
                    py = Decimal("0")

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * SCALE_FACTOR),
                yoff=float(tree_to_place.center_y * SCALE_FACTOR),
            )
            # Add the newly placed tree to the list
            tree_packing.add_tree(tree_to_place)

    return tree_packing


def generate_weighted_angle(rng: random.Random) -> float:
    """
    Generates a random angle with a distribution weighted by abs(sin(2*angle)).
    This helps place more trees in corners, and makes the packing less round.
    """
    while True:
        angle = rng.uniform(0, 2 * math.pi)
        if rng.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle
