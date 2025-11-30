import copy
import math
import random
from decimal import Decimal
from typing import Sequence

from shapely import affinity
from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import SCALE_FACTOR, ChristmasTree, NTree
from solution import Solution


class BaselineIncrementalSolver:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem for 1 to 200 trees."""
        solution = Solution()
        # Initialize an empty list for the first iteration
        n_tree = NTree()
        for tree_count in tqdm(problem_sizes, desc="Placing trees"):
            # Pass the current n_tree to initialize_trees
            n_tree = self.solve_n_tree(
                tree_count, n_tree, batch_size=tree_count - n_tree.tree_count
            )
            solution.add(copy.deepcopy(n_tree))
        return solution

    def solve_n_tree(
        self, tree_count: int, existing_trees: NTree, batch_size: int = 1
    ) -> NTree:
        """
        This builds a simple, greedy starting configuration, by using the previous n-tree
        placements, and adding more tree for the (n+1)-tree configuration. We place a tree
        fairly far away at a (weighted) random angle, and the bring it closer to the center
        until it overlaps. Then we back it up until it no longer overlaps.

        You can easily modify this code to build each n-tree configuration completely
        from scratch.
        """
        for _ in range(batch_size):
            tree_to_place = ChristmasTree(angle=str(self.rng.uniform(0, 360)))
            if not existing_trees.trees:
                # Only place the first tree at origin if starting from scratch
                existing_trees.add_tree(tree_to_place)
                continue
            placed_polygons = existing_trees.polygons
            tree_index = STRtree(placed_polygons)

            best_px = Decimal("0")
            best_py = Decimal("0")
            min_radius = Decimal("Infinity")

            # This loop tries 10 random starting attempts and keeps the best one
            for _ in range(10):
                # The new tree starts at a position 20 from the center, at a random vector angle.
                angle = self.generate_weighted_angle()
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
            existing_trees.add_tree(tree_to_place)

        return existing_trees

    def generate_weighted_angle(self) -> float:
        """
        Generates a random angle with a distribution weighted by abs(sin(2*angle)).
        This helps place more trees in corners, and makes the n_tree less round.
        """
        while True:
            angle = self.rng.uniform(0, 2 * math.pi)
            if self.rng.uniform(0, 1) < abs(math.sin(2 * angle)):
                return angle
