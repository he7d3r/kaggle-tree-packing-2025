import copy
import math
import random
from decimal import Decimal
from typing import Sequence

from shapely.strtree import STRtree
from tqdm import tqdm

from christmas_tree import ChristmasTree, NTree
from solution import Solution


class BaseSolver:
    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem the specified n-tree sizes."""
        solution = Solution()
        for tree_count in tqdm(problem_sizes, desc="Placing trees"):
            n_tree = self.solve_n_tree(tree_count)
            solution.add(copy.deepcopy(n_tree))
        return solution

    def solve_n_tree(self, tree_count: int) -> NTree:
        raise NotImplementedError


class AlternatingGridLayoutSolver(BaseSolver):
    def solve_n_tree(self, tree_count: int) -> NTree:
        """Arrange `tree_count` Christmas trees in a near-square grid."""
        unit = ChristmasTree()
        width, height = unit.sides

        total_area = width * height * tree_count
        ideal_square_side = Decimal(math.sqrt(total_area))

        # Near-square estimate in counts
        w = max(1, math.ceil(ideal_square_side / width))  # candidate columns
        h = max(1, math.ceil(ideal_square_side / height))  # candidate rows

        # Compute how many rows would be needed if we use w columns (row-first)
        rows_needed_rowfirst = math.ceil(tree_count / w)

        # Compute how many columns would be needed if we use h rows (col-first)
        cols_needed_colfirst = math.ceil(tree_count / h)

        # Physical bounding dimensions for col-first
        w_rowfirst = w * width
        h_rowfirst = rows_needed_rowfirst * height

        # Physical bounding dimensions for row-first
        w_colfirst = cols_needed_colfirst * width
        h_colfirst = h * height

        if max(w_rowfirst, h_rowfirst) < max(w_colfirst, h_colfirst):
            # Position trees row by row
            n_tree = NTree()
            for row in range(rows_needed_rowfirst):
                for col in range(w):
                    x = col * width
                    y = row * height
                    n_tree.add_tree(ChristmasTree(x, y))

                    if n_tree.tree_count == tree_count:
                        return n_tree

            return n_tree
        else:
            # Position trees column by column
            n_tree = NTree()
            for col in range(cols_needed_colfirst):
                for row in range(h):
                    x = col * width
                    y = row * height
                    n_tree.add_tree(ChristmasTree(x, y))

                    if n_tree.tree_count == tree_count:
                        return n_tree

            return n_tree


class IncrementalSolver:
    def solve(self, problem_sizes: Sequence[int]) -> Solution:
        """Solves the tree placement problem the specified n-tree sizes."""
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
        raise NotImplementedError


class BaselineIncrementalSolver(IncrementalSolver):
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng

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
            angle = Decimal(self.rng.uniform(0, 360))
            if not existing_trees.trees:
                # Only place the first tree at origin if starting from scratch
                existing_trees.add_tree(ChristmasTree(angle=angle))
                continue
            placed_polygons = existing_trees.polygons
            tree_index = STRtree(placed_polygons)

            best_px = Decimal("0")
            best_py = Decimal("0")
            min_radius = Decimal("Infinity")

            # This loop tries 10 random starting attempts and keeps the best one
            for _ in range(10):
                # The new tree starts at a position 20 from the center, at a random vector angle.
                vector_angle = self.generate_weighted_angle()
                vx = Decimal(str(math.cos(vector_angle)))
                vy = Decimal(str(math.sin(vector_angle)))

                # Move towards center along the vector in steps of 0.5 until collision
                radius = Decimal("20.0")
                step_in = Decimal("0.5")

                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = ChristmasTree(px, py, angle).polygon

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

                        candidate_poly = ChristmasTree(px, py, angle).polygon

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

            tree_to_place = ChristmasTree(best_px, best_py, angle)
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
