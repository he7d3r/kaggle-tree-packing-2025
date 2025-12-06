import copy
import math
from decimal import Decimal
from typing import Sequence

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
