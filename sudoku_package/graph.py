from __future__ import annotations

import torch


def build_sudoku_edges(size: int) -> torch.Tensor:
    """Build an undirected graph connecting cells in the same row, column, or box."""
    box_size = int(size ** 0.5)
    if box_size ** 2 != size:
        raise ValueError("size must be a perfect square")

    edges: set[tuple[int, int]] = set()

    def node(row: int, col: int) -> int:
        return row * size + col

    for row in range(size):
        for col in range(size):
            current = node(row, col)

            for other_col in range(size):
                if other_col != col:
                    edges.add((current, node(row, other_col)))

            for other_row in range(size):
                if other_row != row:
                    edges.add((current, node(other_row, col)))

            box_row = (row // box_size) * box_size
            box_col = (col // box_size) * box_size

            for r in range(box_row, box_row + box_size):
                for c in range(box_col, box_col + box_size):
                    if (r, c) != (row, col):
                        edges.add((current, node(r, c)))

    return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()