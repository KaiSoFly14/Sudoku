from __future__ import annotations

import random
import torch


class Sudoku:
    """Generic Sudoku board supporting sizes such as 4x4, 9x9, and 16x16."""

    def __init__(self, grid: torch.Tensor | None = None, size: int = 4):
        box_size = int(size ** 0.5)
        if box_size ** 2 != size:
            raise ValueError("size must be a perfect square, such as 4, 9, or 16")

        self.size = size
        self.box_size = box_size

        if grid is None:
            self.grid = torch.zeros((size, size), dtype=torch.long)
        else:
            if grid.shape != (size, size):
                raise ValueError(f"grid must have shape ({size}, {size})")

            self.grid = grid.clone().to(dtype=torch.long)
            self._validate_values()

    def _validate_values(self) -> None:
        if torch.any(self.grid < 0) or torch.any(self.grid > self.size):
            raise ValueError(f"grid values must be between 0 and {self.size}")

    def copy(self) -> Sudoku:
        return Sudoku(self.grid, self.size)

    @classmethod
    def generate(cls, size: int = 4) -> Sudoku:
        """Create a random complete Sudoku solution."""
        sudoku = cls(size=size)

        if not sudoku.solve(randomize=True):
            raise RuntimeError("Unable to generate a Sudoku solution")

        return sudoku

    def find_empty(self) -> tuple[int, int] | None:
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 0:
                    return row, col

        return None

    def is_valid(self, row: int, col: int, value: int) -> bool:
        """Return whether value can be placed at row, col."""
        if not 1 <= value <= self.size:
            return False

        row_values = self.grid[row, :]
        col_values = self.grid[:, col]

        if torch.any(row_values[torch.arange(self.size) != col] == value):
            return False

        if torch.any(col_values[torch.arange(self.size) != row] == value):
            return False

        box_row = self.box_size * (row // self.box_size)
        box_col = self.box_size * (col // self.box_size)

        box = self.grid[
            box_row:box_row + self.box_size,
            box_col:box_col + self.box_size,
        ]

        return not torch.any(box == value)

    def solve(self, randomize: bool = False) -> bool:
        """Solve the board in place."""
        empty = self.find_empty()

        if empty is None:
            return True

        row, col = empty
        values = list(range(1, self.size + 1))

        if randomize:
            random.shuffle(values)

        for value in values:
            if self.is_valid(row, col, value):
                self.grid[row, col] = value

                if self.solve(randomize=randomize):
                    return True

                self.grid[row, col] = 0

        return False

    def count_solutions(self, limit: int = 2) -> int:
        """Count solutions, stopping after limit solutions are found."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        empty = self.find_empty()

        if empty is None:
            return 1

        row, col = empty
        count = 0

        for value in range(1, self.size + 1):
            if self.is_valid(row, col, value):
                self.grid[row, col] = value
                count += self.count_solutions(limit)

                self.grid[row, col] = 0

                if count >= limit:
                    return count

        return count

    def make_minimal(self) -> Sudoku:
        """
        Remove as many values as possible while preserving a unique solution.
        Modifies and returns this Sudoku.
        """
        cells = [
            (row, col)
            for row in range(self.size)
            for col in range(self.size)
        ]
        random.shuffle(cells)

        for row, col in cells:
            backup = self.grid[row, col].item()
            self.grid[row, col] = 0

            if self.copy().count_solutions(limit=2) != 1:
                self.grid[row, col] = backup

        return self