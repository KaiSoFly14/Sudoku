from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from .sudoku import Sudoku


class SudokuDataset(InMemoryDataset):
    """Dataset containing Sudoku puzzles and their complete solutions."""

    def __init__(
        self,
        csv_file: str,
        edge_index: torch.Tensor,
        size: int = 4,
        transform=None,
    ):
        self.size = size
        self.edge_index = edge_index

        super().__init__(root=".", transform=transform)

        dataframe = pd.read_csv(csv_file, header=None)
        data_list: list[Data] = []

        for values in dataframe.values:
            solution = torch.tensor(values, dtype=torch.long)

            if solution.numel() != size * size:
                raise ValueError(
                    f"Each solution must contain {size * size} values"
                )

            solution = solution.reshape(size, size)
            puzzle = Sudoku(solution, size=size).make_minimal()

            flat_puzzle = puzzle.grid.flatten()
            flat_solution = solution.flatten()

            # Classes are 0..size, where 0 represents a blank.
            features = F.one_hot(
                flat_puzzle,
                num_classes=size + 1,
            ).float()

            # Predict only cells that were blank in the puzzle.
            prediction_mask = flat_puzzle == 0

            data_list.append(
                Data(
                    x=features,
                    edge_index=edge_index,
                    y=flat_solution,
                    prediction_mask=prediction_mask,
                )
            )

        self.data, self.slices = self.collate(data_list)

    @property
    def num_classes(self) -> int:
        return self.size + 1
