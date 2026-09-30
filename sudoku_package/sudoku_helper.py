from __future__ import annotations

import csv
import numpy as np
import torch
from IPython.display import HTML, display_html  # type: ignore

from .sudoku import Sudoku


def as_grid(board: Sudoku | torch.Tensor) -> torch.Tensor:
    """Convert either a Sudoku object or tensor to a 2D tensor."""
    if isinstance(board, Sudoku):
        return board.grid

    if board.ndim != 2:
        raise ValueError("Board must be a 2D tensor")

    return board


def display_sudoku(board: Sudoku | torch.Tensor) -> None:
    """Display a Sudoku board as a styled Jupyter HTML table."""
    grid = as_grid(board)
    size = grid.shape[0]
    box_size = int(np.sqrt(size))

    html = "<table style='border-collapse: collapse; font-size:25px; text-align:center;'>"

    for row in range(size):
        html += "<tr>"

        for col in range(size):
            value = grid[row, col].item()
            text = str(value) if value != 0 else "&nbsp;"

            style = (
                "width:40px; height:40px; "
                "border:1px solid black; "
                "text-align:center; vertical-align:middle;"
            )

            if row % box_size == 0:
                style += "border-top:3px solid black;"
            if col % box_size == 0:
                style += "border-left:3px solid black;"
            if row == size - 1:
                style += "border-bottom:3px solid black;"
            if col == size - 1:
                style += "border-right:3px solid black;"

            style += "color:gray;" if value == 0 else "font-weight:bold;"

            html += f"<td style='{style}'>{text}</td>"

        html += "</tr>"

    html += "</table>"
    display_html(HTML(html))


def append_sudoku_to_csv(
    board: Sudoku | torch.Tensor,
    filename: str,
    mode: str = "a",
) -> None:
    """Append a flattened Sudoku board to a CSV file."""
    grid = as_grid(board)

    with open(filename, mode, newline="") as file:
        csv.writer(file).writerow(grid.flatten().tolist())