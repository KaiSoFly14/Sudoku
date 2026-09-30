from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class SudokuGCN(nn.Module):
    """GCN that predicts the value of every Sudoku cell."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        output_features: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, output_features)
        self.dropout = dropout

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, data.edge_index)

        return F.log_softmax(x, dim=1)