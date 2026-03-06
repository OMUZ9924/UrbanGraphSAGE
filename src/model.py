"""GraphSAGE model for superpixel node classification.

Implements a multi-layer GraphSAGE architecture for classifying
superpixel nodes as building vs. non-building.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool


class GraphSAGEClassifier(nn.Module):
    """GraphSAGE-based node classifier for building footprint extraction.

    Architecture:
        Input features → SAGEConv × 3 → Dropout → Linear → Prediction

    Args:
        in_channels: Number of input features per node.
        hidden_channels: Hidden dimension size.
        num_classes: Number of output classes (default: 2 for building/non-building).
        num_layers: Number of GraphSAGE layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 2,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Last conv layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass for node classification.

        Args:
            x: Node feature matrix (N, in_channels).
            edge_index: Graph connectivity (2, E).

        Returns:
            Log-softmax predictions (N, num_classes).
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract node embeddings before classification head.

        Useful for visualization (t-SNE) and transfer learning.
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        return x


class GraphSAGEWithAttention(nn.Module):
    """GraphSAGE variant with attention-based neighbor aggregation.

    Uses GATConv instead of SAGEConv for attention-weighted message passing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_classes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.conv1 = GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=dropout)

        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(self.classifier(x), dim=1)
