import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MLPLayer(nn.Module):
    """Simple MLP layer with ReLU activation
    """

    def __init__(self, hidden_dim, norm="layer", learn_norm=True, track_norm=False):
        """
        Args:
            hidden_dim: Hidden dimension size (int)
            norm: Feature normalization scheme ("layer"/"batch"/None)
            learn_norm: Whether the normalizer has learnable affine parameters (True/False)
            track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        """
        super(MLPLayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.norm = norm
        self.learn_norm = learn_norm

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm = {
            "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
            "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
        }.get(self.norm, None)

    def forward(self, x):
        batch_size, num_nodes, hidden_dim = x.shape
        x_in = x

        # Linear transformation
        x = self.U(x)

        # Normalize features
        x = self.norm(
            x.view(batch_size*num_nodes, hidden_dim)
        ).view(batch_size, num_nodes, hidden_dim) if self.norm else x

        # Apply non-linearity
        x = F.relu(x)

        # Make residual connection
        x = x_in + x

        return x


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder with ReLU activation, independent of graph structure.
    """
    def __init__(self, n_layers, hidden_dim, norm="layer",
                 learn_norm=True, track_norm=False, *args, **kwargs):
        super(MLPEncoder, self).__init__()
        self.layers = nn.ModuleList(
            MLPLayer(hidden_dim, norm, learn_norm, track_norm) for _ in range(n_layers)
        )

    def forward(self, x, graph=None):
        """
        Args:
            input: Input node features (B x V x H)
        Returns:
            Updated node features (B x V x H)
        """
        for layer in self.layers:
            x = layer(x)

        return x

