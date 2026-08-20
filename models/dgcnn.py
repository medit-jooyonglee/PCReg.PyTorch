"""Dependency-free DGCNN encoder for pcregmodel.

Only PyTorch operators are used, matching the project's PointNet++ module.
Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds"
https://github.com/WangYueFt/dgcnn
"""

import torch
import torch.nn as nn


def knn(x, k):
    """Indices of the k nearest neighbors in feature space for (B,C,N) input."""
    inner = -2.0 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x, k):
    """Build per-edge features (x_j - x_i, x_i) for the k-NN graph of x."""
    batch_size, channels, point_count = x.size()
    idx = knn(x, k=min(k, point_count))
    idx_base = torch.arange(batch_size, device=x.device).view(-1, 1, 1) * point_count
    idx = (idx + idx_base).view(-1)

    points = x.transpose(2, 1).contiguous()
    neighbors = points.view(batch_size * point_count, -1)[idx, :]
    neighbors = neighbors.view(batch_size, point_count, -1, channels)
    center = points.view(batch_size, point_count, 1, channels).expand_as(neighbors)
    return torch.cat((neighbors - center, center), dim=3).permute(0, 3, 1, 2).contiguous()


class EdgeConv(nn.Module):
    def __init__(self, in_dim, out_dim, gn=False):
        super(EdgeConv, self).__init__()
        layers = [nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=not gn)]
        if gn:
            groups = min(8, out_dim)
            while out_dim % groups:
                groups -= 1
            layers.append(nn.GroupNorm(groups, out_dim))
        layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, k):
        feature = get_graph_feature(x, k=k)
        feature = self.mlp(feature)
        return feature.max(dim=-1)[0]


class DGCNNEncoder(nn.Module):
    """DGCNN encoder returning a global feature vector, dynamically
    recomputing the k-NN graph in feature space at every EdgeConv layer."""

    def __init__(self, in_dim, k=20, mlps=(64, 64, 128, 256),
                 global_dim=1024, gn=False):
        super(DGCNNEncoder, self).__init__()
        self.k = k
        self.edge_convs = nn.ModuleList()
        current = in_dim
        for out_dim in mlps:
            self.edge_convs.append(EdgeConv(current, out_dim, gn=gn))
            current = out_dim

        concat_dim = sum(mlps)
        global_layers = [nn.Conv1d(concat_dim, global_dim, kernel_size=1,
                                   bias=not gn)]
        if gn:
            groups = min(8, global_dim)
            while global_dim % groups:
                groups -= 1
            global_layers.append(nn.GroupNorm(groups, global_dim))
        global_layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.global_mlp = nn.Sequential(*global_layers)
        self.global_dim = global_dim

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError('DGCNNEncoder expects (B, C, N) input')
        features = []
        current = x
        for edge_conv in self.edge_convs:
            current = edge_conv(current, self.k)
            features.append(current)
        concat = torch.cat(features, dim=1)
        return self.global_mlp(concat).max(dim=-1)[0]
