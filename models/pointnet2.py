"""Dependency-free PointNet++ building blocks for pcregmodel.

Only PyTorch operators are used so the module also works with the project's
legacy PyTorch setup without compiling pointnet2 CUDA extensions.
"""

import torch
import torch.nn as nn


def square_distance(source, target):
    """Pairwise squared distances for (B,N,3) and (B,M,3)."""
    return (
        (source ** 2).sum(-1, keepdim=True)
        + (target ** 2).sum(-1).unsqueeze(1)
        - 2.0 * torch.matmul(source, target.transpose(1, 2))
    ).clamp_min(0.0)


def index_points(points, indices):
    batch = torch.arange(points.shape[0], device=points.device)
    batch = batch.view(points.shape[0], *([1] * (indices.dim() - 1)))
    batch = batch.expand_as(indices)
    return points[batch, indices]


def farthest_point_sample(points, sample_count):
    """Deterministic batched FPS with shape-preserving index repetition."""
    batch_size, point_count, _ = points.shape
    if point_count == 0:
        raise ValueError('cannot sample an empty point cloud')
    unique_count = min(sample_count, point_count)
    indices = torch.zeros(batch_size, unique_count, dtype=torch.long,
                          device=points.device)
    distances = torch.full((batch_size, point_count), float('inf'),
                           device=points.device)
    farthest = torch.zeros(batch_size, dtype=torch.long, device=points.device)
    batch = torch.arange(batch_size, device=points.device)
    for index in range(unique_count):
        indices[:, index] = farthest
        centroid = points[batch, farthest].unsqueeze(1)
        current = ((points - centroid) ** 2).sum(-1)
        distances = torch.where(current < distances, current, distances)
        farthest = distances.max(-1)[1]
    if unique_count < sample_count:
        repeat_index = torch.arange(sample_count - unique_count,
                                    device=points.device) % unique_count
        indices = torch.cat((indices, indices[:, repeat_index]), dim=1)
    return indices


def query_knn(neighbor_count, points, centers):
    count = min(neighbor_count, points.shape[1])
    indices = square_distance(centers, points).topk(
        count, dim=-1, largest=False
    )[1]
    if count < neighbor_count:
        padding = indices[..., :1].expand(
            *indices.shape[:-1], neighbor_count - count
        )
        indices = torch.cat((indices, padding), dim=-1)
    return indices


class SetAbstraction(nn.Module):
    def __init__(self, sample_count, neighbor_count, in_channels, mlp_channels,
                 coord_dim=3):
        super(SetAbstraction, self).__init__()
        self.sample_count = sample_count
        self.neighbor_count = neighbor_count
        layers = []
        current = in_channels + coord_dim
        for output in mlp_channels:
            layers.extend((nn.Conv2d(current, output, 1), nn.ReLU(inplace=True)))
            current = output
        self.mlp = nn.Sequential(*layers)

    def forward(self, points, features=None):
        sample_indices = farthest_point_sample(points, self.sample_count)
        centers = index_points(points, sample_indices)
        group_indices = query_knn(self.neighbor_count, points, centers)
        grouped_points = index_points(points, group_indices) - centers.unsqueeze(2)
        if features is not None:
            grouped = torch.cat((grouped_points,
                                 index_points(features, group_indices)), dim=-1)
        else:
            grouped = grouped_points
        encoded = self.mlp(grouped.permute(0, 3, 2, 1).contiguous())
        encoded = encoded.max(2)[0].transpose(1, 2).contiguous()
        return centers, encoded


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder returning global and optional input-point features."""

    def __init__(self, in_dim, sample_count1=256, sample_count2=64,
                 feature_dim=256, global_dim=1024, coord_dim=3):
        super(PointNet2Encoder, self).__init__()
        self.coord_dim = coord_dim
        self.input_mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1), nn.ReLU(inplace=True),
        )
        self.sa1 = SetAbstraction(sample_count1, 32, 64, (64, 128, 128),
                                  coord_dim=coord_dim)
        self.sa2 = SetAbstraction(sample_count2, 32, 128,
                                  (128, 256, feature_dim), coord_dim=coord_dim)
        self.global_mlp = nn.Sequential(
            nn.Conv1d(feature_dim, 512, 1), nn.ReLU(inplace=True),
            nn.Conv1d(512, global_dim, 1), nn.ReLU(inplace=True),
        )
        self.global_dim = global_dim
        self.point_dim = 64

    def forward(self, inputs, return_point_features=False):
        if inputs.dim() != 3 or inputs.shape[1] < self.coord_dim:
            raise ValueError(
                f'PointNet2Encoder expects (B,C,N), C >= {self.coord_dim}')
        points = inputs[:, :self.coord_dim].transpose(1, 2).contiguous()
        point_features = self.input_mlp(inputs).transpose(1, 2).contiguous()
        points1, features1 = self.sa1(points, point_features)
        _, features2 = self.sa2(points1, features1)
        global_features = self.global_mlp(
            features2.transpose(1, 2).contiguous()
        ).max(-1)[0]
        if return_point_features:
            return global_features, point_features
        return global_features
