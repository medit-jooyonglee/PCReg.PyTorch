"""Sparse-control coarse deformation models for point-cloud registration."""

import math

import torch
import torch.nn as nn

from .pointnet2 import (PointNet2Encoder, farthest_point_sample, index_points,
                        square_distance)
try:
    from ..utils import interpolate_control_offsets
except ImportError:
    from utils import interpolate_control_offsets


def interpolate_features(query, points, features, neighbor_count=3):
    count = min(neighbor_count, points.shape[1])
    distances, indices = square_distance(query, points).topk(
        count, dim=-1, largest=False
    )
    neighbors = torch.gather(
        features.unsqueeze(1).expand(-1, query.shape[1], -1, -1), 2,
        indices.unsqueeze(-1).expand(-1, -1, -1, features.shape[-1]),
    )
    weights = (distances + 1e-8).rsqrt()
    weights = weights / weights.sum(-1, keepdim=True)
    return (neighbors * weights.unsqueeze(-1)).sum(2)


class CoarseDeformationNet(nn.Module):
    """Predict a sparse source-to-target deformation field.

    ``control_mode`` supports source FPS seeds (``input_seed``), a fixed
    bounding-box grid (``fixed_grid``), and multi-radius FPS controls
    (``anchors``). Source/target may have different point counts.
    """

    def __init__(self, in_dim=3, control_mode='input_seed', num_controls=64,
                 grid_size=4, anchor_scales=(0.08, 0.16, 0.32),
                 max_displacement=0.15, interpolation_neighbors=4,
                 sample_count1=256, sample_count2=64, global_dim=512):
        super(CoarseDeformationNet, self).__init__()
        if control_mode not in ('input_seed', 'fixed_grid', 'anchors'):
            raise ValueError('invalid control_mode: {}'.format(control_mode))
        if num_controls < 1 or grid_size < 2:
            raise ValueError('num_controls must be positive and grid_size >= 2')
        self.in_dim = in_dim
        self.control_mode = control_mode
        self.num_controls = int(num_controls)
        self.grid_size = int(grid_size)
        self.anchor_scales = tuple(float(value) for value in anchor_scales)
        self.max_displacement = float(max_displacement)
        self.interpolation_neighbors = int(interpolation_neighbors)
        self.encoder = PointNet2Encoder(
            in_dim, sample_count1=sample_count1,
            sample_count2=sample_count2, global_dim=global_dim,
        )
        local_dim = self.encoder.point_dim
        predictor_dim = local_dim * 3 + global_dim * 2 + 4
        self.field_head = nn.Sequential(
            nn.Conv1d(predictor_dim, 512, 1), nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1), nn.ReLU(inplace=True),
            nn.Conv1d(256, 4, 1),
        )
        nn.init.zeros_(self.field_head[-1].weight)
        nn.init.zeros_(self.field_head[-1].bias)

    def make_controls(self, points):
        batch_size = points.shape[0]
        minimum = points.min(1)[0]
        extent = points.max(1)[0] - minimum
        diagonal = extent.norm(dim=-1).clamp_min(1e-6)
        if self.control_mode == 'fixed_grid':
            axis = torch.linspace(0.0, 1.0, self.grid_size,
                                  device=points.device, dtype=points.dtype)
            unit_grid = torch.stack(torch.meshgrid(axis, axis, axis), -1)
            unit_grid = unit_grid.reshape(-1, 3)
            controls = minimum[:, None] + unit_grid[None] * extent[:, None]
            radius = diagonal[:, None].expand(-1, controls.shape[1])
            radius = radius * (1.5 / self.grid_size)
            return controls, radius

        if self.control_mode == 'input_seed':
            indices = farthest_point_sample(points, self.num_controls)
            controls = index_points(points, indices)
            radius = diagonal[:, None].expand(-1, controls.shape[1])
            radius = radius / math.sqrt(self.num_controls)
            return controls, radius

        scale_count = len(self.anchor_scales)
        if scale_count == 0:
            raise ValueError('anchor_scales cannot be empty')
        center_count = int(math.ceil(float(self.num_controls) / scale_count))
        centers = index_points(
            points, farthest_point_sample(points, center_count)
        )
        controls = centers.unsqueeze(2).expand(
            -1, -1, scale_count, -1
        ).reshape(batch_size, -1, 3)
        scale_values = points.new_tensor(self.anchor_scales)
        radius = (diagonal[:, None, None] * scale_values[None, None])
        radius = radius.expand(-1, center_count, -1).reshape(batch_size, -1)
        return controls[:, :self.num_controls], radius[:, :self.num_controls]

    def forward(self, source, target):
        if source.shape[1] != self.in_dim or target.shape[1] != self.in_dim:
            raise ValueError('source and target channels must equal in_dim')
        source_global, source_local = self.encoder(
            source, return_point_features=True
        )
        target_global, target_local = self.encoder(
            target, return_point_features=True
        )
        source_points = source[:, :3].transpose(1, 2).contiguous()
        target_points = target[:, :3].transpose(1, 2).contiguous()
        controls, radius = self.make_controls(source_points)

        source_features = interpolate_features(
            controls, source_points, source_local
        )
        target_features = interpolate_features(
            controls, target_points, target_local
        )
        centroid = source_points.mean(1, keepdim=True)
        relative = controls - centroid
        global_features = torch.cat((source_global, target_global), dim=1)
        global_features = global_features[:, None].expand(
            -1, controls.shape[1], -1
        )
        field_features = torch.cat((
            source_features, target_features,
            target_features - source_features, global_features,
            relative, radius.unsqueeze(-1),
        ), dim=-1)
        prediction = self.field_head(
            field_features.transpose(1, 2).contiguous()
        ).transpose(1, 2).contiguous()
        control_offsets = self.max_displacement * torch.tanh(
            prediction[..., :3]
        )
        confidence = torch.sigmoid(prediction[..., 3])
        dense_offsets = interpolate_control_offsets(
            source_points, controls, control_offsets, radius=radius,
            control_weights=confidence,
            neighbor_count=self.interpolation_neighbors,
        )
        warped_points = source_points + dense_offsets
        if self.in_dim > 3:
            warped_source = torch.cat((
                warped_points, source[:, 3:].transpose(1, 2).contiguous()
            ), dim=-1)
        else:
            warped_source = warped_points
        return {
            'warped_source': warped_source,
            'dense_offsets': dense_offsets,
            'controls': controls,
            'control_offsets': control_offsets,
            'control_confidence': confidence,
            'control_radius': radius,
        }


class SimilarityDeformationRegistration(nn.Module):
    """Similarity alignment followed by coarse non-rigid registration."""

    def __init__(self, similarity, deformation):
        super(SimilarityDeformationRegistration, self).__init__()
        self.similarity = similarity
        self.deformation = deformation

    def forward(self, source, target):
        rotation, translation, scale, transformed_sources = self.similarity(
            source, target
        )
        similarity_source = transformed_sources[-1]
        field = self.deformation(
            similarity_source.transpose(1, 2).contiguous(), target
        )
        field.update({
            'rotation': rotation,
            'translation': translation,
            'scale': scale,
            'similarity_source': similarity_source,
            'similarity_sources': transformed_sources,
        })
        return field
