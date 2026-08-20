import os
import sys

import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
try:
    from ..utils import (batch_quat2mat, batch_angle2mat, batch_transform,
                         batch_similarity_transform, compose_similarity,
                         geometry_layout)
except ImportError:
    from utils import (batch_quat2mat, batch_angle2mat, batch_transform,
                       batch_similarity_transform, compose_similarity,
                       geometry_layout)
from .pointnet2 import PointNet2Encoder
from .dgcnn import DGCNNEncoder


class PointNet(nn.Module):
    def __init__(self, in_dim, gn, mlps=[64, 64, 64, 128, 1024]):
        super(PointNet, self).__init__()
        self.backbone = nn.Sequential()
        for i, out_dim in enumerate(mlps):
            self.backbone.add_module(f'pointnet_conv_{i}',
                                     nn.Conv1d(in_dim, out_dim, 1, 1, 0))
            if gn:
                self.backbone.add_module(f'pointnet_gn_{i}',
                                    nn.GroupNorm(8, out_dim))
            self.backbone.add_module(f'pointnet_relu_{i}',
                                     nn.ReLU(inplace=True))
            in_dim = out_dim

    def forward(self, x):
        x = self.backbone(x)
        x, _ = torch.max(x, dim=2)
        return x


def _make_encoder(backbone, in_dim, gn=False, **kwargs):
    if backbone == 'pointnet':
        return PointNet(in_dim=in_dim, gn=gn), 1024
    if backbone == 'pointnet2':
        coord_dim, _ = geometry_layout(in_dim)
        encoder = PointNet2Encoder(
            in_dim=in_dim,
            sample_count1=kwargs.get('sample_count1', 256),
            sample_count2=kwargs.get('sample_count2', 64),
            global_dim=kwargs.get('global_dim', 1024),
            coord_dim=coord_dim,
        )
        return encoder, encoder.global_dim
    if backbone == 'dgcnn':
        encoder = DGCNNEncoder(
            in_dim=in_dim,
            k=kwargs.get('k', 20),
            mlps=kwargs.get('dgcnn_mlps', (64, 64, 128, 256)),
            global_dim=kwargs.get('global_dim', 1024),
            gn=gn,
        )
        return encoder, encoder.global_dim
    raise ValueError("backbone must be 'pointnet', 'pointnet2' or 'dgcnn'")


def _rotation_dim(coord_dim):
    """Free parameters of the rotation representation: quaternion (3D) or
    (cos, sin) (2D)."""
    return 4 if coord_dim == 3 else 2


def _decode_pose(output, coord_dim, max_log_scale=None):
    """Split a decoder output into (rotation, translation, scale), sharing
    one code path for the 3D quaternion and the 2D (cos, sin) case."""
    translation = output[:, :coord_dim]
    rot_end = coord_dim + _rotation_dim(coord_dim)
    rot_params = output[:, coord_dim:rot_end]
    if coord_dim == 3:
        quat = rot_params / rot_params.norm(dim=1, keepdim=True).clamp_min(1e-8)
        rotation = batch_quat2mat(quat)
    else:
        rotation = batch_angle2mat(rot_params)
    scale = None
    if max_log_scale is not None:
        scale = torch.exp(max_log_scale * torch.tanh(output[:, rot_end]))
    return rotation, translation, scale


def _transform_point_cloud(x, rotation, translation, scale, coord_dim, has_normal):
    """x: (B, C, N) with C = coord_dim (*2 if has_normal). Returns (B, N, C)."""
    x = x.permute(0, 2, 1).contiguous()
    span = coord_dim * (2 if has_normal else 1)
    transformed = [batch_similarity_transform(
        x[..., :coord_dim], rotation, translation, scale
    )]
    if has_normal:
        transformed.append(batch_transform(x[..., coord_dim:span], rotation))
    if x.shape[-1] > span:
        transformed.append(x[..., span:])
    return torch.cat(transformed, dim=-1)


class Benchmark(nn.Module):
    def __init__(self, gn, in_dim1, fcs=(1024, 1024, 512, 512, 256),
                 backbone='pointnet', **kwargs):
        super(Benchmark, self).__init__()
        self.in_dim1 = in_dim1
        self.coord_dim, self.has_normal = geometry_layout(in_dim1)
        self.encoder, encoder_dim = _make_encoder(backbone, in_dim1, gn,
                                                  **kwargs)
        pose_dim = self.coord_dim + _rotation_dim(self.coord_dim)
        current = encoder_dim * 2
        self.decoder = nn.Sequential()
        for i, out_dim in enumerate(list(fcs) + [pose_dim]):
            self.decoder.add_module(f'fc_{i}', nn.Linear(current, out_dim))
            if out_dim != pose_dim:
                if gn:
                    self.decoder.add_module(f'gn_{i}', nn.GroupNorm(8, out_dim))
                self.decoder.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            current = out_dim

    def forward(self, x, y):
        x_f, y_f = self.encoder(x), self.encoder(y)
        concat = torch.cat((x_f, y_f), dim=1)
        out = self.decoder(concat)
        batch_R, batch_t, _ = _decode_pose(out, self.coord_dim)
        transformed_x = _transform_point_cloud(
            x, batch_R, batch_t, None, self.coord_dim, self.has_normal
        )
        return batch_R, batch_t, transformed_x


class IterativeBenchmark(nn.Module):
    def __init__(self, in_dim, niters, gn, backbone='pointnet', **kwargs):
        super(IterativeBenchmark, self).__init__()
        self.benckmark = Benchmark(gn=gn, in_dim1=in_dim,
                                  backbone=backbone, **kwargs)
        self.niters = niters

    def forward(self, x, y):
        transformed_xs = []
        device = x.device
        B = x.size()[0]
        coord_dim = self.benckmark.coord_dim
        transformed_x = torch.clone(x)
        batch_R_res = torch.eye(coord_dim).to(device).unsqueeze(0).repeat(B, 1, 1)
        batch_t_res = torch.zeros(coord_dim, 1).to(device).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.niters):
            batch_R, batch_t, transformed_x = self.benckmark(transformed_x, y)
            transformed_xs.append(transformed_x)
            batch_R_res = torch.matmul(batch_R, batch_R_res)
            batch_t_res = torch.matmul(batch_R, batch_t_res) \
                          + torch.unsqueeze(batch_t, -1)
            transformed_x = transformed_x.permute(0, 2, 1).contiguous()
        batch_t_res = torch.squeeze(batch_t_res, dim=-1)
        return batch_R_res, batch_t_res, transformed_xs


class SimilarityBenchmark(nn.Module):
    """Predict source-to-target rotation, translation and isotropic scale."""

    def __init__(self, in_dim=3, gn=False, backbone='pointnet2',
                 fcs=(1024, 512, 256), transform_head: int = -1,
                 max_log_scale=0.35, **kwargs):
        super(SimilarityBenchmark, self).__init__()
        self.in_dim = in_dim
        self.coord_dim, self.has_normal = geometry_layout(in_dim)
        self.max_log_scale = float(max_log_scale)
        self.encoder, feature_dim = _make_encoder(backbone, in_dim, gn,
                                                  **kwargs)
        pose_dim = self.coord_dim + _rotation_dim(self.coord_dim) + 1

        fcs = list(fcs)
        if transform_head <= 0:
            fcs = fcs + [pose_dim]
        current = feature_dim * 2
        self.decoder = nn.Sequential()
        for i, output in enumerate(fcs):
            self.decoder.add_module(f'fc_{i}', nn.Linear(current, output))
            if output != pose_dim:
                if gn:
                    groups = min(8, output)
                    while output % groups:
                        groups -= 1
                    self.decoder.add_module(f'gn_{i}', nn.GroupNorm(groups, output))
                self.decoder.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            current = output

        self.transform_head = None
        if transform_head > 0:
            self.transform_head = nn.Linear(current, pose_dim)
            nn.init.normal_(self.transform_head.weight, std=0.001)
            nn.init.normal_(self.transform_head.bias, std=0.001)
            with torch.no_grad():
                self.transform_head.bias[self.coord_dim] = 1.0

    def forward(self, source, target):
        source_features = self.encoder(source)
        target_features = self.encoder(target)
        decoded = self.decoder(torch.cat((source_features,
                                         target_features), dim=1))
        output = decoded if self.transform_head is None \
            else self.transform_head(decoded)
        rotation, translation, scale = _decode_pose(
            output, self.coord_dim, self.max_log_scale
        )
        transformed = _transform_point_cloud(
            source, rotation, translation, scale, self.coord_dim, self.has_normal
        )
        return rotation, translation, scale, transformed


class IterativeSimilarityBenchmark(nn.Module):
    """Iterative similarity registration using PointNet++ by default."""

    def __init__(self, in_dim, niters=4, gn=False, **kwargs):
        super(IterativeSimilarityBenchmark, self).__init__()
        if niters < 1:
            raise ValueError('niters must be positive')
        self.benchmark = SimilarityBenchmark(in_dim=in_dim, gn=gn, **kwargs)
        self.benckmark = self.benchmark
        self.niters = niters

    def forward(self, source, target):
        batch_size = source.shape[0]
        device, dtype = source.device, source.dtype
        coord_dim = self.benchmark.coord_dim
        transformed = source
        rotation = torch.eye(coord_dim, device=device, dtype=dtype).unsqueeze(0)
        rotation = rotation.repeat(batch_size, 1, 1)
        translation = torch.zeros(batch_size, coord_dim, device=device, dtype=dtype)
        scale = torch.ones(batch_size, device=device, dtype=dtype)
        transformed_sources = []
        for _ in range(self.niters):
            delta_rotation, delta_translation, delta_scale, transformed_nxc = \
                self.benchmark(transformed, target)
            rotation, translation, scale = compose_similarity(
                rotation, translation, scale,
                delta_rotation, delta_translation, delta_scale,
            )
            transformed_sources.append(transformed_nxc)
            transformed = transformed_nxc.transpose(1, 2).contiguous()
        # Keep the legacy IterativeBenchmark ordering: transforms first, clouds last.
        return rotation, translation, scale, transformed_sources


if __name__ == '__main__':
    in_dim = 4
    x, y = torch.randn(4, in_dim, 5), torch.randn(4, in_dim, 5)
    net = IterativeBenchmark(in_dim=3, niters=2, gn=False)
    print(net)
    batch_R, batch_t, transformed_x = net(x, y)
    print()