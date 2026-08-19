import os
import torch
import torch.nn as nn
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOR_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOR_DIR)
try:
    from ..utils import (batch_quat2mat, batch_transform,
                         batch_similarity_transform, compose_similarity)
except ImportError:
    from utils import (batch_quat2mat, batch_transform,
                       batch_similarity_transform, compose_similarity)
from .pointnet2 import PointNet2Encoder


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
        encoder = PointNet2Encoder(
            in_dim=in_dim,
            sample_count1=kwargs.get('sample_count1', 256),
            sample_count2=kwargs.get('sample_count2', 64),
            global_dim=kwargs.get('global_dim', 1024),
        )
        return encoder, encoder.global_dim
    raise ValueError("backbone must be 'pointnet' or 'pointnet2'")


class Benchmark(nn.Module):
    def __init__(self, gn, in_dim1, in_dim2=2048,
                 fcs=(1024, 1024, 512, 512, 256, 7),
                 backbone='pointnet', **kwargs):
        super(Benchmark, self).__init__()
        self.in_dim1 = in_dim1
        self.encoder, encoder_dim = _make_encoder(backbone, in_dim1, gn,
                                                  **kwargs)
        in_dim2 = encoder_dim * 2
        self.decoder = nn.Sequential()
        for i, out_dim in enumerate(fcs):
            self.decoder.add_module(f'fc_{i}', nn.Linear(in_dim2, out_dim))
            if out_dim != 7:
                if gn:
                    self.decoder.add_module(f'gn_{i}',nn.GroupNorm(8, out_dim))
                self.decoder.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            in_dim2 = out_dim

    def forward(self, x, y):
        x_f, y_f = self.encoder(x), self.encoder(y)
        concat = torch.cat((x_f, y_f), dim=1)
        out = self.decoder(concat)
        batch_t, batch_quat = out[:, :3], out[:, 3:] / torch.norm(out[:, 3:], dim=1, keepdim=True)
        batch_R = batch_quat2mat(batch_quat)
        if self.in_dim1 == 3:
            transformed_x = batch_transform(x.permute(0, 2, 1).contiguous(),
                                            batch_R, batch_t)
        elif self.in_dim1 == 6:
            transformed_pts = batch_transform(x.permute(0, 2, 1)[:, :, :3].contiguous(),
                                            batch_R, batch_t)
            transformed_nls = batch_transform(x.permute(0, 2, 1)[:, :, 3:].contiguous(),
                                              batch_R)
            transformed_x = torch.cat([transformed_pts, transformed_nls], dim=-1)
        else:
            raise ValueError
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
        transformed_x = torch.clone(x)
        batch_R_res = torch.eye(3).to(device).unsqueeze(0).repeat(B, 1, 1)
        batch_t_res = torch.zeros(3, 1).to(device).unsqueeze(0).repeat(B, 1, 1)
        for i in range(self.niters):
            batch_R, batch_t, transformed_x = self.benckmark(transformed_x, y)
            transformed_xs.append(transformed_x)
            batch_R_res = torch.matmul(batch_R, batch_R_res)
            batch_t_res = torch.matmul(batch_R, batch_t_res) \
                          + torch.unsqueeze(batch_t, -1)
            transformed_x = transformed_x.permute(0, 2, 1).contiguous()
        batch_t_res = torch.squeeze(batch_t_res, dim=-1)
        #transformed_x = transformed_x.permute(0, 2, 1).contiguous()
        return batch_R_res, batch_t_res, transformed_xs


class SimilarityBenchmark(nn.Module):
    """Predict source-to-target rotation, translation and isotropic scale."""

    def __init__(self, in_dim, gn=False, backbone='pointnet2',
                 fcs=(1024, 512, 256, 8), max_log_scale=0.35, **kwargs):
        super(SimilarityBenchmark, self).__init__()
        self.in_dim = in_dim
        self.geometry_dim = 6 if in_dim >= 6 else 3
        self.max_log_scale = float(max_log_scale)
        self.encoder, feature_dim = _make_encoder(backbone, in_dim, gn,
                                                  **kwargs)
        current = feature_dim * 2
        self.decoder = nn.Sequential()
        for i, output in enumerate(fcs):
            self.decoder.add_module(f'fc_{i}', nn.Linear(current, output))
            if output != 8:
                if gn:
                    groups = min(8, output)
                    while output % groups:
                        groups -= 1
                    self.decoder.add_module(f'gn_{i}', nn.GroupNorm(groups, output))
                self.decoder.add_module(f'relu_{i}', nn.ReLU(inplace=True))
            current = output

    def forward(self, source, target):
        source_features = self.encoder(source)
        target_features = self.encoder(target)
        output = self.decoder(torch.cat((source_features,
                                         target_features), dim=1))
        translation = output[:, :3]
        quaternion = output[:, 3:7]
        quaternion = quaternion / quaternion.norm(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        rotation = batch_quat2mat(quaternion)
        scale = torch.exp(self.max_log_scale * torch.tanh(output[:, 7]))

        points = source[:, :3].transpose(1, 2).contiguous()
        transformed = [batch_similarity_transform(
            points, rotation, translation, scale
        )]
        if self.geometry_dim == 6:
            normals = batch_transform(
                source[:, 3:6].transpose(1, 2).contiguous(), rotation
            )
            transformed.append(normals)
        if self.in_dim > self.geometry_dim:
            transformed.append(
                source[:, self.geometry_dim:].transpose(1, 2).contiguous()
            )
        return rotation, translation, scale, torch.cat(transformed, dim=-1)


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
        transformed = source
        rotation = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        rotation = rotation.repeat(batch_size, 1, 1)
        translation = torch.zeros(batch_size, 3, device=device, dtype=dtype)
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
    x, y = torch.randn(4, 3, 5), torch.randn(4, 3, 5)
    net = IterativeBenchmark(in_dim1=3, niters=2)
    print(net)
    batch_R, batch_t, transformed_x = net(x, y)
