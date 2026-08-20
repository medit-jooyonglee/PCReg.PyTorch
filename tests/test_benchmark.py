import pytest
import torch

from pcregmodel.models import benchmark
from pcregmodel.utils import geometry_layout


@pytest.mark.parametrize("in_dim", [2, 3, 4, 6])
@pytest.mark.parametrize("transform_head", [-1, 0, 8])
@pytest.mark.parametrize("backbone", ['pointnet', 'dgcnn'])
def test_similarity_benchmark_forward(in_dim, transform_head, backbone):

    print('parameters:', in_dim, transform_head)
    xs  = torch.rand(2, in_dim, 10)  # Batch of 2 samples, each with 10 points
    ys = torch.rand(2, in_dim, 10*2)  # Batch of 2 samples, each with 10 points
    coord_dim, has_normal = geometry_layout(in_dim)

    def normalize_normals(pts):
        if not has_normal:
            return
        idx = list(range(coord_dim, 2 * coord_dim))
        normals = pts[:, idx]
        normals = normals / (normals.norm(dim=-1, keepdim=True).clamp_min(1e-8))
        pts[:, idx] = normals

    normalize_normals(xs)
    normalize_normals(ys)

    model = benchmark.SimilarityBenchmark(
        in_dim=in_dim,
        transform_head=transform_head,
        backbone=backbone)

    b_r, b_t, b_s, b_x = model(xs, ys)
    assert b_r.shape == (2, coord_dim, coord_dim)
    assert b_t.shape == (2, coord_dim)
    assert b_s.shape == (2,)
    if transform_head > 0:
        print(b_r)
        # init identity matrix
        # assert torch.allclose(b_r, torch.eye(coord_dim).expand_as(b_r), atol=1e-5)
        # assert torch.allclose(b_t, torch.zeros_like(b_t), atol=1e-5)
        # assert torch.allclose(b_s, torch.ones_like(b_s), atol=1e-5)
        # assert torch.allclose(xs.transpose(1, 2), b_x, atol=1e-5)

if __name__ == "__main__":
    pytest.main([
        '-s',
        '-rGA',
        __file__])
    