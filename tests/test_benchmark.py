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

@pytest.mark.parametrize("in_dim", [2, 3])
@pytest.mark.parametrize("num_points", [3, 10, 30])
def test_similarity_benchmark_local_pooling_forward(in_dim, num_points):
    coord_dim, _ = geometry_layout(in_dim)
    xs = torch.rand(2, in_dim, num_points)
    ys = torch.rand(2, in_dim, num_points * 2)

    model = benchmark.SimilarityBenchmark(
        in_dim=in_dim, backbone='pointnet2', pooling='local', num_local_points=8,
    )

    b_r, b_t, b_s, b_x = model(xs, ys)
    assert b_r.shape == (2, coord_dim, coord_dim)
    assert b_t.shape == (2, coord_dim)
    assert b_s.shape == (2,)
    assert b_x.shape == (2, num_points, coord_dim)


def test_similarity_benchmark_local_pooling_requires_pointnet2():
    with pytest.raises(ValueError):
        benchmark.SimilarityBenchmark(in_dim=3, backbone='pointnet', pooling='local')
    with pytest.raises(ValueError):
        benchmark.SimilarityBenchmark(in_dim=3, backbone='dgcnn', pooling='local')


def test_similarity_benchmark_invalid_pooling_raises():
    with pytest.raises(ValueError):
        benchmark.SimilarityBenchmark(in_dim=3, backbone='pointnet2', pooling='nope')


def test_similarity_benchmark_local_pooling_backward():
    model = benchmark.SimilarityBenchmark(
        in_dim=3, backbone='pointnet2', pooling='local', num_local_points=4,
    )
    xs = torch.rand(2, 3, 20)
    ys = torch.rand(2, 3, 20)

    b_r, b_t, b_s, b_x = model(xs, ys)
    loss = b_r.sum() + b_t.sum() + b_s.sum() + b_x.sum()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("pooling", ['global', 'local'])
@pytest.mark.parametrize("backbone", ['pointnet', 'pointnet2', 'dgcnn'])
def test_similarity_benchmark_early_fusion_forward(backbone, pooling):
    if pooling == 'local' and backbone != 'pointnet2':
        pytest.skip("pooling='local' requires backbone='pointnet2'")

    xs = torch.rand(2, 3, 20)
    ys = torch.rand(2, 3, 30)  # different point counts on each side

    model = benchmark.SimilarityBenchmark(
        in_dim=3, backbone=backbone, fusion='early', pooling=pooling,
    )
    b_r, b_t, b_s, b_x = model(xs, ys)
    assert b_r.shape == (2, 3, 3)
    assert b_t.shape == (2, 3)
    assert b_s.shape == (2,)
    assert b_x.shape == (2, 20, 3)


def test_similarity_benchmark_early_fusion_backward():
    model = benchmark.SimilarityBenchmark(in_dim=3, backbone='pointnet2', fusion='early')
    xs = torch.rand(2, 3, 15)
    ys = torch.rand(2, 3, 15)

    b_r, b_t, b_s, b_x = model(xs, ys)
    loss = b_r.sum() + b_t.sum() + b_s.sum() + b_x.sum()
    loss.backward()

    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_similarity_benchmark_invalid_fusion_raises():
    with pytest.raises(ValueError):
        benchmark.SimilarityBenchmark(in_dim=3, fusion='nope')


if __name__ == "__main__":
    pytest.main([
        '-s',
        '-rGA',
        __file__])
    