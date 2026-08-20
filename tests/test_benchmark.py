import pytest
import torch

from pcregmodel.models import benchmark


# def test_similarity_benchmark_forward(in_dim, transform_head):
    
#     xs  = torch.rand(2, in_dim, 10)  # Batch of 2 samples, each with 10 points
#     ys = torch.rand(2, in_dim, 10*2)  # Batch of 2 samples, each with 10 points
    
#     model = benchmark.SimilarityBenchmark(
#         in_dim=in_dim,
#         transform_head=transform_head)
#     # res = 
    
#     b_r, b_t, b_x = model(xs, ys)
#     assert b_r.shape == (2, 3, 3)



@pytest.mark.parametrize("in_dim", [3, 6, 9])
@pytest.mark.parametrize("transform_head", [-1, 0, 8])
def test_similarity_benchmark_forward(in_dim, transform_head):
    
    print('parameters:', in_dim, transform_head)
    xs  = torch.rand(2, in_dim, 10)  # Batch of 2 samples, each with 10 points
    ys = torch.rand(2, in_dim, 10*2)  # Batch of 2 samples, each with 10 points
    
    def normalize_nxnynz(pts):
        idx = [3, 4, 5]
        if pts.shape[1] >= 6:
            normals = pts[:, idx]
            normals = normals / (normals.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            pts[:, idx] = normals
    
    # normalize_nxnynz(xs.transpose(1, 2))
    normalize_nxnynz(xs)
    normalize_nxnynz(ys)
    
    
            
        
    
    model = benchmark.SimilarityBenchmark(
        in_dim=in_dim,
        transform_head=transform_head)
    # res = 
    
    b_r, b_t, b_s, b_x = model(xs, ys)
    assert b_r.shape == (2, 3, 3)
    assert b_t.shape == (2, 3)
    assert b_s.shape == (2,)
    if transform_head > 0:
        print(b_r)
        # init identity matrix
        assert torch.allclose(b_r, torch.eye(3).expand_as(b_r), atol=1e-5)
        assert torch.allclose(b_t, torch.zeros_like(b_t), atol=1e-5)
        assert torch.allclose(b_s, torch.ones_like(b_s), atol=1e-5)
        assert torch.allclose(xs.transpose(1, 2), b_x, atol=1e-5)
    else:
        pass
        
    
    # if simlarity
    if in_dim == 6:
        pass
        

    
if __name__ == "__main__":
    pytest.main([
        '-s',
        '-rGA',
        __file__])
    