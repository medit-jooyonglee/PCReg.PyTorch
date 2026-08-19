from .benchmark import (Benchmark, IterativeBenchmark, SimilarityBenchmark,
                        IterativeSimilarityBenchmark)
from .pointnet2 import PointNet2Encoder
from .deformation import (CoarseDeformationNet,
                          SimilarityDeformationRegistration)


def _missing_open3d(*args, **kwargs):
    raise ImportError('open3d is required for FGR/ICP registration')


try:
    from .fgr import fgr
    from .icp import icp
except ImportError:
    # Neural registration does not depend on Open3D.
    fgr = _missing_open3d
    icp = _missing_open3d
