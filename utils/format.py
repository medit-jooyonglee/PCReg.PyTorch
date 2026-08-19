import random
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None


def _require_open3d():
    if o3d is None:
        raise ImportError('open3d is required for point-cloud file I/O')


def readpcd(path, rtype='pcd'):
    _require_open3d()
    assert rtype in ['pcd', 'npy']
    pcd = o3d.io.read_point_cloud(path)
    if rtype == 'pcd':
        return pcd
    npy = np.asarray(pcd.points).astype(np.float32)
    return npy


def npy2pcd(npy, ind=-1):
    _require_open3d()
    colors = [[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]]
    color = colors[ind] if ind < 3 else [random.random() for _ in range(3)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(npy)
    if ind >= 0:
        pcd.paint_uniform_color(color)
    return pcd


def pcd2npy(pcd):
    npy = np.asarray(pcd.points)
    return npy
