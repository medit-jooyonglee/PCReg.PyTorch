import numpy as np
import os
import torch
from torch.utils.data import Dataset

from utils import readpcd
from utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, generate_random_scale, transform, \
    inverse_similarity_transform, similarity_transform


class CustomData(Dataset):
    def __init__(self, *, root='', npts=1024, train=True, estimate_scale=False,
                 min_scale=0.9, max_scale=1.1,
                 in_dim=3, **kwargs):
        super(CustomData, self).__init__()
        dirname = 'train_data' if train else 'val_data'
        path = os.path.join(root, dirname)
        self.train = train
        if os.path.exists(path):
            files = [os.path.join(path, item) for item in sorted(os.listdir(path))]
        else:
            files = []
        self.files = files
        self.npts = npts
        self.estimate_scale = estimate_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.in_dim = in_dim
        
    # def __l

    def __getitem__(self, item):
        if self.files:
            file = self.files[item]
            cloud = readpcd(file, rtype='npy')
            cloud = random_select_points(cloud, m=self.npts)
        else:
            n = np.random.randint(1000, 2000)
            cloud = np.random.rand(n, self.in_dim).astype(np.float32) * 2 - 1
            if self.in_dim >= 6:
                # (x, y, z) noise above is fine as raw positions, but a
                # *normal* needs unit length -- overwrite those columns
                # with proper unit vectors instead of raw uniform noise.
                normals = np.random.randn(n, 3).astype(np.float32)
                normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
                cloud[:, 3:6] = normals

        # pc_normalize/transform/inverse_similarity_transform all assume
        # (N, 3) coordinates, so positions and normals must be split out and
        # transformed separately: normals only rotate (no translation or
        # scale), positions get the full similarity transform.
        if self.in_dim >= 6:
            ref_cloud, ref_normals = cloud[:, :3].copy(), cloud[:, 3:6]
        else:
            ref_cloud, ref_normals = cloud, None
        ref_cloud = pc_normalize(ref_cloud)

        R, t = generate_random_rotation_matrix(-20, 20), \
               generate_random_tranlation_vector(-0.5, 0.5)
        if self.estimate_scale:
            # R, t and scale describe the transform the network must apply:
            # source -> reference. This avoids the inverse-label ambiguity in
            # the historical rigid augmentation path.
            scale = generate_random_scale(self.min_scale, self.max_scale)
            src_cloud = inverse_similarity_transform(ref_cloud, R, t, scale)
        else:
            src_cloud = transform(ref_cloud, R, t)

        if ref_normals is not None:
            src_normals = np.dot(ref_normals, R)
            ref_cloud = np.concatenate([ref_cloud, ref_normals], axis=-1)
            src_cloud = np.concatenate([src_cloud, src_normals], axis=-1)

        if self.train:
            if ref_normals is not None:
                ref_cloud, src_cloud = ref_cloud.copy(), src_cloud.copy()
                ref_cloud[:, :3] = jitter_point_cloud(ref_cloud[:, :3])
                src_cloud[:, :3] = jitter_point_cloud(src_cloud[:, :3])
            else:
                ref_cloud = jitter_point_cloud(ref_cloud)
                src_cloud = jitter_point_cloud(src_cloud)
        if self.estimate_scale:
            return ref_cloud, src_cloud, ref_cloud, R, t, scale
        return ref_cloud, src_cloud, ref_cloud, R, t

    def __len__(self):
        if len(self.files) == 0:
            # some noise data
            return 10
        else:
            return len(self.files)
        # return len(self.files)
