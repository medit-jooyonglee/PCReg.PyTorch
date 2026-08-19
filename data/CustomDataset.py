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
    def __init__(self, root, npts, train=True, estimate_scale=False,
                 min_scale=0.9, max_scale=1.1, **kwargs):
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
        
        
    # def __l

    def __getitem__(self, item):
        if self.files:
            file = self.files[item]
            ref_cloud = readpcd(file, rtype='npy')
            ref_cloud = random_select_points(ref_cloud, m=self.npts)
        else:
            # ref_cloud = np.random.rand(self.npts, 3).astype(np.float32)
            ref_cloud = np.random.rand(np.random.randint(1000, 2000), 3).astype(np.float32) * 2 - 1
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
        if self.train:
            ref_cloud = jitter_point_cloud(ref_cloud)
            src_cloud = jitter_point_cloud(src_cloud)
        if self.estimate_scale:
            return ref_cloud, src_cloud, R, t, scale
        return ref_cloud, src_cloud, R, t

    def __len__(self):
        if len(self.files) == 0:
            # some noise data
            return 10
        else:
            return len(self.files)
        # return len(self.files)
