import math
import numpy as np
import torch


def pc_normalize(pc):
    mean = np.mean(pc, axis=0)
    pc -= mean
    m = np.max(np.sqrt(np.sum(np.power(pc, 2), axis=1)))
    pc /= m
    return pc


def random_select_points(pc, m):
    if m < 0:
        idx = np.arange(pc.shape[0])
        np.random.shuffle(idx)
        return pc[idx, :]
    n = pc.shape[0]
    replace = False if n >= m else True
    idx = np.random.choice(n, size=(m, ), replace=replace)
    return pc[idx, :]


def generate_rotation_x_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[1, 1] = math.cos(theta)
    mat[1, 2] = -math.sin(theta)
    mat[2, 1] = math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_y_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 2] = math.sin(theta)
    mat[2, 0] = -math.sin(theta)
    mat[2, 2] = math.cos(theta)
    return mat


def generate_rotation_z_matrix(theta):
    mat = np.eye(3, dtype=np.float32)
    mat[0, 0] = math.cos(theta)
    mat[0, 1] = -math.sin(theta)
    mat[1, 0] = math.sin(theta)
    mat[1, 1] = math.cos(theta)
    return mat


def generate_random_rotation_matrix(angle1=-45, angle2=45):
    thetax, thetay, thetaz = np.random.uniform(angle1, angle2, size=(3,))
    matx = generate_rotation_x_matrix(thetax / 180 * math.pi)
    maty = generate_rotation_y_matrix(thetay / 180 * math.pi)
    matz = generate_rotation_z_matrix(thetaz / 180 * math.pi)
    return np.dot(matz, np.dot(maty, matx))


def generate_random_tranlation_vector(range1=-1, range2=1):
    tranlation_vector = np.random.uniform(range1, range2, size=(3, )).astype(np.float32)
    return tranlation_vector


def generate_random_scale(min_scale=0.9, max_scale=1.1):
    """Sample a positive isotropic scale uniformly in log space."""
    if min_scale <= 0 or max_scale < min_scale:
        raise ValueError('scales must satisfy 0 < min_scale <= max_scale')
    return np.float32(np.exp(np.random.uniform(np.log(min_scale),
                                               np.log(max_scale))))


def transform(pc, R, t=None):
    pc = np.dot(pc, R.T)
    if t is not None:
        pc = pc + t
    return pc


def similarity_transform(pc, R, t=None, scale=1.0):
    """Apply isotropic scale, rotation and translation to numpy points."""
    coords, normals, others = pc[:, :3], pc[:, 3:6], pc[:, 6:]
    t_coords = scale * np.dot(coords, R.T)

    t_normals = np.dot(normals, R.T) if normals.size > 0 else normals
    if t is not None:
        t_coords = t_coords + t
    result = np.concatenate([t_coords, t_normals, others], axis=-1)
    return result


def inverse_similarity_transform(pc, R, t, scale):
    if scale <= 0:
        raise ValueError('scale must be positive')
    coords, normals, others = pc[:, :3], pc[:, 3:6], pc[:, 6:]

    t_coords = np.dot(coords - t, R) / scale
    t_normals = np.dot(normals, R) if normals.size > 0 else normals
    t_res = np.concatenate([t_coords, t_normals, others], axis=-1)
    return t_res



def batch_transform(batch_pc, batch_R, batch_t=None):
    '''

    :param batch_pc: shape=(B, N, 3)
    :param batch_R: shape=(B, 3, 3)
    :param batch_t: shape=(B, 3)
    :return: shape(B, N, 3)
    '''
    transformed_pc = torch.matmul(batch_pc, batch_R.permute(0, 2, 1).contiguous())
    if batch_t is not None:
        transformed_pc = transformed_pc + torch.unsqueeze(batch_t, 1)
    return transformed_pc


def batch_similarity_transform(batch_pc, batch_R, batch_t=None,
                               batch_scale=None):
    """Apply batched ``p -> scale * R @ p + t`` to (B,N,3) points."""
    transformed = torch.matmul(batch_pc, batch_R.transpose(1, 2).contiguous())
    if batch_scale is not None:
        transformed = transformed * batch_scale.reshape(-1, 1, 1)
    if batch_t is not None:
        transformed = transformed + batch_t.unsqueeze(1)
    return transformed


def geometry_layout(in_dim):
    """Map a channel count to (coord_dim, has_normal).

    in_dim=2 -> (x, y); 3 -> (x, y, z); 4 -> xy + normal xy;
    6 -> xyz + normal xyz.
    """
    if in_dim not in (2, 3, 4, 6):
        raise ValueError('in_dim must be one of 2, 3, 4, 6')
    return (2 if in_dim in (2, 4) else 3), in_dim in (4, 6)


def project_geometry(pc, coord_dim, has_normal, source_dim=3):
    """Reduce a tensor laid out as coords(source_dim)/normals(source_dim)/others
    down to coord_dim (+coord_dim normals), dropping the trailing axes
    (e.g. z) -- the same coords/normals/others split as similarity_transform,
    just parameterized by the target coordinate dimensionality.
    """
    coords = pc[..., :source_dim]
    normals = pc[..., source_dim:2 * source_dim]
    others = pc[..., 2 * source_dim:]
    if has_normal and normals.shape[-1] < coord_dim:
        raise ValueError('has_normal requested but pc has no normal channels')
    parts = [coords[..., :coord_dim]]
    if has_normal:
        parts.append(normals[..., :coord_dim])
    if others.shape[-1]:
        parts.append(others)
    return torch.cat(parts, dim=-1)


def project_pose(rotation, translation, coord_dim):
    """Reduce a 3D similarity pose to its leading coord_dim block; exact
    when the transform only acts within the first coord_dim axes (e.g. a
    z=0 planar point cloud rotated only about z).
    """
    if rotation.shape[-1] == coord_dim:
        return rotation, translation
    return rotation[..., :coord_dim, :coord_dim].contiguous(), \
        translation[..., :coord_dim].contiguous()


def batch_angle2mat(batch_cos_sin):
    '''

    :param batch_cos_sin: shape=(B, 2), unnormalized (cos, sin)
    :return: shape=(B, 2, 2)
    '''
    norm = batch_cos_sin.norm(dim=1, keepdim=True).clamp_min(1e-8)
    cos, sin = (batch_cos_sin / norm).unbind(dim=1)
    R = torch.zeros(cos.shape[0], 2, 2, dtype=batch_cos_sin.dtype,
                    device=batch_cos_sin.device)
    R[:, 0, 0], R[:, 0, 1] = cos, -sin
    R[:, 1, 0], R[:, 1, 1] = sin, cos
    return R


def compose_similarity(rotation1, translation1, scale1,
                       rotation2, translation2, scale2):
    """Compose T2(T1(p)) for batched similarity transforms."""
    scale1 = scale1.reshape(-1)
    scale2 = scale2.reshape(-1)
    rotation = torch.matmul(rotation2, rotation1)
    translation = (
        scale2[:, None, None]
        * torch.matmul(rotation2, translation1.unsqueeze(-1))
    ).squeeze(-1) + translation2
    return rotation, translation, scale2 * scale1


def interpolate_control_offsets(points, controls, offsets, radius=None,
                                control_weights=None, neighbor_count=4,
                                eps=1e-8):
    """Interpolate sparse control offsets to all source points."""
    if controls.shape[:2] != offsets.shape[:2]:
        raise ValueError('controls and offsets must share (B,K)')
    count = min(neighbor_count, controls.shape[1])
    distances = torch.cdist(points, controls) ** 2
    nearest_distance, nearest_index = distances.topk(
        count, dim=-1, largest=False
    )
    nearest_offsets = torch.gather(
        offsets.unsqueeze(1).expand(-1, points.shape[1], -1, -1), 2,
        nearest_index.unsqueeze(-1).expand(-1, -1, -1, 3),
    )
    if radius is None:
        weights = (nearest_distance + eps).rsqrt()
    else:
        if not torch.is_tensor(radius):
            radius = points.new_tensor(radius)
        if radius.dim() == 0:
            nearest_radius = radius
        else:
            radius = radius.reshape(radius.shape[0], -1)
            nearest_radius = torch.gather(
                radius.unsqueeze(1).expand(-1, points.shape[1], -1), 2,
                nearest_index,
            )
        weights = torch.exp(
            -nearest_distance / (2.0 * nearest_radius ** 2 + eps)
        )
    if control_weights is not None:
        control_weights = control_weights.reshape(control_weights.shape[0], -1)
        nearest_confidence = torch.gather(
            control_weights.unsqueeze(1).expand(-1, points.shape[1], -1), 2,
            nearest_index,
        )
        weights = weights * nearest_confidence.clamp_min(eps)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(eps)
    return (nearest_offsets * weights.unsqueeze(-1)).sum(2)


# The transformation between unit quaternion and rotation matrix is referenced to
# https://zhuanlan.zhihu.com/p/45404840

def quat2mat(quat):
    w, x, y, z = quat
    R = np.zeros((3, 3), dtype=np.float32)
    R[0][0] = 1 - 2*y*y - 2*z*z
    R[0][1] = 2*x*y - 2*z*w
    R[0][2] = 2*x*z + 2*y*w
    R[1][0] = 2*x*y + 2*z*w
    R[1][1] = 1 - 2*x*x - 2*z*z
    R[1][2] = 2*y*z - 2*x*w
    R[2][0] = 2*x*z - 2*y*w
    R[2][1] = 2*y*z + 2*x*w
    R[2][2] = 1 - 2*x*x - 2*y*y
    return R


def batch_quat2mat(batch_quat):
    '''

    :param batch_quat: shape=(B, 4)
    :return:
    '''
    w, x, y, z = batch_quat[:, 0], batch_quat[:, 1], batch_quat[:, 2], \
                 batch_quat[:, 3]
    device = batch_quat.device
    B = batch_quat.size()[0]
    R = torch.zeros(dtype=batch_quat.dtype, size=(B, 3, 3), device=device)
    R[:, 0, 0] = 1 - 2 * y * y - 2 * z * z
    R[:, 0, 1] = 2 * x * y - 2 * z * w
    R[:, 0, 2] = 2 * x * z + 2 * y * w
    R[:, 1, 0] = 2 * x * y + 2 * z * w
    R[:, 1, 1] = 1 - 2 * x * x - 2 * z * z
    R[:, 1, 2] = 2 * y * z - 2 * x * w
    R[:, 2, 0] = 2 * x * z - 2 * y * w
    R[:, 2, 1] = 2 * y * z + 2 * x * w
    R[:, 2, 2] = 1 - 2 * x * x - 2 * y * y
    return R


def mat2quat(mat):
    w = math.sqrt(mat[0, 0] + mat[1, 1] + mat[2, 2] + 1) / 2
    x = (mat[2, 1] - mat[1, 2]) / (4 * w)
    y = (mat[0, 2] - mat[2, 0]) / (4 * w)
    z = (mat[1, 0] - mat[0, 1]) / (4 * w)
    return w, x, y, z


def jitter_point_cloud(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(np.float32)
    jittered_data += pc
    return jittered_data


def shift_point_cloud(pc, shift_range=0.1):
    N, C = pc.shape
    shifts = np.random.uniform(-shift_range, shift_range, (1, C)).astype(np.float32)
    pc += shifts
    return pc


def random_scale_point_cloud(pc, scale_low=0.8, scale_high=1.25):
    scale = np.random.uniform(scale_low, scale_high, 1)
    pc *= scale
    return pc


def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    inv_t = - inv_R @ t[..., None]
    return inv_R, torch.squeeze(inv_t, -1)


def uniform_2_sphere(num: int = None):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def random_crop(pc, p_keep):
    rand_xyz = uniform_2_sphere()
    centroid = np.mean(pc[:, :3], axis=0)
    pc_centered = pc[:, :3] - centroid

    dist_from_plane = np.dot(pc_centered, rand_xyz)
    mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
    return pc[mask, :]

