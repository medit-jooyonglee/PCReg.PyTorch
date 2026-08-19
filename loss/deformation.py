"""Robust losses for sparse-control point-cloud deformation."""

import torch


def _trimmed_mean(values, trim_ratio):
    if not 0.0 < trim_ratio <= 1.0:
        raise ValueError('trim_ratio must be in (0, 1]')
    keep_count = max(1, int(values.shape[-1] * trim_ratio))
    return values.topk(keep_count, dim=-1, largest=False)[0].mean()


def trimmed_chamfer_loss(source, target, trim_ratio=0.9, symmetric=True):
    """Chamfer loss robust to missing and non-corresponding regions."""
    distances = torch.cdist(source, target) ** 2
    loss = _trimmed_mean(distances.min(-1)[0], trim_ratio)
    if symmetric:
        reverse = _trimmed_mean(distances.min(-2)[0], trim_ratio)
        loss = 0.5 * (loss + reverse)
    return loss


def control_smoothness_loss(controls, offsets, neighbor_count=6):
    if controls.shape[1] < 2:
        return offsets.sum() * 0.0
    count = min(neighbor_count + 1, controls.shape[1])
    indices = torch.cdist(controls, controls).topk(
        count, dim=-1, largest=False
    )[1][..., 1:]
    neighbor_offsets = torch.gather(
        offsets.unsqueeze(1).expand(-1, controls.shape[1], -1, -1), 2,
        indices.unsqueeze(-1).expand(-1, -1, -1, 3),
    )
    return ((offsets.unsqueeze(2) - neighbor_offsets) ** 2).mean()


def coarse_deformation_loss(prediction, target_points, trim_ratio=0.9,
                            data_weight=1.0, smoothness_weight=0.1,
                            magnitude_weight=0.01,
                            confidence_weight=0.001):
    """Return total and component losses for CoarseDeformationNet."""
    data = trimmed_chamfer_loss(
        prediction['warped_source'][..., :3], target_points[..., :3],
        trim_ratio=trim_ratio,
    )
    smoothness = control_smoothness_loss(
        prediction['controls'], prediction['control_offsets']
    )
    magnitude = (prediction['control_offsets'] ** 2).mean()
    confidence = (1.0 - prediction['control_confidence']).mean()
    total = (data_weight * data + smoothness_weight * smoothness
             + magnitude_weight * magnitude
             + confidence_weight * confidence)
    return {
        'loss': total,
        'data_loss': data,
        'smoothness_loss': smoothness,
        'magnitude_loss': magnitude,
        'confidence_loss': confidence,
    }
