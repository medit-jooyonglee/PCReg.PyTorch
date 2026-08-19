# import torch.utils.data import Data
import torch
import tqdm
# import math
import glob
import json
import os
import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
# from shapely import transform
from torch.utils.data.dataloader import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Union, Literal
from trainer import diskmanager, get_logger, vtk_utils, timefn, image_utils
from trainer import vtk_utils, geometry_numpy, get_logger, time_strftime, utils_numpy, torch_utils
from trainer.image_utils import cv2_imread, cv2_imwrite
# from reversereg.preproc.sampler import cv2_imread, cv2_imwrite, to_rgba, blend_images
import torchvision
# from rfdetr.datasets.coco import CocoDetection
from PIL import Image

from reversereg.preproc import sampler

from pcregmodel.utils import pc_normalize, random_select_points, shift_point_cloud, \
    jitter_point_cloud, generate_random_rotation_matrix, \
    generate_random_tranlation_vector, generate_random_scale, transform, \
    inverse_similarity_transform, similarity_transform



def sampling_points(points, pose):
    # center = points.mean(axis=0)
    vmax, vmin = points.max(axis=0), points.min(axis=0)
    ratio = 2/3 if pose == 'upper' else 1/3
    threshold = vmin[1] + (vmax[1] - vmin[1]) * ratio
    
    if pose == 'upper':
        return points[points[:, 1] > threshold]
    elif pose == 'lower':
        return points[points[:, 1] < threshold]
    else:
        raise ValueError(f"Invalid pose: {pose}. Must be 'upper' or 'lower'.")
    
def sampling_points_and_polygons(uniform_polygons, sample_args, pose):
    
    this_polys = [sampling_points(uniform_polygons[i], pose) for i in sample_args]
    # sampling
    this_archs = []
    for p in this_polys:
        # y0 = p
        i = np.argmax(p[:, 1])
        this_archs.append(p[i])
        
    this_archs = np.array(this_archs)
    return this_polys, this_archs
        

def coco_directory_structure_check(coco_dataset:torchvision.datasets.CocoDetection):
    coco = coco_dataset.coco
    for image_info in coco.dataset.get("images", []):
        file_name = image_info.get("file_name")

        if isinstance(file_name, str):
            image_info["file_name"] = file_name.replace("\\", "/")
    coco.createIndex()
    
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, *,
                 img_folder='', ann_file='', include_masks=True,
                 train=True,
                 npts=256,
                 estimate_scale=True,
                 min_scale=0.9,
                 max_scale=1.1,
                 **kwargs):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        # internal-splits
        coco_directory_structure_check(self)
        self.train = train
        self.split = 0.8 if train else 0.2
        # self._transforms = transforms
        self.include_masks = include_masks
        self.prepare = ConvertCoco(include_masks=include_masks, npts=npts,
                                   estimate_scale=estimate_scale,
                                   min_scale=min_scale, max_scale=max_scale)
        
    def __len__(self):
        total_len = len(self.ids) * 2
        return int(total_len * self.split)
        # for debugging
        # return 10

    def __getitem__(self, idx):
        while True:
            try:
                res = self.item(idx)
                return res
            except Exception as e:
                print(f"Error processing index {idx}: {e}")
                idx = (idx + 1) % len(self.ids)  # Move to the next index
                
    def item(self, in_idx0):
        total_len = len(self.ids) * 2
        idx0 = in_idx0 if self.train else in_idx0 + int(total_len * (1 - self.split))
        
        pose = 'upper' if idx0 % 2 == 0 else 'lower'
        idx = idx0 // 2
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        # try:
        return self.prepare(img, target, pose=pose)
        # except Exception as e:
            


class ConvertCoco(object):

    def __init__(self, include_masks=False, npts=256, estimate_scale=True,
                 min_scale=0.9, max_scale=1.1):
        self.include_masks = include_masks
        self.npts = npts
        self.estimate_scale = estimate_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

    # def 
    def __call__(self, image, target, pose='upper'):
        w, h = image.size

        image_id = target["image_id"]
        image_id = [image_id]

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        # guard against no boxes via resizing
        # (x, y, w, h) -> (x1, y1, x2, y2)
        # boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], a_min=0, a_max=w)
        
        # boxes[:, 1::2].clamp_(min=0, max=h)

        classes = np.array([obj["category_id"] for obj in anno])
        # classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        uniform_sampling_points_dist = 1.5
        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                # resampling curved
                assert [len(seg) == 1 for seg in segmentations], "Only single polygon segmentation is supported."
                polygons = [np.array(polygon[0]).reshape(-1, 2) for polygon in segmentations]
                # closed curve
                uniform_polygons = []
                for poly in polygons:

                    poly = geometry_numpy.uniform_resampling(
                        np.concatenate([poly, poly[:1]]), 
                        interval=uniform_sampling_points_dist,
                        iteration=5,
                    )
                    uniform_polygons.append(poly.astype(np.float32))
                    
            else:
                raise ValueError("No segmentation found in annotations.")


            target["polygons"] = uniform_polygons
        pc_normalize = True
        if pc_normalize:
            max_len = np.max([w, h])
            for i, pts in enumerate(uniform_polygons):
                uniform_polygons[i] = (pts - np.array([w/2, h/2])) / max_len
        else:
            raise NotImplementedError("Currently only pc_normalize=True is supported.")
        
        
        # from xray - from left to right
        upper_sort_universal = [i for i in range(1, 17)]
        lower_sort_universal = [i for i in range(17, 33)][::-1]
        _, _, upper_inds = np.intersect1d(upper_sort_universal, classes, return_indices=True)
        _, _, lower_inds = np.intersect1d(lower_sort_universal, classes, return_indices=True)
        
        
        pose_inds = upper_inds if pose == 'upper' else lower_inds
        this_polys, this_archs = sampling_points_and_polygons(uniform_polygons, pose_inds, pose)
        # upper_arch = [np.mean(uniform_polygons[v], axis=0) for v in upper_inds]
        # assert len(this_polys)  > 0, f"No polygons found for pose '{pose}' with indices {pose_inds}."
        # source - points
        this_polys_concat = np.concatenate(this_polys, axis=0)
        
        
        fit_upper_coeef = fit_quadratic_least_squares(this_archs)
        # center
        fit_upper_coeef = list(fit_upper_coeef)
        
        # make 2d points to 3d points
        target_points = this_polys_concat #np.concatenate([this_polys_concat, np.zeros_like(this_polys_concat[:, :1])], axis=-1)
        # target_points = this_polys_concat
        moving_points = target_points.copy()
        # fixed center of deform 
        
        

        deform_mode = False
        if deform_mode:

            deform_upper_arch, new_coeff = deform_quadratic_curve(
                this_archs,
                *fit_upper_coeef,
                a_ratio=0.08,
                b_ratio=0.0,
                c_range=0.05,
                x_scale_ratio=0.05,
                x_shift_range=2.0,
                x_noise_ratio=0.0,
            )

            moving_points = sampler.apply_proxy_idw_deformation_points( moving_points,
                                                    this_archs,
                                                    deform_upper_arch,

                                                    )
        else:
            pass
            # deform_upper_arch = this_archs
            # new_coeff = fit_upper_coeef

        moving_points = np.concatenate([moving_points, np.zeros_like(moving_points[:, :1])], axis=-1)
        target_points = np.concatenate([target_points, np.zeros_like(target_points[:, :1])], axis=-1)

        # sample the N moving points independently of the M target points:
        # moving_target_points keeps their pre-augmentation, target-frame
        # position for the loss (index-aligned with moving_points), while
        # target_points (M pts) stays only the model's reference-cloud input.
        # moving_target_points = random_select_points(moving_points, int(target_points.shape[0] * 0.8))
        
        # teeth-by-teeth drop
        args = [i for i in range(len(this_polys))]
        if np.random.uniform(0, 1) < 0.8:
            select = np.sort(np.random.choice(args, int(len(args)*0.8), replace=False))
        else:
            select = args
        # select = np.sort(np.random.choic?e(args, int(len(args)*0.8), replace=False))
        # all_indices = [np.arange(len(this_polys[i])) for i in range(len(this_polys))]
        all_indices = []
        start = 0
        for i in range(len(this_polys)):
            all_indices.append(start + np.arange(len(this_polys[i])))
            start += len(this_polys[i])
        select_indices = np.concatenate([all_indices[i] for i in select])
        # drop-points
        
        moving_points = moving_points[select_indices]
        moving_target_points = moving_points.copy()
        # moving_target_points = np.concatenate([this_polys[i] for i in select], axis=0)
        # moving_points = np.concatenate([this_polys[i] for i in select], axis=0)
        # all_indices_concat = np.concatenate(all_indices, axis=0)
        
        # all_indices_concat = 
        # select_indices = [np.arange(len(this_polys[i])) for i in select]
        
        # select_indices = [np]
        
        

        # pivot transform
        scale_range = [self.min_scale, self.max_scale] if self.estimate_scale else [1.0, 1.0]
        afm_mat = image_utils.batch_aug_params(
            {
                # 10 degree??
                'rotate': [np.pi/20, 0, 0 ],
                'translate': [0.005, 0.005 , 0.0],
                'scale': scale_range,
            },
            1,
            [0, 0, 0],
            pivot_scale=True,
        )

        rot, scale, translate = geometry_numpy.decompose_complete_matrix(afm_mat[0], first_scale=True)

        tol = 1e-6
        scale0 = scale.copy()
        np.fill_diagonal(scale0, 0)
        assert np.allclose(scale0, 0, atol=tol), 'we set identical x, y,z  scaling'
        R, scale, t = rot, scale[0, 0], translate
        moving_points = inverse_similarity_transform(moving_target_points, R, t, scale)

        if self.estimate_scale:
            items = [target_points, moving_points, moving_target_points, R, t, scale]
            target_points, moving_points, moving_target_points, R, t, scale = \
                [v.astype(np.float32) for v in items]
            return target_points, moving_points, moving_target_points, R, t, scale
        items = [target_points, moving_points, moving_target_points, R, t]
        target_points, moving_points, moving_target_points, R, t = \
            [v.astype(np.float32) for v in items]
        return target_points, moving_points, moving_target_points, R, t
        
        # vtk_utils.show([this_polys_concat, padding_pts1, vtk_utils.get_axes(200)])
        # # geometry_numpy.decompose_matrix(res[0])
        # # 
        # # similarity-transform
        # # 
        # # drop polygons
        
        # https://github.com/medit-AI/reverse-registration/blob/develop_registration/outputs
        # vtk_utils.split_show([
        #     vtk_utils.create_curve_actor(this_archs),
        #     vtk_utils.create_curve_actor(deform_upper_arch),
        #     uniform_polygons,
        #     vtk_utils.get_axes(100)
        # ], [
        #     deform_this_polys,
        #     this_polys_concat
        # ])

        # target["orig_size"] = torch.as_tensor([int(h), int(w)])
        # target["size"] = torch.as_tensor([int(h), int(w)])
        
        
        # upper & ar
        # import fpsample
        
        
        # # 
        # # transform
        
        # # smile-transform
        
        # dental_arch_center = 
    
        
        
        
        # num = 2048
        n_neighbor = 10
        # args = fpsample.bucket_fps_kdtree_sampling(res, num)
        # # mesh = vtk_utils.reconstruct_polydata(v, f)
        
        # from sklearn.neighbors import NearestNeighbors
        
        
        # return image, target




def fit_quadratic_least_squares(points):
    """
    (N, 2) points에 대해 quadratic curve fitting.

        y = a*x^2 + b*x + c

    Least Squares를 이용해서 a, b, c를 계산합니다.

    Parameters
    ----------
    points : (N, 2) ndarray
        입력 point 좌표 [x, y]

    Returns
    -------
    a, b, c : float
        fitted quadratic coefficients
    """

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if len(points) < 3:
        raise ValueError("At least 3 points are required.")

    x = points[:, 0]
    y = points[:, 1]

    # y = a*x^2 + b*x + c
    #
    # [x1^2  x1  1] [a]   [y1]
    # [x2^2  x2  1] [b] = [y2]
    # [ ...        ] [c]   [...]
    A = np.stack(
        [
            x ** 2,
            x,
            np.ones_like(x),
        ],
        axis=1,
    )

    coeffs, _, _, _ = np.linalg.lstsq(
        A,
        y,
        rcond=None,
    )

    a, b, c = coeffs

    return a, b, c

def deform_quadratic_curve(
    points,
    a,
    b,
    c,
    a_ratio=0.1,
    b_ratio=0.1,
    c_range=2.0,
    x_scale_ratio=0.05,
    x_shift_range=2.0,
    x_noise_ratio=0.0,
    rng=None,
):
    """
    quadratic curve coefficient와 x축을 함께 random perturbation하여
    point들을 augmentation합니다.

    Parameters
    ----------
    points : (N, 2) ndarray
        입력 points [x, y]

    a, b, c : float
        원본 quadratic fitting coefficient

    a_ratio : float
        a coefficient 상대 variation
        ex) 0.1 -> ±10%

    b_ratio : float
        b coefficient 상대 variation
        ex) 0.1 -> ±10%

    c_range : float
        c absolute random offset

    x_scale_ratio : float
        x축 전체 scale augmentation 범위
        ex) 0.05 -> 중심 기준 ±5% 확대/축소

    x_shift_range : float
        x축 전체 shift 범위
        ex) 2.0 -> [-2, +2]

    x_noise_ratio : float
        각 point별 작은 x noise.
        point spacing 대비 비율.
        보통 0~0.02 정도 권장.

    rng : np.random.Generator, optional

    Returns
    -------
    deformed_points : (N, 2)

    new_coeffs : tuple
        (new_a, new_b, new_c)
    """

    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2)")

    if rng is None:
        rng = np.random.default_rng()

    x = points[:, 0]
    y = points[:, 1]

    # ---------------------------------------------------------
    # 1. Original fitted curve / residual
    # ---------------------------------------------------------
    fitted_y = a * x**2 + b * x + c
    residual = y - fitted_y

    # ---------------------------------------------------------
    # 2. x-axis deformation
    #
    # 중심 기준으로 scale하여 arch width를 변화시킴.
    # ---------------------------------------------------------
    x_center = np.mean(x)

    x_scale = 1.0 + rng.uniform(
        -x_scale_ratio,
        x_scale_ratio,
    )

    x_shift = rng.uniform(
        -x_shift_range,
        x_shift_range,
    )

    new_x = (
        x_center
        + (x - x_center) * x_scale
        + x_shift
    )

    # optional point-wise small jitter
    if x_noise_ratio > 0:
        x_range = np.ptp(x)

        noise_scale = x_range * x_noise_ratio

        new_x += rng.uniform(
            -noise_scale,
            noise_scale,
            size=x.shape,
        )

    # ---------------------------------------------------------
    # 3. Quadratic coefficient augmentation
    # ---------------------------------------------------------
    new_a = a * (
        1.0 + rng.uniform(-a_ratio, a_ratio)
    )

    new_b = b * (
        1.0 + rng.uniform(-b_ratio, b_ratio)
    )

    new_c = c + rng.uniform(
        -c_range,
        c_range,
    )

    # ---------------------------------------------------------
    # 4. 변형된 x 위치에서 새로운 quadratic 계산
    # ---------------------------------------------------------
    augmented_y = (
        new_a * new_x**2
        + new_b * new_x
        + new_c
    )

    # 기존 curve로부터의 residual 유지
    new_y = augmented_y + residual

    # ---------------------------------------------------------
    # 5. output
    # ---------------------------------------------------------
    deformed_points = np.stack(
        [new_x, new_y],
        axis=1,
    )

    return deformed_points, (new_a, new_b, new_c)



class ResizeCocoSample:
    """
    Resize (image, target) pair.

    target format:
    {
        "boxes": np.ndarray[N, 4],       # xyxy
        "labels": np.ndarray[N],
        "masks": np.ndarray[N, H, W],    # optional
        ...
    }

    mask_resize_mode:
        "full" : full mask resize
        "roi"  : bbox 영역만 crop -> resize -> target canvas paste
    """

    def __init__(
        self,
        size: Tuple[int, int],
        keep_ratio: bool = True,
        mask_resize_mode: str = "roi",
    ):
        """
        size = (target_h, target_w)
        """
        self.target_h, self.target_w = size
        self.keep_ratio = keep_ratio
        self.mask_resize_mode = mask_resize_mode
        self.refer_width = 640

    def __call__(self, image: np.ndarray, target: dict):
        src_h, src_w = image.shape[:2]
        
        
        # (h, w)
        # size =  get_target_image_size([src_h, src_w], self.refer_width)

        if self.keep_ratio:
            image, scale_x, scale_y, offset_x, offset_y = \
                self._resize_letterbox(image)
        else:
            image = cv2.resize(
                image,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_LINEAR,
            )

            scale_x = self.target_w / src_w
            scale_y = self.target_h / src_h
            offset_x = 0
            offset_y = 0

        target = dict(target)

        boxes_src = np.asarray(
            target.get("boxes", []),
            dtype=np.float32,
        ).reshape(-1, 4)

        boxes_dst = self._resize_boxes(
            boxes_src,
            scale_x,
            scale_y,
            offset_x,
            offset_y,
        )

        target["boxes"] = boxes_dst

        if "masks" in target and target["masks"] is not None:
            masks = target["masks"]

            if self.mask_resize_mode == "roi":
                target["masks"] = self._resize_masks_roi(
                    masks,
                    boxes_src,
                    boxes_dst,
                )
            else:
                target["masks"] = self._resize_masks_full(
                    masks,
                    scale_x,
                    scale_y,
                    offset_x,
                    offset_y,
                )

        target["size"] = np.array(
            [self.target_h, self.target_w],
            dtype=np.int64,
        )

        return image, target

    # ---------------------------------------------------------
    # image
    # ---------------------------------------------------------

    def _resize_letterbox(self, image):
        src_h, src_w = image.shape[:2]

        scale = min(
            self.target_w / src_w,
            self.target_h / src_h,
        )

        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        offset_x = (self.target_w - new_w) // 2
        offset_y = (self.target_h - new_h) // 2

        if image.ndim == 3:
            canvas = np.zeros(
                (self.target_h, self.target_w, image.shape[2]),
                dtype=image.dtype,
            )
        else:
            canvas = np.zeros(
                (self.target_h, self.target_w),
                dtype=image.dtype,
            )

        canvas[
            offset_y:offset_y + new_h,
            offset_x:offset_x + new_w,
        ] = resized

        return (
            canvas,
            scale,
            scale,
            offset_x,
            offset_y,
        )

    # ---------------------------------------------------------
    # bbox
    # ---------------------------------------------------------

    def _resize_boxes(
        self,
        boxes,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
    ):
        if boxes.size == 0:
            return boxes.copy()

        boxes = boxes.copy()

        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] += offset_x
        boxes[:, [1, 3]] += offset_y

        boxes[:, [0, 2]] = np.clip(
            boxes[:, [0, 2]],
            0,
            self.target_w,
        )

        boxes[:, [1, 3]] = np.clip(
            boxes[:, [1, 3]],
            0,
            self.target_h,
        )

        return boxes

    # ---------------------------------------------------------
    # mask: optimized ROI resize
    # ---------------------------------------------------------

    def _resize_masks_roi(
        self,
        masks,
        boxes_src,
        boxes_dst,
    ):
        """
        Full resolution mask를 resize하지 않고
        bbox 영역만 잘라서 resize.

        입력:
            masks: [N, src_h, src_w]

        출력:
            [N, target_h, target_w]
        """

        masks = np.asarray(masks)
        masks_dtype = masks.dtype
        n = len(masks)

        out_masks = np.zeros(
            (n, self.target_h, self.target_w),
            dtype=masks.dtype,
        )

        for i in range(n):
            sx1, sy1, sx2, sy2 = boxes_src[i]

            # bbox -> integer crop range
            sx1 = max(int(np.floor(sx1)), 0)
            sy1 = max(int(np.floor(sy1)), 0)
            sx2 = min(int(np.ceil(sx2)), masks.shape[2])
            sy2 = min(int(np.ceil(sy2)), masks.shape[1])

            if sx2 <= sx1 or sy2 <= sy1:
                continue

            # -------------------------------------------
            # 핵심:
            # 4K full mask를 resize하지 않고 ROI만 추출
            # -------------------------------------------

            mask_crop = masks[
                i,
                sy1:sy2,
                sx1:sx2,
            ]

            dx1, dy1, dx2, dy2 = boxes_dst[i]

            dx1 = max(int(np.floor(dx1)), 0)
            dy1 = max(int(np.floor(dy1)), 0)
            dx2 = min(int(np.ceil(dx2)), self.target_w)
            dy2 = min(int(np.ceil(dy2)), self.target_h)

            dst_w = dx2 - dx1
            dst_h = dy2 - dy1

            if dst_w <= 0 or dst_h <= 0:
                continue

            mask_small = cv2.resize(
                mask_crop.astype(np.uint8),
                (dst_w, dst_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(masks_dtype)

            out_masks[
                i,
                dy1:dy2,
                dx1:dx2,
            ] = mask_small

        return out_masks

    # ---------------------------------------------------------
    # mask: simple/reference implementation
    # ---------------------------------------------------------

    def _resize_masks_full(
        self,
        masks,
        scale_x,
        scale_y,
        offset_x,
        offset_y,
    ):
        masks = np.asarray(masks)

        out_masks = np.zeros(
            (
                len(masks),
                self.target_h,
                self.target_w,
            ),
            dtype=np.uint8,
        )

        for i, mask in enumerate(masks):
            src_h, src_w = mask.shape

            new_w = int(round(src_w * scale_x))
            new_h = int(round(src_h * scale_y))

            resized = cv2.resize(
                mask,
                (new_w, new_h),
                interpolation=cv2.INTER_NEAREST,
            )

            out_masks[
                i,
                offset_y:offset_y + new_h,
                offset_x:offset_x + new_w,
            ] = resized

        return out_masks

import pycocotools.mask as coco_mask


# def convert_coco_poly_to_mask(segmentations, height, width):
def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


if __name__ == "__main__":
    dataset = CocoDetection(
        img_folder='E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm',
        ann_file='E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm/annotations.json',
        # None
        
        
    )
    
    assert len(dataset) > 0
    
    
    for _ in range(len(dataset)):
        item = dataset[np.random.randint(len(dataset))]
        tgt, src, moving_target, R, t, scale = item
        
        fit_src = similarity_transform(src, R, t, scale)
        # vtk_utils.split_show([
        #     src, tgt
        # ], [
        #     tgt, src
        # ])
        print(torch_utils.get_shape(item))
        
        tgt, src, moving_target, R, t, scale = item
        
        fit_src = similarity_transform(src, R, t, scale)
        vtk_utils.split_show([
            vtk_utils.create_points_actor(tgt, point_size=5, color=(0, 1, 0)),
            vtk_utils.create_points_actor(src, point_size=2, color=(1, 1, 0)),
        ], [
            vtk_utils.create_points_actor(tgt, point_size=5, color=(0, 1, 0)),
            vtk_utils.create_points_actor(fit_src, point_size=2, color=(1, 1, 0)),
            
        ], [
            vtk_utils.create_points_actor(moving_target, point_size=5, color=(0, 1, 0)),
            vtk_utils.create_points_actor(fit_src, point_size=2, color=(1, 1, 0)),
            
        ])
        print(torch_utils.get_shape(item))