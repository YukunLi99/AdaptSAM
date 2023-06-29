# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
from torchvision.transforms import InterpolationMode

from copy import deepcopy
from typing import Tuple, List
from PIL import Image

from detectron2.data import transforms as T

class ResizeLongestEdge(T.Augmentation):  #数据增强修改完毕
    """
    Resize the image while keeping the aspect ratio unchanged.
    It attempts to scale the shorter edge to the given `short_edge_length`,
    as long as the longer edge does not exceed `max_size`.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    @torch.jit.unused
    def __init__(
        self, target_length
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        self.target_length = target_length
        self._init(locals())

    @torch.jit.unused
    def get_transform(self, image):
        h, w = image.shape[:2]
        return ResizeLongestSide(h, w, self.target_length)
    
class ResizeLongestSide(T.Transform):
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, h, w, target_length: int) -> None:
        self.target_length = target_length
        self.old_h  = h
        self.old_w = w

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        # return np.array(resize(to_pil_image(image), target_size))
    
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        if len(image.shape) == 3:
            return np.array(resize(to_pil_image(image), target_size))
        else:
            return np.array(resize(to_pil_image(image), target_size, interpolation=InterpolationMode.NEAREST))

    # def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = self.old_h, self.old_w
        new_h, new_w = self.get_preprocess_shape(
            old_h, old_w, self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    # def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
    def apply_boxes(self, boxes: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        # boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2))
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


    def apply_multi_coords(self, coords: np.ndarray, original_size: List[Tuple[int, ...]]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        original_size = np.stack(original_size, axis=0) #batchsize, 2
        old_h, old_w = original_size[:, 0:1], original_size[:, 1:] #batchsize, 1

        # old_h, old_w = original_size
        # new_h, new_w = self.get_preprocess_shape(
        #     original_size[0], original_size[1], self.target_length
        # )

        new_h, new_w = self.get_batch_preprocess_shape(
            old_h, old_w, self.target_length
        ) #batchsize, 1

        coords = deepcopy(coords).astype(float)  #batchsize, 100, 2

        #before
        # coords[..., 0] = coords[..., 0] * (new_w[:, None, :] / old_w[:, None, :])  #batchsize, 100, 1
        # coords[..., 1] = coords[..., 1] * (new_h[:, None, :] / old_h[:, None, :])  #batchsize, 100, 1

        #after
        if len(coords.shape) == 4:
            new_w, old_w, new_h, old_h = new_w[:, None], old_w[:, None], new_h[:, None], old_h[:, None]
            
        coords[..., 0] = coords[..., 0] * (new_w / old_w)  #batchsize, 100
        coords[..., 1] = coords[..., 1] * (new_h / old_h)  #batchsize, 100

        return coords #batchsize, 100, 2

    @staticmethod
    def get_batch_preprocess_shape(oldh: np.ndarray, oldw: np.ndarray, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """

        scale = long_side_length * 1.0 / np.maximum(oldh, oldw)  #batchsize, 1
        newh, neww = oldh * scale, oldw * scale
        # neww = int(neww + 0.5)
        # newh = int(newh + 0.5)
        neww = (neww + 0.5).astype(int)
        newh = (newh + 0.5).astype(int)
        
        return (newh, neww)  #batchsize, 1