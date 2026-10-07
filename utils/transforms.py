# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchvision.transforms.functional as F

# taken from https://github.com/aimagelab/mammoth/blob/master/datasets/transforms/permutation.py
class Permutation(object):
    def __init__(self, perm) -> None:
        self.perm = perm

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample)
        old_shape = sample.shape

        return sample.flatten()[self.perm].reshape(old_shape)


# taken from https://github.com/aimagelab/mammoth/blob/master/datasets/transforms/rotation.py
class FixedRotation(object):
    def __init__(self, degrees: int) -> None:
        self.degrees = degrees

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return F.rotate(x, self.degrees)


class BilinearTensorRotation(object):
    """Bilinear rotation on the raw ``[0, 1]`` tensor, then normalize.

    ``expand=False`` keeps the input spatial size (28×28 for MNIST). Used by
    the cyclic / one-pass rot-MNIST streams. ``FixedRotation`` keeps the
    default nearest-neighbour interpolation.
    """

    def __init__(self, degrees, mean, std) -> None:
        self.degrees = float(degrees)
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if not torch.is_tensor(img):
            img = F.to_tensor(img)
        rotated = F.rotate(
            img,
            self.degrees,
            interpolation=F.InterpolationMode.BILINEAR,
            expand=False,
        )
        return F.normalize(rotated, self.mean, self.std)
