from collections import OrderedDict

import numpy as np
import torch
from torchvision.transforms import functional as functional_transforms
from PIL import Image

from ...config import prepare_config, ConfigError
from ...torch import tmp_seed
from ..common import DatasetWrapper


class LossyDataset(DatasetWrapper):
    """Add lossy transformations to input data."""

    @staticmethod
    def get_default_config(seed=0, center_crop_range=[0.25, 1.0]):
        """Get lossy dataset parameters.

        Args:
            center_crop_range: Minimum and maximum size of center crop.
        """
        return OrderedDict([
            ("seed", seed),
            ("center_crop_range", center_crop_range)
        ])

    def __init__(self, dataset, config=None):
        super().__init__(dataset)
        self._config = prepare_config(self, config)

        if not dataset.classification:
            raise NotImplementedError("Only lossy classification datasets are supported.")

        crop_min, crop_max = self._config["center_crop_range"]
        if crop_min > crop_max:
            raise ConfigError("Crop min size is greater than max.")
        with tmp_seed(self._config["seed"]):
            self._center_crop = np.random.random(len(dataset)) * (crop_max - crop_min) + crop_min

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return True

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label, quality).
        Verification dataset returns ((image1, image2), label, (quality1, quality2)).

        """
        assert self.dataset.classification
        image, label = self.dataset[index][:2]

        if isinstance(image, Image.Image):
            image = np.asarray(image)

        center_crop = self._center_crop[index]
        if abs(center_crop - 1) > 1e-6:
            if isinstance(image, np.ndarray):
                # Image in HWC format.
                size = int(round(min(image.shape[0], image.shape[1]) * center_crop))
                y_offset = (image.shape[0] - size) // 2
                x_offset = (image.shape[1] - size) // 2
                image = image[y_offset:y_offset + size, x_offset:x_offset + size]
            elif isinstance(image, torch.Tensor):
                # Image in CHW format.
                size = int(round(min(image.shape[1], image.shape[2]) * center_crop))
                image = functional_transforms.center_crop(image, size)
            else:
                raise ValueError("Expected Numpy or torch Tensor.")
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        quality = center_crop
        return image, label, quality
