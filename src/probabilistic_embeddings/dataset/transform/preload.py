import numpy as np
import torch
from torchvision.transforms.functional import resize
from tqdm import tqdm

from ..common import DatasetWrapper
from .base import TransformDataset
from PIL import Image


class ResizePad:
    """Helper transform for preloading.

    Returns padded image and original shape.

    For classification dataset:
      image, label -> (image, shape), label
    For verification dataset:
      (image1, image2), label -> ((image1, shape1), (image2, shape2)), label

    """
    def __init__(self, image_size):
        self._image_size = image_size

    def __call__(self, image):
        assert isinstance(image, np.ndarray)
        max_size = max(image.shape[0], image.shape[1])
        scale_factor = self._image_size / max_size
        width = int(round(image.shape[1] * scale_factor))
        height = int(round(image.shape[0] * scale_factor))
        image = Image.fromarray(image)
        image = resize(image, (height, width))
        image = np.asarray(image)
        if image.shape[0] < self._image_size:
            image = np.concatenate((image, np.zeros((self._image_size - image.shape[0], image.shape[1], 3), dtype=image.dtype)), 0)
        elif image.shape[1] < self._image_size:
            image = np.concatenate((image, np.zeros((image.shape[0], self._image_size - image.shape[1], 3), dtype=image.dtype)), 1)
        assert image.shape == (self._image_size, self._image_size, 3)
        return image, [height, width]


class PreloadDataset(DatasetWrapper):
    """Load full dataset to memory.

    Useful for experiments with small datasets and large images.
    """

    def __init__(self, dataset, image_size, batch_size=32, num_workers=0):
        if dataset.has_quality:
            raise NotImplementedError("Can't preload datasets with sample quality available.")
        super().__init__(dataset)
        self._batch_size = batch_size
        dataset = TransformDataset(dataset, ResizePad(image_size))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        self._batches = list(tqdm(loader))

    def __getitem__(self, index):
        batch = self._batches[index // self._batch_size]
        index = index % self._batch_size
        if self.classification:
            image = self._crop(batch[0][0][index], (batch[0][1][0][index], batch[0][1][1][index]))
            label = batch[1][index]
            return image.numpy(), label
        else:
            image1 = self._crop(batch[0][0][0][index], (batch[0][0][1][0][index], batch[0][0][1][1][index]))
            image2 = self._crop(batch[0][1][0][index], (batch[0][1][1][0][index], batch[0][1][1][1][index]))
            label = batch[1][index]
            return (image1.numpy(), image2.numpy()), label

    def _crop(self, image, shape):
        return image[:shape[0], :shape[1]]
