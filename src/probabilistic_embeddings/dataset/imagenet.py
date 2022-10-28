import os

import numpy as np

from .common import Dataset, imread
from pathlib import Path
from scipy.io import loadmat


class ImageNetDataset(Dataset):
    """ImageNet dataset class.

    Args:
        root: Dataset root.
        train: Whether to use train or val part of the dataset.
    """

    def __init__(self, root, train=True):
        super().__init__()

        if train:
            image_dir = "train"
            image_dir = Path(os.path.join(root, image_dir))

            meta = loadmat(os.path.join(root, "meta.mat"))
            dir2label = {syn[0][1][0]: int(syn[0][0][0][0]) - 1 for syn in meta["synsets"]}

            image_paths = sorted(list(image_dir.rglob("*.JPEG")))
            image_labels = [dir2label[path.parent.name] for path in image_paths]
        else:
            image_dir = "val"
            image_dir = Path(os.path.join(root, image_dir))
            image_paths = sorted(list(image_dir.rglob("*.JPEG")))
            with open(os.path.join(root, "ILSVRC2012_validation_ground_truth.txt"), "r") as f:
                image_labels = [int(label) - 1 for label in f.readlines()]

        assert min(image_labels) == 0
        assert max(image_labels) == 999
        assert len(image_paths) == len(image_labels)
        self._image_paths = image_paths
        self._image_labels = np.array(image_labels)

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return True

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return False

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1], where N is number of classes

        """
        return self._image_labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns tuple (image, label).

        """
        path = self._image_paths[index]
        label = self._image_labels[index]
        image = imread(path)

        return image, label
