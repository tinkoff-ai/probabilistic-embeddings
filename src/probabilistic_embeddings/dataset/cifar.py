from abc import abstractmethod

import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100

from .common import Dataset


class TorchVisionDataset(Dataset):  # Dataset is already ABC.
    """Common class for several TorchVision datasets.

    Args:
        root: Dataset root.
        train: Whether to use train or val part of the dataset.
    """

    def __init__(self, root, train=True, download=True):
        super().__init__()
        self._dataset = self.get_cls()(root, train=train, download=download)

    @staticmethod
    @abstractmethod
    def get_cls():
        pass

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
        return self._dataset.targets

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns tuple (image, label).

        """
        image, label = self._dataset[index]
        return image, int(label)


class CIFAR10Dataset(TorchVisionDataset):
    @staticmethod
    def get_cls():
        return CIFAR10


class CIFAR100Dataset(TorchVisionDataset):
    @staticmethod
    def get_cls():
        return CIFAR100
