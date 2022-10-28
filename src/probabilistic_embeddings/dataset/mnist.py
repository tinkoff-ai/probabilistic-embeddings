import numpy as np
from torchvision.datasets import MNIST

from .common import Dataset, DatasetWrapper
from .transform import MergedDataset, split_classes


class MnistDataset(Dataset):
    """MNIST dataset class.

    Args:
        root: Dataset root.
        train: Whether to use train or val part of the dataset.
    """

    def __init__(self, root, train=True, download=True):
        super().__init__()
        self._dataset = MNIST(root, train=train, download=download)

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
        return image.convert("RGB"), int(label)


class MnistSplitClassesDataset(DatasetWrapper):
    """MNIST dataset with different classes in train and test sets."""

    def __init__(self, root, *, train=True, interleave=False):
        merged = MergedDataset(MnistDataset(root, train=True), MnistDataset(root, train=False))
        trainset, testset = split_classes(merged, interleave=interleave)
        if train:
            super().__init__(trainset)
        else:
            super().__init__(testset)

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True
