from abc import ABC, abstractmethod
from contextlib import contextmanager

from PIL import Image
import jpeg4py
import numpy as np
import torch


def imread(filename):
    try:
        return Image.fromarray(jpeg4py.JPEG(filename).decode()).convert("RGB")
    except:
        return Image.open(filename).convert("RGB")


class Dataset(ABC, torch.utils.data.Dataset):
    """Dataset interface for metric learning."""

    @property
    @abstractmethod
    def classification(self):
        """Whether dataset is classification or verification."""
        pass

    @property
    @abstractmethod
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        pass

    @property
    @abstractmethod
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        pass

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return False

    @abstractmethod
    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).

        """
        pass

    @property
    def num_classes(self):
        """Get total number of classes."""
        if len(self.labels) == 0:
            return 0
        return max(self.labels) + 1

    @property
    def priors(self):
        """Get array of class priors."""
        counts = np.bincount(self.labels)
        return counts / np.sum(counts)

    def __len__(self):
        """Get dataset length."""
        return len(self.labels)


class DatasetWrapper(Dataset):
    """Base class for dataset extension."""

    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        """Get base dataset."""
        return self._dataset

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return self.dataset.classification

    @property
    def openset(self):
        return self.dataset.openset

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].
        """
        return self.dataset.labels

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return self.dataset.has_quality

    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).

        """
        return self.dataset[index]
