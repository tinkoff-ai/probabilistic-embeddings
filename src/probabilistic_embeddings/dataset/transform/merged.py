import numpy as np
from collections import defaultdict

from ..common import Dataset


class MergedDataset(Dataset):
    """Merge multiple datasets sharing the same set of labels."""
    def __init__(self, *datasets):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("Empty datasets list.")
        for dataset in datasets[1:]:
            if dataset.classification != datasets[0].classification:
                raise ValueError("Can't merge classification and verification datasets.")
            if dataset.has_quality != datasets[0].has_quality:
                raise ValueError("Can't merge datasets with and without quality scores.")
            if dataset.num_classes != datasets[0].num_classes:
                raise ValueError("Different number of classes in datasets.")
            if dataset.openset != datasets[0].openset:
                raise ValueError("Different openset flag in datasets.")
        self._datasets = datasets
        self._labels = np.concatenate([dataset.labels for dataset in datasets])

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return self._datasets[0].classification

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return self._datasets[0].openset

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return self._datasets[0].has_quality

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).

        """
        for dataset in self._datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError(index)


class ClassMergedDataset(Dataset):
    """Merge multiple datasets sharing different sets of labels."""
    def __init__(self, *datasets):
        super().__init__()
        if len(datasets) == 0:
            raise ValueError("Empty datasets list.")
        for dataset in datasets:
            if not dataset.classification:
                raise ValueError("Expected classification dataset.")
        for dataset in datasets[1:]:
            if dataset.has_quality != datasets[0].has_quality:
                raise ValueError("Can't merge datasets with and without quality scores.")
            if dataset.openset != datasets[0].openset:
                raise ValueError("Different openset flag in datasets.")
        dataset_labels = []
        total_labels = 0
        for dataset in datasets:
            dataset_labels.append([total_labels + label for label in dataset.labels])
            total_labels += max(dataset.labels) + 1
        self._datasets = datasets
        self._labels = np.concatenate(dataset_labels)

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return True

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return self._datasets[0].openset

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return self._datasets[0].has_quality

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).

        """
        for dataset in self._datasets:
            if index < len(dataset):
                item = list(dataset[index])
                item[1] = self._labels[index]
                return tuple(item)
            index -= len(dataset)
        raise IndexError(index)
