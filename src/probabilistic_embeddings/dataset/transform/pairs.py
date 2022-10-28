import random
from collections import defaultdict

import numpy as np

from ...torch import tmp_seed
from ..common import Dataset


class SamplePairsDataset(Dataset):
    """Verification dataset based on dataset with labels.

    Args:
        dataset: Classification dataset to sample pairs from.
        size_factor: The number of pairs in verification dataset is
            `2 * N * size_factor`, where N is the number of images.
        seed: Random seed.
    """

    def __init__(self, dataset, size_factor=1, seed=0):
        super().__init__()
        self._dataset = dataset
        self._size_factor = size_factor
        with tmp_seed(seed):
            same_pairs = self._sample_same_pairs(dataset.labels)
            diff_pairs = self._sample_diff_pairs(dataset.labels)
        self._labels = [1] * len(same_pairs) + [0] * len(diff_pairs)
        self._pairs = same_pairs + diff_pairs

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return False

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return False

    @property
    def has_quality(self):
        """Whether dataset assigns quality score to each sample or not."""
        return self._dataset.has_quality

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are 0/1 integers.

        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns ((image1, image2), label).

        """
        index1, index2 = self._pairs[index]
        item1 = self._dataset[index1]
        item2 = self._dataset[index2]
        label = self._labels[index]
        if self._dataset.has_quality:
            return (item1[0], item2[0]), label, (item1[2], item2[2])
        else:
            return (item1[0], item2[0]), label

    @staticmethod
    def _permute_ne(n):
        """Generate random permutation such that p[i] != i."""
        p = np.random.permutation(n)
        equals = np.nonzero(p == np.arange(n))[0]
        if len(equals) > 1:
            # Cycle shift for equals.
            p[equals] = p[np.roll(equals, 1)]
        elif len(equals) == 1:
            i = equals[0]
            j = np.random.randint(0, n - 1)
            if j >= i:
                j += 1
            p[i] = p[j]
            p[j] = i
        return p

    def _sample_same_pairs(self, labels):
        """Sample pairs of samples with the same label.

        Output number of pairs is len(labels) * size_factor.
        """
        by_label = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[label].append(i)
        all_labels = list(sorted(by_label))
        pairs = []
        for label in all_labels:
            indices = by_label[label]
            if len(indices) == 1:
                continue
            for _ in range(self._size_factor):
                for i, j in enumerate(self._permute_ne(len(indices))):
                    pairs.append((indices[i], indices[j]))
        return pairs

    def _sample_diff_pairs(self, labels):
        """Sample pairs with different labels.

        Output number of pairs is len(labels) * size_factor.
        """
        by_label = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[label].append(i)
        pairs = []
        for i, label in enumerate(labels):
            for _ in range(self._size_factor):
                alt_label = label
                while alt_label == label:
                    alt_label = random.choice(labels)
                j = random.choice(by_label[alt_label])
                pairs.append((i, j))
        return pairs
