import random
from collections import defaultdict

import numpy as np
import torch


class UniformLabelsSampler:
    """Sample labels with equal probabilities."""
    def __init__(self, labels, labels_per_batch, num_batches):
        self._labels = set(labels)
        self._labels_per_batch = labels_per_batch
        self._num_batches = num_batches
        if len(self._labels) < labels_per_batch:
            raise ValueError("Can't sample equal number of labels. Batch is too large.")

    def __iter__(self):
        labels = list(self._labels)
        i = 0
        for _ in range(self._num_batches):
            if i + self._labels_per_batch > len(labels):
                random.shuffle(labels)
                i = 0
            yield list(labels[i:i + self._labels_per_batch])


class BalancedLabelsSampler:
    """Sample labels with probabilities equal to labels frequency."""
    def __init__(self, labels, labels_per_batch, num_batches):
        counts = np.bincount(labels)
        self._probabilities = counts / np.sum(counts)
        self._labels_per_batch = labels_per_batch
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = np.random.choice(len(self._probabilities), self._labels_per_batch, p=self._probabilities, replace=False)
            yield list(batch)


class ShuffledClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Sampler which extracts balanced number of samples for each class.

    Args:
        data_source: Source dataset. Labels field must be implemented.
        batch_size: Required batch size.
        samples_per_class: Number of samples for each class in the batch.
            Batch size must be a multiple of samples_per_class.
        uniform: If true, sample labels uniformly. If false, sample labels according to frequency.
    """

    def __init__(self, data_source, batch_size, samples_per_class, uniform=False):
        if batch_size > len(data_source):
            raise ValueError("Dataset size {} is too small for batch size {}.".format(
                len(data_source), batch_size))
        if batch_size % samples_per_class != 0:
            raise ValueError("Batch size must be a multiple of samples_per_class, but {} != K * {}.".format(
                batch_size, samples_per_class))

        self._data_source = data_source
        self._batch_size = batch_size
        self._labels_per_batch = self._batch_size // samples_per_class
        self._samples_per_class = samples_per_class
        label_sampler_cls = UniformLabelsSampler if uniform else BalancedLabelsSampler
        self._label_sampler = label_sampler_cls(data_source.labels, self._labels_per_batch,
                                                num_batches=len(self))

        by_label = defaultdict(list)
        for i, label in enumerate(data_source.labels):
            by_label[label].append(i)
        self._by_label = list(by_label.values())
        if self._labels_per_batch > len(self._by_label):
            raise ValueError("Can't sample {} classes from dataset with {} classes.".format(
                self._labels_per_batch, len(self._by_label)))

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        for labels in self._label_sampler:
            batch = []
            for label in labels:
                batch.extend(np.random.choice(self._by_label[label], size=self._samples_per_class, replace=True))
            yield batch

    def __len__(self):
        num_samples = len(self._data_source)
        num_batches = num_samples // self._batch_size
        return num_batches


class SameClassMixupCollator:
    """Applies same-class mixup to a batch from base sampler."""

    def __call__(self, values):
        images, labels = torch.utils.data._utils.collate.default_collate(values)
        return self._mixup(images, labels)

    def _mixup(self, images, labels):
        if isinstance(images, (list, tuple)):
            raise ValueError("Expected classification dataset for mixup.")
        cpu_labels = labels.long().cpu().numpy()
        by_label = defaultdict(list)
        for i, label in enumerate(cpu_labels):
            by_label[label].append(i)
        alt_indices = [random.choice(by_label[label]) for label in cpu_labels]
        alt_indices = torch.tensor(alt_indices, dtype=torch.long, device=labels.device)
        alt_images = images[alt_indices]
        weights = torch.rand(len(labels)).reshape(-1, 1, 1, 1)
        new_images = images * weights + alt_images * (1 - weights)
        return new_images, labels
