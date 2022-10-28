#!/usr/bin/env python3
from collections import Counter
from unittest import TestCase, main

import numpy as np

from probabilistic_embeddings.dataset.debug import DebugDataset
from probabilistic_embeddings.dataset.sampler import *


class TestShuffledClassBalancedBatchSampler(TestCase):
    def test_sampler(self):
        dataset = DebugDataset(root=None)
        assert dataset.num_classes == 4
        batch_size = 4
        labels_per_batch = 2
        samples_per_class = batch_size // labels_per_batch
        for uniform in [True, False]:
            sampler = ShuffledClassBalancedBatchSampler(dataset, batch_size=batch_size,
                                                        samples_per_class=samples_per_class,
                                                        uniform=uniform)
            self.assertEqual(len(sampler), len(dataset) // batch_size)
            for batch in sampler:
                for i in batch:
                    self.assertLessEqual(i, len(dataset))
                    self.assertGreaterEqual(i, 0)
                labels = [dataset.labels[i] for i in batch]
                counts = Counter(labels)
                self.assertEqual(len(counts), labels_per_batch)
                for v in counts.values():
                    self.assertEqual(v, samples_per_class)

    def test_balanced_sampler(self):
        labels = [0, 0, 3]
        sampler = BalancedLabelsSampler(labels, 2, num_batches=10)
        sampled = sum(sampler, [])
        counts = Counter(sampled)
        self.assertEqual(sum(counts.values()), 20)
        self.assertEqual(counts[0], 10)
        self.assertEqual(counts[3], 10)


class TestSameClassMixupCollator(TestCase):
    def test_simple(self):
        mixup = SameClassMixupCollator()
        images = torch.tensor([
            0.0,  # 0.
            1.0,  # 0.
            1.0,  # 1.
            2.0,  # 1.
            2.0,  # 2.
            2.5,  # 2.
            3.0,  # 2.
        ]).float().reshape(-1, 1, 1, 1)
        labels = torch.tensor([
            0,
            0,
            1,
            1,
            2,
            2,
            2
        ]).long()
        for _ in range(10):
            mixed_images, mixed_labels = mixup._mixup(images, labels)
            mixed_images = mixed_images.squeeze()
            self.assertTrue((mixed_labels == labels).all())
            self.assertTrue(((mixed_images >= labels) & (mixed_images <= labels + 1)).all())


if __name__ == "__main__":
    main()
