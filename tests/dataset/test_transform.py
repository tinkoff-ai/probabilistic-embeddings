#!/usr/bin/env python3
import random
from unittest import TestCase, main

import numpy as np
import torch
from torchvision.transforms import ToTensor

from probabilistic_embeddings.dataset.common import Dataset
from probabilistic_embeddings.dataset.transform import *


class SimpleDataset(Dataset):
    def __init__(self, features="label"):
        super().__init__()
        self._labels = np.concatenate([
            np.arange(10),
            np.random.randint(0, 10, size=90)
        ]).astype(np.uint8)
        if features == "label":
            self._features = np.tile(self._labels[:, None, None, None], (1, 32, 8, 3))
        elif features == "range":
            self._features = (np.tile(np.arange(32 * 8).reshape(1, 32, 8, 1), (len(self._labels), 1, 1, 3)) % 255).astype(np.uint8)
        else:
            raise ValueError("Unknown features type: {}".format(features))

    @property
    def classification(self):
        return True

    @property
    def openset(self):
        return False

    @property
    def labels(self):
        return self._labels

    def __getitem__(self, index):
        return self._features[index], self._labels[index]


class TestTransform(TestCase):
    def test_dataset(self):
        dataset = SimpleDataset()
        self.assertEqual(len(dataset), 100)
        self.assertEqual(dataset.num_classes, 10)
        self.assertAlmostEqual(np.sum(dataset.priors), 1)
        self.assertTrue(np.all(dataset.priors > 0))
        for i in random.sample(range(100), 20):
            self.assertEqual(dataset[i][0][0, 0, 0], dataset[i][1])

    def test_repeat(self):
        dataset = SimpleDataset()
        dataset = RepeatDataset(dataset, 3)
        self.assertEqual(len(dataset), 300)
        for i in random.sample(range(len(dataset)), 20):
            self.assertEqual(dataset[i][0][0, 0, 0], dataset[i][1])

    def test_merged(self):
        dataset1 = SimpleDataset()
        dataset2 = SimpleDataset()
        dataset = MergedDataset(dataset1, dataset2)
        self.assertEqual(len(dataset), len(dataset1) + len(dataset2))
        self.assertEqual(dataset.num_classes, 10)
        for i in random.sample(range(len(dataset)), 20):
            if i < len(dataset1):
                self.assertEqual(dataset[i][0][0, 0, 0], dataset1[i][1])
            else:
                self.assertEqual(dataset[i][0][0, 0, 0], dataset2[i - len(dataset1)][1])

    def test_sample_pairs(self):
        base_dataset = SimpleDataset()
        for size_factor in [1, 3]:
            dataset = SamplePairsDataset(base_dataset, size_factor=size_factor)
            self.assertEqual(len(dataset), 2 * len(base_dataset) * size_factor)
            self.assertEqual(dataset.priors[0], 0.5)
            self.assertEqual(dataset.priors[1], 0.5)
            for i in random.sample(range(len(dataset)), 20):
                f1, f2 = dataset[i][0]
                label = dataset[i][1]
                if label:
                    self.assertEqual(f1[0, 0, 0], f2[0, 0, 0])
                else:
                    self.assertNotEqual(f1[0, 0, 0], f2[0, 0, 0])

    def test_preload(self):
        dataset = SimpleDataset()
        preloaded = PreloadDataset(dataset, image_size=8)
        self.assertEqual(len(dataset), len(preloaded))
        for i in random.sample(range(len(dataset)), 20):
            self.assertEqual(dataset[i][0][0, 0, 0], preloaded[i][0][0, 0, 0])
            self.assertEqual(dataset[i][1], preloaded[i][1])

    def test_split_classes(self):
        dataset = SimpleDataset()
        for interleave in [True, False]:
            train, val = split_classes(dataset, 0.3, interleave=interleave)
            self.assertEqual(train.num_classes, 3)
            self.assertEqual(val.num_classes, 7)
            self.assertEqual(len(train) + len(val), len(dataset))
            train_labels = {int(train[i][0][0, 0, 0]) for i in range(len(train))}
            val_labels = {int(val[i][0][0, 0, 0]) for i in range(len(val))}
            self.assertFalse(train_labels & val_labels)
            self.assertEqual(train_labels | val_labels, set(dataset.labels))

    def test_split_crossval_classes(self):
        dataset = SimpleDataset()
        for interleave in [True, False]:
            train, val = split_crossval_classes(dataset, 0, 5, interleave=interleave)
            self.assertEqual(train.num_classes, 8)
            self.assertEqual(val.num_classes, 2)
            self.assertEqual(len(train) + len(val), len(dataset))
            train_labels = {int(train[i][0][0, 0, 0]) for i in range(len(train))}
            val_labels = {int(val[i][0][0, 0, 0]) for i in range(len(val))}
            self.assertFalse(train_labels & val_labels)
            self.assertEqual(train_labels | val_labels, set(dataset.labels))

            train2, val2 = split_crossval_classes(dataset, 1, 5, interleave=interleave)
            val2_labels = {int(val2[i][0][0, 0, 0]) for i in range(len(val2))}
            self.assertFalse(val_labels & val2_labels)

    def test_lossy(self):
        base_dataset = SimpleDataset(features="range")
        lossy_config = {"center_crop_range": [0.25, 0.25]}
        image_gt_list = np.asarray([
            [15 * 8 + 3, 15 * 8 + 4],
            [16 * 8 + 3, 16 * 8 + 4]
        ]) % 255

        # Test Numpy.
        dataset = LossyDataset(base_dataset, config=lossy_config)
        image = np.asarray(dataset[5][0])
        image_gt = np.tile(image_gt_list.reshape(2, 2, 1), (1, 1, 3))
        self.assertTrue((image == image_gt).all())

        # Test Torch.
        dataset = LossyDataset(TransformDataset(base_dataset, ToTensor()), config=lossy_config)
        image = (dataset[5][0] * 255).round()
        image_gt = torch.tile(torch.tensor(image_gt_list).reshape(1, 2, 2), (3, 1, 1))
        self.assertTrue((image == image_gt).all())


if __name__ == "__main__":
    main()
