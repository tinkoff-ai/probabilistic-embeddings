import os

import numpy as np

from .common import Dataset, imread


class ImageNetLTDataset(Dataset):
    """
    ImageNet-LT dataset class.
    https://github.com/zhmiao/OpenLongTailRecognition-OLTR

    Args:
        root: Dataset root.
        mode: Whether to use train, val or test part of the dataset.
    """

    TEST_SETUPS = {
        "overall": lambda count: True,
        "many-shot": lambda count: count > 100,
        "medium-shot": lambda count: 100 >= count > 20,
        "few-shot": lambda count: count < 20
    }

    def __init__(self, root, mode="train", test_setup=None):
        if test_setup not in (None, "overall", "many-shot", "medium-shot", "few-shot"):
            raise ValueError("Unknown test setup.")
        if mode not in ("train", "val", "test"):
            raise ValueError("Unknown dataset mode.")
        file_list_path = f"{mode}.txt"
        file_list_path = os.path.join(root, file_list_path)

        image_paths = []
        image_labels = []

        with open(file_list_path, "r") as f:
            for line in f.readlines():
                img_path, img_label = line.split(" ")
                image_paths.append(os.path.join(root, "/".join([img_path.split("/")[0], img_path.split("/")[-1]])))
                image_labels.append(int(img_label))

        self._image_paths = image_paths
        self._image_labels = image_labels

        if test_setup:
            self._apply_test_setup(root, test_setup)

    def _apply_test_setup(self, root, setup):
        train_file_list = os.path.join(root, "train.txt")
        train_image_labels = []

        with open(train_file_list, "r") as f:
            for line in f.readlines():
                _, img_label = line.split(" ")
                train_image_labels.append(int(img_label))

        labels, label_counts = np.unique(np.array(train_image_labels), return_counts=True)
        label_counts = dict(zip(list(labels), list(label_counts)))

        image_paths = []
        image_labels = []

        for path, label in zip(self._image_paths, self._image_labels):
            if self.TEST_SETUPS[setup](label_counts[label]):
                image_labels.append(label)
                image_paths.append(path)

        self._image_paths = image_paths
        self._image_labels = image_labels

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
