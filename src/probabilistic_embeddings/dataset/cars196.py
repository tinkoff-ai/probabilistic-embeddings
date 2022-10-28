import os

import scipy.io

from .common import Dataset, DatasetWrapper, imread
from .transform import MergedDataset, split_classes


class Cars196Dataset(Dataset):
    """Original cars dataset. Train and test are splitted by sample.

    See https://ai.stanford.edu/%7Ejkrause/cars/car_dataset.html

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.
    """

    TRAIN_DIR = "cars_train"
    TEST_DIR = "cars_test"
    TRAIN_LABELS = os.path.join("devkit", "cars_train_annos.mat")
    TEST_LABELS = os.path.join("devkit", "cars_test_annos_withlabels.mat")

    def __init__(self, root, *, train=True):
        super().__init__()

        if train:
            annotations = scipy.io.loadmat(os.path.join(root, self.TRAIN_LABELS))["annotations"]
            image_root = os.path.join(root, self.TRAIN_DIR)
        else:
            annotations = scipy.io.loadmat(os.path.join(root, self.TEST_LABELS))["annotations"]
            image_root = os.path.join(root, self.TEST_DIR)

        self._image_paths = []
        self._image_labels = []
        for record in annotations[0]:
            label = int(record["class"][0, 0]) - 1
            path = str(record["fname"][0])
            self._image_paths.append(os.path.join(image_root, path))
            self._image_labels.append(label)
        num_classes = len(set(self._image_labels))
        assert num_classes == 196
        assert max(self._image_labels) == num_classes - 1

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

        Labels are integers in the range [0, N-1].

        """
        return self._image_labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        """
        path = self._image_paths[index]
        label = self._image_labels[index]
        image = imread(path)
        return image, label


class Cars196SplitClassesDataset(DatasetWrapper):
    """Cars dataset with different classes in train and test sets."""

    def __init__(self, root, *, train=True, interleave=False):
        merged = MergedDataset(Cars196Dataset(root, train=True), Cars196Dataset(root, train=False))
        trainset, testset = split_classes(merged, interleave=interleave)
        if train:
            super().__init__(trainset)
        else:
            super().__init__(testset)

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True
