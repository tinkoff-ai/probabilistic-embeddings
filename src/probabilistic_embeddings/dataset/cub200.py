import os

from .common import Dataset, DatasetWrapper, imread
from .transform import MergedDataset, split_classes


class CUB200Dataset(Dataset):
    """Original Caltech-UCSD Birds 200 dataset. Train and test are splitted by sample.

    See http://www.vision.caltech.edu/visipedia/CUB-200.html

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.
        classification: If true, use original classification dataset.
            If false, sample pairs and provide verification dataset.
    """

    IMAGE_DIR = "images"
    IMAGE_FILENAME = "images.txt"
    LABELS_FILENAME = "image_class_labels.txt"
    SPLIT_FILENAME = "train_test_split.txt"

    def __init__(self, root, *, train=True):
        super().__init__()

        split_indices = []
        with open(os.path.join(root, self.SPLIT_FILENAME)) as fp:
            for line in fp:
                index, part = line.strip().split()
                if int(part) == int(train):
                    split_indices.append(index)
        split_indices = set(split_indices)

        indices = []
        image_labels = {}
        with open(os.path.join(root, self.LABELS_FILENAME)) as fp:
            for line in fp:
                index, label = line.strip().split()
                if index not in split_indices:
                    continue
                label = int(label) - 1
                indices.append(index)
                image_labels[index] = label
        num_classes = len(set(image_labels.values()))
        assert num_classes == 200
        assert max(image_labels.values()) == num_classes - 1

        image_paths = {}
        with open(os.path.join(root, self.IMAGE_FILENAME)) as fp:
            for line in fp:
                index, path = line.strip().split()
                if index not in split_indices:
                    continue
                image_paths[index] = os.path.join(root, self.IMAGE_DIR, path)
        self._paths = [image_paths[index] for index in indices]
        self._labels = [image_labels[index] for index in indices]

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
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns tuple (image, label).

        """
        path = self._paths[index]
        label = self._labels[index]
        image = imread(path)
        return image, label


class CUB200SplitClassesDataset(DatasetWrapper):
    """CUB200 dataset with different classes in train and test sets."""

    def __init__(self, root, *, train=True, interleave=False):
        merged = MergedDataset(CUB200Dataset(root, train=True), CUB200Dataset(root, train=False))
        trainset, testset = split_classes(merged, interleave=interleave)
        if train:
            super().__init__(trainset)
        else:
            super().__init__(testset)

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True
