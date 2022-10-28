import os
import numpy as np

from scipy.io import loadmat

from .common import Dataset, imread


class Flower102Dataset(Dataset):
    """
    102 Category Flower Dataset dataset class.
    https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.
    """

    def __init__(self, root, annotation_key="trnid"):
        assert annotation_key in ("trnid", "valid", "tstid")
        split_indices = loadmat(os.path.join(root, "setid.mat"))[annotation_key][0]

        image_paths = np.array(sorted(os.listdir(os.path.join(root, "jpg"))))
        image_paths = image_paths[split_indices - 1]
        image_paths = [os.path.join(root, "jpg", p) for p in image_paths]

        image_labels = loadmat(os.path.join(root, "imagelabels.mat"))["labels"][0]
        image_labels = image_labels[split_indices - 1]

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
