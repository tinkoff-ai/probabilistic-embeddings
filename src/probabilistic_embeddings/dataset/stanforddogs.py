import os
import numpy as np

from scipy.io import loadmat

from .common import Dataset, imread


class StanfordDogsDataset(Dataset):
    """
    Stanford Dogs dataset class.
    https://vision.stanford.edu/aditya86/ImageNetDogs/

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.
    """

    def __init__(self, root, *, train=True):
        lists_path = "lists/train_list.mat" if train else "lists/test_list.mat"
        lists_path = os.path.join(root, lists_path)
        image_list = loadmat(lists_path)

        image_paths = [os.path.join(root, "images", a[0][0]) for a in image_list["file_list"]]
        image_labels = np.array(image_list["labels"].T[0], dtype=np.int) - 1

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
