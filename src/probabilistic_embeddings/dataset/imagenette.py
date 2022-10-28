import os
from pathlib import Path

from .common import Dataset, imread


class ImagenetteDataset(Dataset):
    """
    Imagenette datasets class. These datasets are subsets of ImageNet dataset.
    Imagenette official page: https://github.com/fastai/imagenette.
    This dataset class is applicable for Imagenette, Imagewoof, Image网, and TinyImagenet datasets.

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.
    """

    def __init__(self, root, *, train=True):
        super().__init__()

        image_dir = "train" if train else "val"
        image_dir = Path(os.path.join(root, image_dir))

        class_dirs = sorted(os.listdir(os.path.join(root, "train")))
        dir2label = {path: i for i, path in enumerate(class_dirs)}

        image_paths = sorted(list(image_dir.rglob("*.JPEG")))
        image_labels = [dir2label[path.parent.name] for path in image_paths]

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
