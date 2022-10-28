from torchvision.datasets import SVHN

from .common import Dataset


class SVHNDataset(Dataset):
    """SVHN dataset class.

    Args:
        root: Dataset root.
        train: Whether to use train or val part of the dataset.
    """

    def __init__(self, root, split="train", download=True):
        super().__init__()
        self._dataset = SVHN(root, split=split, download=download)

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
        return self._dataset.labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns tuple (image, label).

        """
        image, label = self._dataset[index]
        return image.convert("RGB"), int(label)
