from .common import Dataset


class EmptyDataset(Dataset):
    def __init__(self, root=None, classification=True, openset=True):
        super().__init__()
        self._classification = classification
        self._openset = openset

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return self._classification

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return self._openset

    @property
    def labels(self):
        return []

    def __getitem__(self, index):
        raise IndexError("No items in the dataset.")
