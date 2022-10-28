from ..common import DatasetWrapper


class TransformDataset(DatasetWrapper):
    """Apply transform to the dataset."""

    def __init__(self, dataset, transform):
        super().__init__(dataset)
        self._transform = transform
        self._openset = dataset.openset

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        """
        if self.classification:
            item = self.dataset[index]
            image = self._transform(item[0])
            return (image,) + item[1:]
        else:
            item = self.dataset[index]
            image1 = self._transform(item[0][0])
            image2 = self._transform(item[0][1])
            return ((image1, image2),) + item[1:]


class RepeatDataset(DatasetWrapper):
    """Repeat dataset multiple times."""

    def __init__(self, dataset, n_times=1):
        super().__init__(dataset)
        self._n_times = n_times
        self._openset = dataset.openset

    def __len__(self):
        return len(self.dataset) * self._n_times

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]
