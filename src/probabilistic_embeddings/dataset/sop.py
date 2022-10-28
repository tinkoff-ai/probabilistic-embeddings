import os

from .common import Dataset, imread


class SOPDataset(Dataset):
    """Original Stanford Online Products dataset. Train and test are splitted by sample.

    See: https://cvgl.stanford.edu/projects/lifted_struct/

    Args:
        root: Dataset root.
        train: Whether to use train or test part of the dataset.

    """

    TRAIN_LABELS = "Ebay_train.txt"
    TEST_LABELS = "Ebay_test.txt"

    def __init__(self, root, *, train=True):
        super().__init__()

        if train:
            labels_file = os.path.join(root, self.TRAIN_LABELS)
        else:
            labels_file = os.path.join(root, self.TEST_LABELS)

        self._image_paths = []
        self._image_labels = []
        with open(labels_file) as fp:
            assert fp.readline().strip() == "image_id class_id super_class_id path"
            for line in fp:
                _, label_low, label_high, path = line.strip().split()
                label = int(label_low) - 1
                if not train:
                    label -= 11318
                self._image_paths.append(os.path.join(root, path))
                self._image_labels.append(label)
        num_classes = len(set(self._image_labels))
        assert num_classes == 11318 if train else num_classes == 11316
        assert min(self._image_labels) == 0
        assert max(self._image_labels) == num_classes - 1

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return True

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

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
