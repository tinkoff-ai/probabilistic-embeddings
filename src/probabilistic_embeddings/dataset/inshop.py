import os

from .common import Dataset, imread


class InShopClothesDataset(Dataset):
    """In-shop clothes retrieval dataset.

    Test part of the dataset is obtained by joining gallery and query samples.

    See: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

    Args:
        root: Dataset root (with img subfolder and list_eval_partition.txt).
        train: Whether to use train or test part of the dataset.

    """

    IMG_ROOT = "img"
    LABELS = "list_eval_partition.txt"

    def __init__(self, root, *, train=True):
        super().__init__()

        self._image_paths = []
        labels = []

        with open(os.path.join(root, self.LABELS)) as fp:
            if fp.readline().strip() != "52712":
                raise RuntimeError("Unexpected labels file. Make sure you use original labels file.")
            if fp.readline().strip() != "image_name item_id evaluation_status":
                raise RuntimeError("Unexpected labels file. Make sure you use original labels file.")
            for line in fp:
                path, label, part = line.strip().split()
                if part == "train" and (not train):
                    continue
                if part != "train" and train:
                    continue
                self._image_paths.append(os.path.join(root, path))
                labels.append(label)
        part_labels = list(sorted(list(set(labels))))
        label_mapping = {label: i for i, label in enumerate(part_labels)}
        self._image_labels = [label_mapping[label] for label in labels]

        num_classes = len(set(self._image_labels))
        assert num_classes == 3997 if train else num_classes == 3985
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
