import os

from .common import Dataset, imread


class LFWDataset(Dataset):
    """PyTorch interface to LFW dataset with classification labels or pairs.

    Args:
        root: Path to the dataset root with images and annotations.
        train: If True, use training part of the dataset. If False, use validation or testing part
            depending on `cross_val_step`.
        classification: If False, sample positive and negative pairs. Label will contain SAME label.
            If True, samples images and integer class label.
        cross_val_step: Index of cross validation step in the range [0, 9].
            If not provided, standard train/dev split will be used.
    """

    IMAGES_ROOT = "lfw-deepfunneled"

    TRAIN_LABELS = "peopleDevTrain.txt"
    VALIDATION_LABELS = "peopleDevTest.txt"
    CROSS_LABELS = "people.txt"

    TRAIN_PAIRS = "pairsDevTrain.txt"
    VALIDATION_PAIRS = "pairsDevTest.txt"
    CROSS_PAIRS = "pairs.txt"

    def __init__(self, root, *, train=True, classification=True, cross_val_step=None):
        super().__init__()

        if cross_val_step is not None:
            raise NotImplementedError("Cross-validation")

        self._train = train
        self._classification = classification

        images_root = os.path.join(root, self.IMAGES_ROOT)
        self._image_paths, self._image_labels, label_to_indices = self._find_images(images_root)

        if classification:
            labels_filename = self.TRAIN_LABELS if train else self.VALIDATION_LABELS
            labels = self._read_classification_labels(os.path.join(root, labels_filename))
            subset = list(sorted(sum([label_to_indices[label] for label in labels], [])))
            label_mapping = {label: i for i, label in enumerate(labels)}
            self._image_paths = [self._image_paths[i] for i in subset]
            self._image_labels = [label_mapping[self._image_labels[i]] for i in subset]
        else:
            pairs_filename = self.TRAIN_PAIRS if train else self.VALIDATION_PAIRS
            self._pairs, self._pair_labels = self._read_pairs(os.path.join(root, pairs_filename), label_to_indices)

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return self._classification

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        if self._classification:
            return self._image_labels
        else:
            return self._pair_labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        """
        if self._classification:
            path = self._image_paths[index]
            label = self._image_labels[index]
            image = imread(path)
            return image, label
        else:
            index1, index2 = self._pairs[index]
            label = self._pair_labels[index]
            image1 = imread(self._image_paths[index1])
            image2 = imread(self._image_paths[index2])
            return (image1, image2), label

    @staticmethod
    def _find_images(images_root):
        image_paths = []
        image_labels = []
        label_to_indices = {}
        for label in sorted(os.listdir(images_root)):
            label_to_indices[label] = []
            for filename in sorted(os.listdir(os.path.join(images_root, label))):
                assert filename.endswith(".jpg")
                label_to_indices[label].append(len(image_paths))
                image_paths.append(os.path.join(images_root, label, filename))
                image_labels.append(label)
        return image_paths, image_labels, label_to_indices

    @staticmethod
    def _read_classification_labels(filename):
        labels = []
        with open(filename) as fp:
            n = int(fp.readline())
            for _ in range(n):
                labels.append(fp.readline().strip().split()[0])
        return list(sorted(labels))

    @staticmethod
    def _read_pairs(filename, label_to_indices):
        pairs = []
        labels = []
        with open(filename) as fp:
            n = int(fp.readline())
            for _ in range(n):
                label, index1, index2 = fp.readline().strip().split()
                index1, index2 = int(index1) - 1, int(index2) - 1
                pairs.append((label_to_indices[label][index1], label_to_indices[label][index2]))
                labels.append(1)
            for _ in range(n):
                label1, index1, label2, index2 = fp.readline().strip().split()
                index1, index2 = int(index1) - 1, int(index2) - 1
                pairs.append((label_to_indices[label1][index1], label_to_indices[label2][index2]))
                labels.append(0)
        return pairs, labels


class CrossLFWTestset(Dataset):
    """PyTorch interface to CALFW and CPLFW.

    Args:
        root: Path to the images root.
    """

    def __init__(self, root):
        super().__init__()
        labels = []
        self._image_paths = []
        for subroot, _, filenames in os.walk(root):
            for filename in filenames:
                basename, ext = os.path.splitext(filename)
                if not ext.lower() == ".jpg":
                    continue
                label, _ = basename.rsplit("_", 1)
                labels.append(label)
                self._image_paths.append(os.path.join(subroot, filename))
        label_mapping = {label: i for i, label in enumerate(sorted(set(labels)))}
        self._image_labels = [label_mapping[label] for label in labels]

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
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

        Classification dataset returns tuple (image, label).

        """
        label = self._image_labels[index]
        image = imread(self._image_paths)
        return image, label
