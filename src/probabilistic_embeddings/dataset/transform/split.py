from sklearn.model_selection import train_test_split, KFold

from ..common import Dataset


def train_test_interleave_split(classes, test_size):
    """Split classes into train and test subsets.

    Args:
        test_size: Fraction of the test in the [0, 1] range.

    Returns:
        Train classes and test classes.
    """
    classes1 = []
    classes2 = []
    s = 0
    for c in classes:
        s += 1 - test_size
        if s + 1e-6 > 1:
            s -= 1
            classes1.append(c)
        else:
            classes2.append(c)
    if not classes1 or not classes2:
        raise ValueError("Can't split into two non-empty datasets with the given fraction.")
    return classes1, classes2


class KFoldInterleave:
    def __init__(self, n_splits):
        self._n_splits = n_splits

    def split(self, classes):
        folds = [[] for _ in range(self._n_splits)]
        for i, c in enumerate(classes):
            folds[i % self._n_splits].append(c)
        sets = []
        for i in range(self._n_splits):
            train = sum([folds[j] for j in range(self._n_splits) if j != i], [])
            test = folds[i]
            sets.append((train, test))
        return sets


class ClassSubsetDataset(Dataset):
    """Helper class for labels subset selection."""

    def __init__(self, dataset, classes):
        super().__init__()
        if max(classes) + 1 > dataset.num_classes:
            raise ValueError("More classes than dataset has")
        self._dataset = dataset
        self._indices = []
        labels = []
        classes = set(classes)
        for i, label in enumerate(dataset.labels):
            if label not in classes:
                continue
            self._indices.append(i)
            labels.append(label)

        label_mapping = {label: i for i, label in enumerate(sorted(classes))}
        self._labels = [label_mapping[label] for label in labels]

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return self._dataset.classification

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return self._dataset.openset

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).

        """
        item = self._dataset[self._indices[index]]
        return (item[0], self._labels[index]) + item[2:]


class ElementSubsetDataset(Dataset):
    """Helper class for subset selection. Allows to select a subset of indices."""

    def __init__(self, dataset, indices):
        super().__init__()

        if max(indices) + 1 > len(dataset):
            raise ValueError("More indices than dataset has.")

        self._dataset = dataset
        self._indices = indices
        self._labels = [self._dataset.labels[i] for i in self._indices]

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return self._dataset.classification

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return self._dataset.openset

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Classification dataset returns tuple (image, label).
        Verification dataset returns ((image1, image2), label).

        Datasets with quality assigned to each sample return tuples like
        (image, label, quality) or ((image1, image2), label, (quality1, quality2)).

        """
        return self._dataset[self._indices[index]]


def split_classes(dataset, fraction=0.5, interleave=False):
    """Split dataset into two parts with different sets of labels.

    Function is deterministic. Split is based on hash values, not random.

    Returns:
        Two datasets. The size of the first dataset is proportional to fraction,
        the size of the second is proportional to (1 - fraction).

    """
    classes = list(range(dataset.num_classes))
    if interleave:
        classes1, classes2 = train_test_interleave_split(classes, test_size=1 - fraction)
    else:
        classes1, classes2 = train_test_split(classes, test_size=1 - fraction, shuffle=False)
    if not classes1 or not classes2:
        raise ValueError("Can't split into two non-empty datasets with the given fraction.")
    return ClassSubsetDataset(dataset, classes1), ClassSubsetDataset(dataset, classes2)


def split_crossval_classes(dataset, i, k=4, interleave=False):
    """Get i-th training and validation sets using k class-based folds."""
    if i >= k:
        raise IndexError(i)
    classes = list(range(dataset.num_classes))
    if interleave:
        kfolder = KFoldInterleave(n_splits=k)
    else:
        kfolder = KFold(n_splits=k, shuffle=False)
    train_classes, val_classes = list(kfolder.split(classes))[i]
    return ClassSubsetDataset(dataset, train_classes), ClassSubsetDataset(dataset, val_classes)


def split_crossval_elements(dataset, i, k=4, interleave=False):
    """Get i-th training and validation sets using k element-based folds."""
    if i >= k:
        raise IndexError(i)
    indices = list(range(len(dataset)))
    if interleave:
        kfolder = KFoldInterleave(n_splits=k)
    else:
        kfolder = KFold(n_splits=k, shuffle=True, random_state=0)
    train_indices, val_indices = list(kfolder.split(indices))[i]
    train, val = ElementSubsetDataset(dataset, train_indices), ElementSubsetDataset(dataset, val_indices)
    if train.num_classes < val.num_classes:
        raise RuntimeError("The number of classes in train and test doesn't match.")
    return train, val
