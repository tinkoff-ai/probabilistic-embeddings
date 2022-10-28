import os
import pickle
from collections import defaultdict

import jpeg4py
import mxnet as mx
import numpy as np
from PIL import Image

from ..io import read_yaml, write_yaml
from .common import Dataset


CASIA_TESTS = ["lfw", "cfp_fp", "cfp_ff", "calfw", "cplfw", "agedb_30"]
MS1MV2_TESTS = ["lfw", "cfp_fp", "cfp_ff", "calfw", "cplfw", "agedb_30", "vgg2_fp"]
MS1MV3_TESTS = ["lfw", "cfp_fp", "cfp_ff", "calfw", "cplfw", "agedb_30", "vgg2_fp"]


def imdecode(packed_image):
    return jpeg4py.JPEG(np.frombuffer(packed_image, dtype=np.uint8)).decode()


class MXNetTrainset(Dataset):
    """PyTorch interface to MXNet serialized training dataset.

    Args:
        root: Path to the dataset root with images and annotations.
    """

    INDEX_FILENAME = "train.idx"
    DATA_FILENAME = "train.rec"
    META_FILENAME = "property"
    LABELS_FILENAME = "labels.yaml"  # Use labels cache to speedup initialization.

    def __init__(self, root):
        super().__init__()
        self._root = root

        with open(os.path.join(root, self.META_FILENAME)) as fp:
            self._num_classes = int(fp.read().strip().split(",")[0])

        self._reader = mx.recordio.MXIndexedRecordIO(
            os.path.join(root, self.INDEX_FILENAME),
            os.path.join(root, self.DATA_FILENAME),
            "r"
        )
        self._header_end = 1
        self._images_end, self._pairs_end = map(int, self._get_record(0)[1])
        self._num_images = self._images_end - self._header_end

        try:
            labels_path = os.path.join(root, self.LABELS_FILENAME)
            self._labels = np.array(read_yaml(labels_path))
        except FileNotFoundError:
            self._labels = self._get_labels()

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
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns:
            Tuple (image, label).

        """
        image, label = self._get_record(index + self._header_end)
        return image, int(label)

    def dump_labels_cache(self):
        """Dump labels to dataset folder."""
        labels_path = os.path.join(self._root, self.LABELS_FILENAME)
        write_yaml(self._labels.tolist(), labels_path)

    def _get_labels(self):
        labels = []
        for i in range(self._num_images):
            record = self._reader.read_idx(i + self._header_end)
            header, image = mx.recordio.unpack(record)
            labels.append(int(header.label))
        return np.asarray(labels)

    def _get_record(self, i):
        record = self._reader.read_idx(i)
        header, image = mx.recordio.unpack(record)
        if len(image) > 0:
            image = imdecode(image)
        return image, header.label


class MXNetValset(Dataset):
    """PyTorch interface to MXNet pairs validation dataset.

    Args:
        filename: Path to the binary.
    """

    def __init__(self, filename):
        super().__init__()
        with open(filename, "rb") as fp:
            images, labels = pickle.load(fp, encoding="bytes")
            image_shape = imdecode(images[0]).shape
            images = np.stack([imdecode(image) for image in images]).reshape((len(labels), 2, *image_shape))
        self._images = images
        self._labels = labels

    @property
    def classification(self):
        """Whether dataset is classification or matching."""
        return False

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].
        """
        return self._labels

    def __getitem__(self, index):
        """Get element of the dataset.

        Returns:
            Tuple ((image1, image2), label).

        """
        image1, image2 = self._images[index]
        label = self._labels[index]
        return (image1, image2), label


class SerializedDataset(Dataset):
    """MXNet-serialized dataset."""
    def __init__(self, index_path):
        super().__init__()
        prefix = os.path.splitext(index_path)[0]
        self._meta = read_yaml(prefix + ".yaml")
        self._labels = read_yaml(prefix + ".labels")

        self._reader = mx.recordio.MXIndexedRecordIO(
            prefix + ".idx",
            prefix + ".rec",
            "r"
        )

    @property
    def classification(self):
        """Whether dataset is classification or verification."""
        return self._meta["classification"]

    @property
    def openset(self):
        """Whether dataset is for open-set or closed-set classification."""
        return True

    @property
    def labels(self):
        """Get dataset labels array.

        Labels are integers in the range [0, N-1].

        """
        return self._labels

    def __getitem__(self, index):
        record = self._reader.read_idx(index)
        header, image = mx.recordio.unpack(record)
        image = imdecode(image)
        return Image.fromarray(image), int(header.label)

    @staticmethod
    def from_folder(root):
        datasets = defaultdict(dict)
        for filename in os.listdir(root):
            base, ext = os.path.splitext(filename)
            if ext.lower() in {".idx", ".labels", ".yaml", ".rec"}:
                datasets[base][ext.lower()] = filename
        datasets = {k: v for k, v in datasets.items() if ".rec" in v}
        if "train" not in datasets:
            raise FileNotFoundError("Can't find trainset in {}.".format(root))
        return {k: SerializedDataset(os.path.join(root, v[".idx"])) for k, v in datasets.items()}
