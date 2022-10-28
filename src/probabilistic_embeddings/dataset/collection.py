import os
from collections import OrderedDict

import torch
from torchvision import transforms

from ..config import prepare_config, ConfigError
from .cars196 import Cars196SplitClassesDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .cub200 import CUB200SplitClassesDataset
from .dummy import EmptyDataset
from .imagenette import ImagenetteDataset
from .stanforddogs import StanfordDogsDataset
from .flower102 import Flower102Dataset
from .imagenet import ImageNetDataset
from .imagenetlt import ImageNetLTDataset
from .inshop import InShopClothesDataset
from .debug import DebugDataset
from .lfw import LFWDataset, CrossLFWTestset
from .mnist import MnistDataset, MnistSplitClassesDataset
from .svhn import SVHNDataset
from .mxnet import CASIA_TESTS, MS1MV2_TESTS, MS1MV3_TESTS
from .mxnet import MXNetTrainset, MXNetValset, SerializedDataset
from .sop import SOPDataset
from .sampler import ShuffledClassBalancedBatchSampler, SameClassMixupCollator
from .transform import ImageTransform, ImageTestTransform, ImageAugmenter
from .transform import TransformDataset, RepeatDataset, PreloadDataset, SamplePairsDataset
from .transform import LossyDataset, MergedDataset, ClassMergedDataset
from .transform import split_crossval_classes, split_crossval_elements


def discard_key(mapping, key):
    mapping = mapping.copy()
    del mapping[key]
    return mapping


class DatasetCollection:
    """Dataset selector and constructor."""

    DEVSETS = {
        "casia-openset": MXNetTrainset,
        "ms1mv2-openset": MXNetTrainset,
        "ms1mv3-openset": MXNetTrainset,
        "lfw-openset": LFWDataset,
        "clfw-openset": lambda root: EmptyDataset(classification=True, openset=True),
        "lfw-joined-openset": lambda root: EmptyDataset(classification=True, openset=True),
        "cub200-openset": CUB200SplitClassesDataset,
        "cars196-openset": Cars196SplitClassesDataset,
        "cub200-interleave-openset": lambda root: CUB200SplitClassesDataset(root, interleave=True),
        "cars196-interleave-openset": lambda root: Cars196SplitClassesDataset(root, interleave=True),
        "sop-openset": SOPDataset,
        "inshop-openset": InShopClothesDataset,
        "mnist-openset": MnistSplitClassesDataset,
        "imagenette": ImagenetteDataset,
        "tinyimagenet": ImagenetteDataset,
        "imagenet": ImageNetDataset,
        "stanforddogs": StanfordDogsDataset,
        "flower102": Flower102Dataset,
        "imagenetlt": ImageNetLTDataset,
        "cifar10": CIFAR10Dataset,
        "cifar100": CIFAR100Dataset,
        "mnist": MnistDataset,
        "svhn": SVHNDataset,
        # TODO: Add closed-set serialized and debug when needed.
        "serialized-openset": lambda root: SerializedDataset.from_folder(root)["train"],
        "debug-openset": DebugDataset
    }

    VALSETS = {
        "flower102": lambda root: Flower102Dataset(root, annotation_key="valid"),
        "imagenetlt": lambda root: ImageNetLTDataset(root, mode="val")
    }

    TESTSETS = {
        "casia-openset": lambda root: OrderedDict([(name, MXNetValset(os.path.join(root, name + ".bin"))) for name in CASIA_TESTS]),
        "ms1mv2-openset": lambda root: OrderedDict([(name, MXNetValset(os.path.join(root, name + ".bin"))) for name in MS1MV2_TESTS]),
        "ms1mv3-openset": lambda root: OrderedDict([(name, MXNetValset(os.path.join(root, name + ".bin"))) for name in MS1MV3_TESTS]),
        "lfw-openset": lambda root: LFWDataset(root, train=False, classification=False),
        "clfw-openset": CrossLFWTestset,
        "lfw-joined-openset": lambda root: ClassMergedDataset(LFWDataset(root), LFWDataset(root, train=False)),
        "cub200-openset": lambda root: CUB200SplitClassesDataset(root, train=False),
        "cars196-openset": lambda root: Cars196SplitClassesDataset(root, train=False),
        "cub200-interleave-openset": lambda root: CUB200SplitClassesDataset(root, train=False, interleave=True),
        "cars196-interleave-openset": lambda root: Cars196SplitClassesDataset(root, train=False, interleave=True),
        "sop-openset": lambda root: SOPDataset(root, train=False),
        "inshop-openset": lambda root: InShopClothesDataset(root, train=False),
        "mnist-openset": lambda root: MnistSplitClassesDataset(root, train=False),
        "imagenette": lambda root: ImagenetteDataset(root, train=False),
        "tinyimagenet": lambda root: ImagenetteDataset(root, train=False),
        "imagenet": lambda root: ImageNetDataset(root, train=False),
        "stanforddogs": lambda root: StanfordDogsDataset(root, train=False),
        "flower102": lambda root: Flower102Dataset(root, annotation_key="tstid"),
        "imagenetlt": lambda root: {
            "imagenetlt-overall": ImageNetLTDataset(root, mode="test", test_setup="overall"),
            "imagenetlt-many-shot": ImageNetLTDataset(root, mode="test", test_setup="many-shot"),
            "imagenetlt-medium-shot": ImageNetLTDataset(root, mode="test", test_setup="medium-shot"),
            "imagenetlt-few-shot": ImageNetLTDataset(root, mode="test", test_setup="few-shot"),
        },
        "cifar10": lambda root: CIFAR10Dataset(root, train=False),
        "cifar100": lambda root: CIFAR100Dataset(root, train=False),
        "mnist": lambda root: MnistDataset(root, train=False),
        "svhn": lambda root: SVHNDataset(root, split="test"),
        "serialized-openset": lambda root: discard_key(SerializedDataset.from_folder(root), "train"),
        "debug-openset": lambda root: DebugDataset(root, train=False)
    }

    MIXUP = {
        "same_class": SameClassMixupCollator
    }

    @staticmethod
    def get_default_config(name=None,
                           validation_fold=None, num_validation_folds=4, validation_split_interleave=False,
                           transform_params=None, transform_test_params=None, augmenter_params=None,
                           mixup_type=None,
                           batch_size=256, samples_per_class=4, uniform_sampling=False,
                           num_workers=8, num_valid_workers=None, persistent_workers=False,
                           shuffle_train=True, train_repeat=1, preload=False,
                           add_lossy_valsets=False, add_lossy_testsets=False, lossy_params=None,
                           add_verification_valsets=False, add_verification_testsets=True, validate_on_test=False):
        """Get collection parameters.

        Args:
            name: Type of the training dataset (`casia`, `ms1mv2`, `ms1mv3`, `lfw`, `cub200`, `cars196` or `sop`).
            validation_fold: Fold index used for validation.
            num_validation_folds: Number of validation splits.
            validation_split_interleave: If True, use interleave splitting scheme. Split using segments otherwise.
            transform_params: Parameters of :class:`ImageTransform`.
            transform_test_params: Parameters of :class:`ImageTestTransform` used during testing.
            augmenter_params: Parameters of :class:`ImageAugmenter` used during training.
            mixup_type: Type of mixup strategy for classification datasets. (None or "same_class").
            batch_size: Batch size.
            samples_per_class: If not None, sample classes uniformly with the given number of samples per class.
            uniform_sampling: If true and samples_per_class is not None, classes are sampled uniformly for each batch.
            num_workers: Number of loader workers.
            num_valid_workers: Number of workers used for validation. Set None to use the same number as in train.
            persistent_workers: Keep loader workers alive after iteration.
            shuffle_train: Whether to shuffle train or not.
            train_repeat: Number of training set repetition during epoch (useful for small datasets).
            preload: Load full dataset to the memory before training.
            add_lossy_valsets: Add lossy variants of validation sets.
            add_lossy_testsets: Add lossy variants of test sets.
            lossy_params: Parameters of lossy datasets.
            add_verification_valsets: Whether to add verification validation sets in addition to classification.
            add_verification_testsets: Whether to add verification testsets in addition to classification.
            validate_on_test: Compute test metrics between epochs.
        """
        return OrderedDict([
            ("name", name),
            ("validation_fold", validation_fold),
            ("num_validation_folds", num_validation_folds),
            ("validation_split_interleave", validation_split_interleave),
            ("transform_params", transform_params),
            ("transform_test_params", transform_test_params),
            ("augmenter_params", augmenter_params),
            ("mixup_type", mixup_type),
            ("batch_size", batch_size),
            ("samples_per_class", samples_per_class),
            ("uniform_sampling", uniform_sampling),
            ("num_workers", num_workers),
            ("persistent_workers", persistent_workers),
            ("num_valid_workers", num_valid_workers),
            ("shuffle_train", shuffle_train),
            ("train_repeat", train_repeat),
            ("preload", preload),
            ("add_lossy_valsets", add_lossy_valsets),
            ("add_lossy_testsets", add_lossy_testsets),
            ("lossy_params", lossy_params),
            ("add_verification_testsets", add_verification_testsets),
            ("add_verification_valsets", add_verification_valsets),
            ("validate_on_test", validate_on_test)
        ])

    def __init__(self, data_root, *, config):
        self._config = prepare_config(self, config)
        if self._config["name"] is None:
            raise ConfigError("Dataset type must be provided")

        self._data_root = data_root
        self._image_transform = ImageTransform(config=self._config["transform_params"])
        self._image_test_transform = ImageTestTransform(self.image_size, config=self._config["transform_test_params"])
        self._augmenter = ImageAugmenter(self.image_size, config=self._config["augmenter_params"])

        trainset = self.get_trainset(transform=False)
        self._num_classes = trainset.num_classes
        self._openset = trainset.openset
        self._priors = trainset.priors

    @property
    def image_size(self):
        """Get dataset image size."""
        return self._image_transform.image_size

    @property
    def num_train_classes(self):
        """Get total number of classes in train."""
        return self._num_classes

    @property
    def openset(self):
        return self._openset

    @property
    def train_priors(self):
        """Get array of trainset class priors."""
        return self._priors

    @property
    def validation_fold(self):
        return self._config["validation_fold"]

    def get_trainset(self, transform=True, augment=True):
        """Get training dataset."""
        dataset = self.DEVSETS[self._config["name"]](self._data_root)
        if self._config["validation_fold"] is not None:
            if self._config["name"] in self.VALSETS:
                raise ConfigError("`validation_fold` is not None. Cannot perform validation split,"
                                  "because this dataset has author's validation split.")
            if dataset.openset:
                dataset = split_crossval_classes(dataset,
                                                 i=self._config["validation_fold"],
                                                 k=self._config["num_validation_folds"],
                                                 interleave=self._config["validation_split_interleave"])[0]

            else:
                dataset = split_crossval_elements(dataset,
                                                  i=self._config["validation_fold"],
                                                  k=self._config["num_validation_folds"],
                                                  interleave=self._config["validation_split_interleave"])[0]
        if transform:
            if augment:
                transform = transforms.Compose([self._augmenter, transforms.ToTensor(), self._image_transform])
            else:
                transform = transforms.Compose([transforms.ToTensor(), self._image_transform])
            if self._config["preload"]:
                dataset = PreloadDataset(dataset, image_size=int(self.image_size * 1.5),
                                         num_workers=self._config["num_workers"])
            dataset = RepeatDataset(dataset, self._config["train_repeat"])
            dataset = TransformDataset(dataset, transform)
        return dataset

    def get_valsets(self, transform=True):
        """Get validation datasets. Returns None if not available."""

        if self._config["validation_fold"] is not None:
            if self._config["name"] in self.VALSETS:
                raise ConfigError("`validation_fold` is not None. Cannot perform validation split,"
                                  "because this dataset has author's validation split.")
            dataset = self.DEVSETS[self._config["name"]](self._data_root)
            if dataset.openset:
                dataset = split_crossval_classes(dataset,
                                                 i=self._config["validation_fold"],
                                                 k=self._config["num_validation_folds"],
                                                 interleave=self._config["validation_split_interleave"])[1]

            else:
                dataset = split_crossval_elements(dataset,
                                                  i=self._config["validation_fold"],
                                                  k=self._config["num_validation_folds"],
                                                  interleave=self._config["validation_split_interleave"])[1]
        elif self._config["name"] in self.VALSETS:
            dataset = self.VALSETS[self._config["name"]](self._data_root)
        else:
            return {}

        base_valsets = {"valid": dataset}
        if self._config["add_lossy_valsets"]:
            for name, dataset in list(base_valsets.items()):
                if dataset.classification:
                    base_valsets[name + "-lossy"] = LossyDataset(dataset, config=self._config["lossy_params"])

        valsets = OrderedDict()
        for name, dataset in base_valsets.items():
            if transform:
                transform = transforms.Compose([transforms.ToTensor(), self._image_test_transform, self._image_transform])
                if self._config["preload"]:
                    dataset = PreloadDataset(dataset, image_size=self.image_size,
                                             num_workers=self._config["num_workers"])
                dataset = TransformDataset(dataset, transform)
            valsets[name] = dataset
            if dataset.classification and self._config["add_verification_valsets"]:
                valsets[name + "-pairs"] = SamplePairsDataset(dataset)
        return valsets

    def get_testsets(self, transform=True):
        """Get dictionary of testsets."""
        if self._config["name"] not in self.TESTSETS:
            return {}
        base_testsets = self.TESTSETS[self._config["name"]](self._data_root)
        if not isinstance(base_testsets, (dict, OrderedDict)):
            base_testsets = {self._config["name"]: base_testsets}
        base_testsets = {"infer-" + k: v for k, v in base_testsets.items()}
        if self._config["add_lossy_testsets"]:
            for name, dataset in list(base_testsets.items()):
                if dataset.classification:
                    base_testsets[name + "-lossy"] = LossyDataset(dataset, config=self._config["lossy_params"])
        testsets = OrderedDict()
        for name, dataset in base_testsets.items():
            if transform:
                transform = transforms.Compose([transforms.ToTensor(),
                                                self._image_test_transform, self._image_transform])
                if self._config["preload"]:
                    dataset = PreloadDataset(dataset, image_size=self.image_size,
                                             num_workers=self._config["num_workers"])
                dataset = TransformDataset(dataset, transform)
            testsets[name] = dataset
            if dataset.classification and self._config["add_verification_testsets"]:
                testsets[name + "-pairs"] = SamplePairsDataset(dataset)
        return testsets

    def get_datasets(self, train=True, transform=True, augment_train=True):
        """Get datasets dictionary.

        Args:
            train: Whether to make training set or not.
            transform: Whether to apply transforms or not.

        """
        datasets = OrderedDict()
        if train:
            datasets["train"] = self.get_trainset(transform=transform, augment=augment_train)
        datasets.update(self.get_valsets(transform=transform))
        if (not train) or self._config["validate_on_test"]:
            datasets.update(self.get_testsets(transform=transform))
        return datasets

    def get_loaders(self, train=True, transform=True, augment_train=True):
        """Get dataset loaders."""
        datasets = self.get_datasets(train=train, transform=transform, augment_train=augment_train)
        loaders = OrderedDict([
            (name, self._get_loader(dataset, train=(name == "train")))
            for name, dataset in datasets.items()
        ])
        return loaders

    def _get_loader(self, dataset, train):
        kwargs = {}
        num_workers = self._config["num_workers"]
        if (not train) and (self._config["num_valid_workers"] is not None):
            num_workers = self._config["num_valid_workers"]
        batch_size = self._config["batch_size"]
        if train and dataset.classification and (self._config["samples_per_class"] is not None):
            if not self._config["shuffle_train"]:
                raise ValueError("Balanced sampling requires shuffling.")
            kwargs["batch_sampler"] = ShuffledClassBalancedBatchSampler(dataset,
                                                                        batch_size=batch_size,
                                                                        samples_per_class=self._config["samples_per_class"],
                                                                        uniform=self._config["uniform_sampling"])
        else:
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = train
            kwargs["shuffle"] = self._config["shuffle_train"] if train else False
        if train and (self._config["mixup_type"] is not None):
            kwargs["collate_fn"] = self.MIXUP[self._config["mixup_type"]]()
        return torch.utils.data.DataLoader(dataset,
                                           num_workers=num_workers,
                                           pin_memory=torch.cuda.device_count() > 0,
                                           persistent_workers=self._config["persistent_workers"],
                                           **kwargs)
