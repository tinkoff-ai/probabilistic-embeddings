# Datasets with classification testset.
from .cars196 import Cars196Dataset, Cars196SplitClassesDataset
from .cifar import CIFAR10Dataset, CIFAR100Dataset
from .cub200 import CUB200Dataset, CUB200SplitClassesDataset
from .sop import SOPDataset
from .imagenette import ImagenetteDataset
from .inshop import InShopClothesDataset
from .stanforddogs import StanfordDogsDataset
from .flower102 import Flower102Dataset
from .imagenet import ImageNetDataset
from .imagenetlt import ImageNetLTDataset
from .mnist import MnistDataset, MnistSplitClassesDataset

# Datasets with verification testset.
from .lfw import LFWDataset, CrossLFWTestset

# Datasets in serialized MxNet format.
from .mxnet import MXNetTrainset, MXNetValset, SerializedDataset
from .mxnet import CASIA_TESTS, MS1MV2_TESTS, MS1MV3_TESTS

# Transforms.
from .transform import ImageTransform, ImageTestTransform, ImageAugmenter
from .transform import TransformDataset, RepeatDataset
from .transform import MergedDataset, PreloadDataset, SamplePairsDataset
from .transform import split_classes, split_crossval_classes

# Main class.
from .collection import DatasetCollection
