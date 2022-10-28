import numpy as np

import pretrainedmodels
import torch
import torchvision

from ...third_party import efficientnet, ModelM3
from ...third_party import PyramidNet as PyramidNetImpl
from ...third_party.cotraining.model import resnet as cotraining
from . import cgd
from . import bn_inception_simple


class ResNetModel(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        self._model = getattr(torchvision.models, name)(pretrained=pretrained)
        self._channels = self._model.fc.in_features
        self._model.avgpool = torch.nn.Identity()
        self._model.fc = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return 224

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return [0.229, 0.224, 0.225]

    def forward(self, input):
        x = input
        x = self._model.conv1(x)
        x = self._model.bn1(x)
        x = self._model.relu(x)
        x = self._model.maxpool(x)

        x = self._model.layer1(x)
        x = self._model.layer2(x)
        x = self._model.layer3(x)
        x = self._model.layer4(x)
        return x


class TorchVGGModel(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        self._model = getattr(torchvision.models, name)(pretrained=pretrained)
        self._channels = self._model.features[-3].out_channels
        self._model.avgpool = torch.nn.Identity()
        self._model.classifier = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return 224

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return [0.229, 0.224, 0.225]

    def forward(self, input):
        x = input
        x = self._model.features(x)

        return x


class CotrainingModel(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        if pretrained:
            raise ValueError("Pretrained co-training models are not available.")
        super().__init__()
        self._model = getattr(cotraining, name)()
        self._channels = self._model.fc.in_features
        self._model.avgpool = torch.nn.Identity()
        self._model.fc = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return 224

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return [0.229, 0.224, 0.225]

    def forward(self, input):
        x = input
        x = self._model.layer0(x)
        x = self._model.layer1(x)
        x = self._model.layer2(x)
        x = self._model.layer3(x)
        if self._model.layer4 is not None:
            x = self._model.layer4(x)
        return x


class EfficientNet(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        self._model = getattr(efficientnet, name)(pretrained=pretrained)
        self._channels = self._model.classifier[-1].in_features
        self._model.avgpool = torch.nn.Identity()
        self._model.classifier = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return 224

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return [0.229, 0.224, 0.225]

    def forward(self, input):
        return self._model.features(input)


class PyramidNet(PyramidNetImpl):
    def __init__(self, dataset, depth, alpha, pretrained=False):
        if pretrained:
            raise NotImplementedError("No pretrained PyramidNet available.")
        super().__init__(dataset, depth=depth, alpha=alpha, num_classes=1)
        self._channels = self.fc.in_features
        self.avgpool = torch.nn.Identity()
        self.fc = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return 224

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return [0.229, 0.224, 0.225]

    def forward(self, input):
        return super().features(input)


class PMModel(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        super().__init__()
        pretrained = "imagenet" if pretrained else None
        self._model = self._get_model(name, pretrained=pretrained)
        self._channels = self._model.last_linear.in_features
        self._model.global_pool = torch.nn.Identity()
        self._model.last_linear = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        try:
            return self._model.input_size[1]
        except AttributeError:
            raise RuntimeError("Input size is available only for pretrained models.")

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        scale = self._model.input_range[1]
        result = np.array(self._model.mean) / scale
        result = self._adjust_colorspace(result[None])[0]
        return result

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        scale = self._model.input_range[1]
        result = np.array(self._model.std) / scale
        result = self._adjust_colorspace(result[None])[0]
        return result

    def forward(self, input):
        x = self._adjust_colorspace(input)
        x = self._model.features(x)
        return x

    def _get_model(self, name, pretrained):
        return getattr(pretrainedmodels, name)(num_classes=1000, pretrained=pretrained)

    def _adjust_colorspace(self, input):
        if input.shape[1] != 3:
            raise ValueError("Bad input shape")
        if self._model.input_space == "RGB":
            return input
        assert self._model.input_space == "BGR"
        return input.flip(1)


class VGG(torch.nn.Module):
    def __init__(self, name, pretrained=False):
        super(VGG, self).__init__()
        if pretrained:
            raise ValueError("Pretrained weights are not available for VGG model.")
        if name == "M3":
            self._model = ModelM3()
        else:
            raise ValueError(f"Model name {name} is not available for VGG models.")
        self._channels = self._model.conv10.out_channels
        self._model.global_pool = torch.nn.Identity()
        self._model.last_linear = torch.nn.Identity()

    @property
    def input_size(self):
        """Pretrained model input image size."""
        return 28

    @property
    def channels(self):
        """Number of output channels."""
        return self._channels

    @property
    def mean(self):
        """Pretrained model input normalization mean."""
        return [0.5, 0.5, 0.5]

    @property
    def std(self):
        """Pretrained model input normalization STD."""
        return [1.0, 1.0, 1.0]

    def forward(self, input):
        return self._model.get_embedding(input)


class CGDModel(PMModel):
    def _get_model(self, name, pretrained):
        return getattr(cgd, name)(num_classes=1000, pretrained=pretrained)


class PAModel(PMModel):
    def _get_model(self, name, pretrained):
        if name != "bn_inception_simple":
            raise ValueError("Unknown model {}.".format(name))
        return getattr(bn_inception_simple, name)(pretrained=bool(pretrained))
