import torch
from pretrainedmodels.models.senet import SENet, SEResNetBottleneck, pretrained_settings
from torch.utils import model_zoo


class CGDSENet(SENet):
    """Implementation of CGD network with multiple global pooling branches.

    See original paper:
        Combination of Multiple Global Descriptors for Image Retrieval (2019).
    """
    def __init__(self, block, layers, groups, reduction, dropout_p=None,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        super().__init__(block=block, layers=layers, groups=groups, reduction=reduction, dropout_p=dropout_p,
                         inplanes=inplanes, input_3x3=input_3x3, downsample_kernel_size=downsample_kernel_size,
                         downsample_padding=downsample_padding, num_classes=num_classes)
        # Remove downsampling before the last block of layers.
        self.layer4[0].conv1.stride_ = (1, 1)
        self.layer4[0].downsample[0].stride_ = (1, 1)


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings["num_classes"], \
        "num_classes should be {}, but is {}".format(
            settings["num_classes"], num_classes)
    checkpoint = model_zoo.load_url(settings["url"])
    # Don't init last layer since its size is different from checkpoint.
    checkpoint["last_linear.weight"] = model.state_dict()["last_linear.weight"]
    checkpoint["last_linear.bias"] = model.state_dict()["last_linear.bias"]
    model.load_state_dict(checkpoint)


def cgd_se_resnet50(num_classes=1000, pretrained="imagenet"):
    model = CGDSENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                     dropout_p=None, inplanes=64, input_3x3=False,
                     downsample_kernel_size=1, downsample_padding=0,
                     num_classes=num_classes)
    settings = pretrained_settings["se_resnet50"][pretrained]
    if pretrained is not None:
        initialize_pretrained_model(model, num_classes, settings)
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]
    return model
