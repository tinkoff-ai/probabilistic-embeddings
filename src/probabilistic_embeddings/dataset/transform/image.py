from collections import OrderedDict

from torchvision import transforms

from .cutout import Cutout
from .rotation import RandomRotation
from probabilistic_embeddings.config import prepare_config


class ImageTransform(transforms.Compose):
    """Image transform for the model."""
    @staticmethod
    def get_default_config(image_size=112,
                           center_crop=True,
                           mean=(0.5, 0.5, 0.5),
                           std=(0.5, 0.5, 0.5)):
        """Get transform config.

        Args:
            image_size: Resize and center crop image to that size.
            center_crop: Whether to make center crop or resize full image.
            mean: Mean channel stats for normalization.
            std: Std channel stats for normalization.
        """
        return OrderedDict([
            ("image_size", image_size),
            ("center_crop", center_crop),
            ("mean", mean),
            ("std", std)
        ])

    def __init__(self, *, config=None):
        config = prepare_config(self, config)

        image_transforms = []
        if config["center_crop"]:
            image_transforms.append(transforms.Resize(config["image_size"]))
            image_transforms.append(transforms.CenterCrop(config["image_size"]))
        else:
            image_transforms.append(transforms.Resize((config["image_size"], config["image_size"])))
        image_transforms.append(transforms.Normalize(mean=config["mean"], std=config["std"]))
        super().__init__(image_transforms)
        self._config = config

    @property
    def image_size(self):
        return self._config["image_size"]


class ImageTestTransform(transforms.Compose):
    """Image transform used for testing."""

    @staticmethod
    def get_default_config(prescale_size=None,
                           preserve_aspect=True):
        """Get transform config.

        Args:
            prescale_size: If specified, resize to the given size and crop to image_size.
            preserve_aspect: Whether to preserve aspect during prescaling or not.
        """
        return OrderedDict([
            ("prescale_size", prescale_size),
            ("preserve_aspect", preserve_aspect)
        ])

    def __init__(self, image_size, *, config=None):
        config = prepare_config(self, config)
        image_transforms = []
        if config["prescale_size"] is not None:
            if config["preserve_aspect"]:
                image_transforms.append(transforms.Resize(config["prescale_size"]))
            else:
                image_transforms.append(transforms.Resize((config["prescale_size"], config["prescale_size"])))
            image_transforms.append(transforms.CenterCrop(image_size))
        super().__init__(image_transforms)
        self._config = config


class ImageAugmenter(transforms.Compose):
    """Image augmenter for face recognition.

    Center crop and random flip by default.

    Args:
        image_size: Output image size.
    """
    @staticmethod
    def get_default_config(random_crop_scale=[1, 1], random_crop_ratio=[1, 1],
                           random_flip_probability=0.5,
                           brightness_range=0, contrast_range=0, saturation_range=0,
                           autoaug=None, randaug_num=2, randaug_magnitude=0, cutout_n_holes=0,
                           cutout_size=0, cutout_probability=0.5,
                           translate_ratios=(.0, .0), rotation_max_angle=.0):
        """Get augmenter config.

        Args:
            cutout_n_holes: Number of cutout patches.
            cutout_size: Cutout patch size from 0 to 1 (proportion of image size).
            cutout_probability: Probability to apply cutout augmentation.
            translate_ratios: Relative left-right and up-down shift max values.
            rotation_max_angle: Maximum absolute value (in degrees) of rotation angle.
        """
        return OrderedDict([
            ("random_crop_scale", random_crop_scale),
            ("random_crop_ratio", random_crop_ratio),
            ("random_flip_probability", random_flip_probability),
            ("brightness_range", brightness_range),
            ("contrast_range", contrast_range),
            ("saturation_range", saturation_range),
            ("autoaug", autoaug),  # None, imagenet, cifar10 or svhn.
            ("randaug_num", randaug_num),
            ("randaug_magnitude", randaug_magnitude),
            ("cutout_n_holes", cutout_n_holes),
            ("cutout_size", cutout_size),
            ("cutout_probability", cutout_probability),
            ("translate_ratios", translate_ratios),
            ("rotation_max_angle", rotation_max_angle)
        ])

    def __init__(self, image_size, *, config=None):
        self._config = prepare_config(self, config)
        augmenters = [transforms.RandomResizedCrop(image_size,
                                                   scale=self._config["random_crop_scale"],
                                                   ratio=self._config["random_crop_ratio"])]
        if self._config["autoaug"] is not None:
            policies = {
                "imagenet": transforms.AutoAugmentPolicy.IMAGENET,
                "cifar10": transforms.AutoAugmentPolicy.CIFAR10,
                "svhn": transforms.AutoAugmentPolicy.SVHN
            }
            augmenters.append(transforms.AutoAugment(policies[self._config["autoaug"]]))

        if self._config["randaug_magnitude"] > 0:
            augmenters.append(transforms.RandAugment(num_ops=self._config["randaug_num"],
                                                     magnitude=self._config["randaug_magnitude"]))

        if self._config["random_flip_probability"] > 0:
            augmenters.append(transforms.RandomHorizontalFlip(p=self._config["random_flip_probability"]))

        if self._config["brightness_range"] > 0 or self._config["contrast_range"] > 0 or \
                self._config["saturation_range"] > 0:
            augmenters.append(transforms.ColorJitter(brightness=self._config["brightness_range"],
                                                     contrast=self._config["contrast_range"],
                                                     saturation=self._config["saturation_range"]))

        if self._config["cutout_size"] > image_size:
            raise ValueError("Cutout length cannot be greater then image size.")

        if self._config["cutout_size"] > 0 and self._config["cutout_n_holes"] > 0\
                and self._config["cutout_probability"]:
            augmenters.append(Cutout(self._config["cutout_n_holes"], int(image_size * self._config["cutout_size"]),
                                     self._config["cutout_probability"]))

        if self._config["rotation_max_angle"] > 0.0:
            augmenters.append(RandomRotation(self._config["rotation_max_angle"]))

        if self._config["translate_ratios"][0] > 0.0 and self._config["translate_ratios"][1] > 0.0:
            augmenters.append(transforms.RandomAffine(0, translate=self._config["translate_ratios"]))

        super().__init__(augmenters)
