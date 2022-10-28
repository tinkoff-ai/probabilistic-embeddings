from collections import OrderedDict

import torch

from ..config import prepare_config
from ..third_party import SAM as SAMImpl


class SGDOptimizer(torch.optim.SGD):
    """Configurable SGD."""

    @staticmethod
    def get_default_config(lr=0.1, momentum=0.9, weight_decay=5e-4):
        """Get optimizer parameters."""
        return OrderedDict([
            ("lr", lr),
            ("momentum", momentum),
            ("weight_decay", weight_decay)
        ])

    def __init__(self, parameters, *, config=None):
        self._config = prepare_config(self, config)
        super().__init__(parameters,
                         lr=self._config["lr"],
                         momentum=self._config["momentum"],
                         weight_decay=self._config["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        return super().step()


class RMSpropOptimizer(torch.optim.RMSprop):
    """Configurable RMSprop."""

    @staticmethod
    def get_default_config(lr=0.1, momentum=0.9, weight_decay=5e-4):
        """Get optimizer parameters."""
        return OrderedDict([
            ("lr", lr),
            ("momentum", momentum),
            ("weight_decay", weight_decay)
        ])

    def __init__(self, parameters, *, config=None):
        self._config = prepare_config(self, config)
        super().__init__(parameters,
                         lr=self._config["lr"],
                         momentum=self._config["momentum"],
                         weight_decay=self._config["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        return super().step()


class AdamOptimizer(torch.optim.Adam):
    """Configurable Adam."""

    @staticmethod
    def get_default_config(lr=0.1, weight_decay=5e-4):
        """Get optimizer parameters."""
        return OrderedDict([
            ("lr", lr),
            ("weight_decay", weight_decay)
        ])

    def __init__(self, parameters, *, config=None):
        self._config = prepare_config(self, config)
        super().__init__(parameters,
                         lr=self._config["lr"],
                         weight_decay=self._config["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        return super().step()


class AdamWOptimizer(torch.optim.AdamW):
    """Configurable AdamW."""

    @staticmethod
    def get_default_config(lr=0.1, weight_decay=5e-4):
        """Get optimizer parameters."""
        return OrderedDict([
            ("lr", lr),
            ("weight_decay", weight_decay)
        ])

    def __init__(self, parameters, *, config=None):
        self._config = prepare_config(self, config)
        super().__init__(parameters,
                         lr=self._config["lr"],
                         weight_decay=self._config["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        return super().step()


class SamOptimizer(SAMImpl):
    BASE_OPTIMIZERS = {
        "sgd": SGDOptimizer,
        "rmsprop": RMSpropOptimizer,
        "adam": AdamOptimizer,
        "adamw": AdamWOptimizer
    }

    @staticmethod
    def get_default_config(rho=0.5, adaptive=True, base_type="sgd", base_params=None,
                           adaptive_bias_and_bn=False):
        """Get optimizer parameters."""
        return OrderedDict([
            ("rho", rho),
            ("adaptive", adaptive),
            ("base_type", base_type),
            ("base_params", base_params),
            ("adaptive_bias_and_bn", adaptive_bias_and_bn)
        ])

    def __init__(self, parameters, *, config=None):
        config = prepare_config(self, config)
        if not config["adaptive_bias_and_bn"]:
            parameters = self._split_bias_and_bn_groups(parameters, {"adaptive": False})

        super().__init__(parameters, self.BASE_OPTIMIZERS[config["base_type"]],
                         config=config["base_params"],
                         rho=config["rho"], adaptive=config["adaptive"])

    @staticmethod
    def _split_bias_and_bn_groups(parameters, bias_and_bn_params):
        """Split each parameter groups into two parts with tensors of rank > 1 and tensors of rank <= 1.
        Apply extra parameters for those tensors with rank <= 1."""
        parameters = list(parameters)
        if not isinstance(parameters[0], dict):
            parameters = [{"params": parameters}]
        new_parameters = []
        for group in parameters:
            nbn_group = dict(group)
            bn_group = dict(group)
            bn_group.update(bias_and_bn_params)
            nbn_group["params"] = []
            bn_group["params"] = []
            for p in group["params"]:
                if p.ndim > 1:
                    nbn_group["params"].append(p)
                else:
                    bn_group["params"].append(p)
            if nbn_group["params"]:
                new_parameters.append(nbn_group)
            if bn_group["params"]:
                new_parameters.append(bn_group)
        return new_parameters
