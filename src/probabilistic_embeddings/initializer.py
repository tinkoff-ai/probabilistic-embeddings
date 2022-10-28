import math
from collections import OrderedDict

import torch

from .config import prepare_config
from .layers.distribution import VMFDistribution
from .layers.classifier import VMFClassifier
from .layers.scorer import HIBScorer
from .torch import try_cuda


class Initializer:
    """Model weights and parameters initializer.

    Args:
        model: Model to initialize.
        train_loader: Train batches loader (for statistics computation in vMF-loss initializer).
    """

    INITIALIZERS = {
        "normal": torch.nn.init.normal_,
        "xavier_uniform": torch.nn.init.xavier_uniform_,
        "xavier_normal": torch.nn.init.xavier_normal_,
        "kaiming_normal": torch.nn.init.kaiming_normal_,
        "kaiming_normal_fanout": lambda tensor: torch.nn.init.kaiming_normal_(tensor, mode="fan_out")
    }

    @staticmethod
    def get_default_config(matrix_initializer=None, num_statistics_batches=10):
        """Get initializer parameters.

        Args:
            matrix_initializer: Type of matrix initialization ("normal", "xavier_uniform", "xavier_normal",
                "kaiming_normal", "kaiming_normal_fanout" or None). Use PyTorch default if None is provided.
            num_statistics_batches: Number of batches used for statitistics computation in vMF-loss initialization.
        """
        return OrderedDict([
            ("matrix_initializer", matrix_initializer),
            ("num_statistics_batches", num_statistics_batches)
        ])

    def __init__(self, *, config):
        self._config = prepare_config(self, config)

    def __call__(self, model, train_loader):
        if self._config["matrix_initializer"] is not None:
            init_fn = self.INITIALIZERS[self._config["matrix_initializer"]]
            for p in model.parameters():
                if p.ndim == 2:
                    init_fn(p)

        if model.classification and isinstance(model.classifier, VMFClassifier):
            if not isinstance(model.distribution, VMFDistribution):
                raise RuntimeError("Unexpected distribution for vMF-loss: {}.".format(type(model.distribution)))
            # Use vMF-loss embeddings scale initialization.
            # See "von Mises–Fisher Loss: An Exploration of Embedding Geometries for Supervised Learning" (2021).
            model.embedder.output_scale = 1.0
            mean_abs = self._get_mean_abs_embedding(model, train_loader, normalize=False)
            l = model.classifier.kappa_confidence
            dim = model.distribution.dim
            scale = l / (1 - l * l) * (dim - 1) / math.sqrt(dim) / mean_abs
            model.embedder.output_scale = scale

        if isinstance(model.scorer, HIBScorer):
            mean_abs = self._get_mean_abs_embedding(model, train_loader)
            model.scorer.scale.data.fill_(1 / mean_abs)

    def _get_mean_abs_embedding(self, model, train_loader, normalize=True):
        model = try_cuda(model).train()
        all_means = []
        for i, batch in enumerate(train_loader):
            if i >= self._config["num_statistics_batches"]:
                break
            images, labels = batch
            images = try_cuda(images)
            with torch.no_grad():
                distributions = model.embedder(images)  # (B, P).
                _, means, _ = model.distribution.split_parameters(distributions, normalize=normalize)
            all_means.append(means)
        means = torch.cat(all_means)  # (N, D).
        mean_abs = means.abs().mean().item()
        return mean_abs
