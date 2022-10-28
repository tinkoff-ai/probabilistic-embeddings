from collections import OrderedDict

import torch

from .common import non_diag
from ..config import prepare_config, ConfigError


class ProxyNCALoss:
    """Implementation of the Proxy-NAC loss with custom scorer.

    For details see original paper:
    https://openaccess.thecvf.com/content_ICCV_2017/papers/Movshovitz-Attias_No_Fuss_Distance_ICCV_2017_paper.pdf
    """

    @staticmethod
    def get_default_config():
        """Get default config."""
        return OrderedDict([])

    def __init__(self, *, config=None, aggregation="mean"):
        self._config = prepare_config(self, config)
        self._aggregation = aggregation

    def __call__(self, embeddings, labels, target_embeddings, scorer):
        if embeddings.ndim != 2:
            raise ValueError("Expected embeddings with shape (B, D), got {}".format(embeddings.shape))
        distances = -scorer(embeddings[:, None, :], target_embeddings[None, :, :])  # (B, C).
        deltas = distances.take_along_dim(labels.unsqueeze(-1), -1) - distances
        mask = torch.ones_like(deltas, dtype=torch.bool)
        mask.scatter_(-1, labels.unsqueeze(-1), False)
        deltas = deltas[mask].reshape(len(labels), len(target_embeddings) - 1)  # (B, C - 1).
        losses = torch.logsumexp(deltas, dim=-1)  # (B).
        if self._aggregation == "none":
            return losses
        elif self._aggregation == "mean":
            return losses.mean()
        else:
            raise ValueError("Unknown aggregation: {}".format(self._aggregation))
