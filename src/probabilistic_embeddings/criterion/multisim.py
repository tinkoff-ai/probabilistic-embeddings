from collections import OrderedDict

import torch

from .common import non_diag
from ..config import prepare_config, ConfigError


class MultiSimilarityLoss:
    """Implementation of the Multi-similarity loss with custom scorer.

    For details see original paper:
    https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf

    Implementation was largely motivited by:
    https://github.com/msight-tech/research-ms-loss/blob/master/ret_benchmark/losses/multi_similarity_loss.py
    """

    @staticmethod
    def get_default_config(threshold=0.5, margin=0.1,
                           positive_scale=2.0, negative_scale=40.0):
        """Get default config."""
        return OrderedDict([
            ("threshold", threshold),
            ("margin", margin),
            ("positive_scale", positive_scale),
            ("negative_scale", negative_scale)
        ])

    def __init__(self, *, config=None, aggregation="mean"):
        self._config = prepare_config(self, config)
        self._aggregation = aggregation

    def __call__(self, embeddings, labels, scorer):
        if embeddings.shape[:-1] != labels.shape:
            raise ValueError("Embeddings and labels shape mismatch")
        prefix = tuple(embeddings.shape[:-1])
        dim = embeddings.shape[-1]
        embeddings = embeddings.reshape(-1, dim)  # (B, D).
        labels = labels.flatten()
        all_scores = non_diag(scorer(embeddings[:, None, :], embeddings[None, :, :]))  # (B, B - 1).
        all_same = non_diag(labels[:, None] == labels[None, :])  # (B, B - 1).
        zero_loss = 0 * embeddings.flatten()[0]

        losses = []
        for same, scores in zip(all_same, all_scores):
            positive_scores = scores[same]
            negative_scores = scores[~same]
            if len(negative_scores) == 0 or len(positive_scores) == 0:
                losses.append(zero_loss)
                continue

            selected_negative_scores = negative_scores[negative_scores + self._config["margin"] > min(positive_scores)]
            selected_positive_scores = positive_scores[positive_scores - self._config["margin"] < max(negative_scores)]

            if len(selected_negative_scores) == 0 or len(selected_positive_scores) == 0:
                losses.append(zero_loss)
                continue
            positive_loss = 1.0 / self._config["positive_scale"] * torch.log(
                1 + torch.sum(torch.exp(-self._config["positive_scale"] * (selected_positive_scores - self._config["threshold"]))))
            negative_loss = 1.0 / self._config["negative_scale"] * torch.log(
                1 + torch.sum(torch.exp(self._config["negative_scale"] * (selected_negative_scores - self._config["threshold"]))))
            losses.append(positive_loss + negative_loss)
        losses = torch.stack(losses)
        if self._aggregation == "none":
            return losses
        elif self._aggregation == "mean":
            return losses.mean()
        else:
            raise ValueError("Unknown aggregation: {}".format(self._aggregation))
