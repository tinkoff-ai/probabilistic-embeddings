from collections import OrderedDict
from typing import Tuple, Dict

import numpy as np
import torch
from catalyst.core.runner import IRunner
from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.utils.distributed import all_gather, get_rank
from catalyst.utils.misc import get_attr
from scipy.stats import spearmanr

from ..config import prepare_config, ConfigError
from ..torch import get_base_module
from .nearest import NearestNeighboursMetrics


class DummyMetric(ICallbackLoaderMetric):
    """No metric."""
    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)

    def reset(self, num_batches, num_samples) -> None:
        pass

    def update(self) -> None:
        pass

    def compute(self):
        return tuple()

    def compute_key_value(self):
        return {}


class NearestMetric(ICallbackLoaderMetric):
    """Metric interface to MAP@R and GAP@R."""

    def __init__(self, config: Dict = None,
                 compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self._config = dict(config)
        self._filter_out_bins = self._config.pop("num_filter_out_bins", 0)
        self._distribution = None
        self._scorer = None
        self.reset(None, None)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution

    @property
    def scorer(self):
        return self._scorer

    @scorer.setter
    def scorer(self, scorer):
        self._scorer = scorer

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._embeddings = []
        self._targets = []

    def update(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            embeddings: Tensor with embeddings distribution.
            targets: Tensor with target labels.
        """
        self._embeddings.append(embeddings.detach())
        self._targets.append(targets.detach())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes nearest neighbours metrics."""
        if self.scorer is None:
            raise RuntimeError("Scorer must be set on stage_start.")
        if self.distribution is None:
            raise RuntimeError("Distribution must be set on stage_start.")
        metric = NearestNeighboursMetrics(self._distribution, self._scorer, config=self._config)
        embeddings = torch.cat(self._embeddings)
        targets = torch.cat(self._targets)

        if self._is_ddp:
            embeddings = torch.cat(all_gather(embeddings)).detach()
            targets = torch.cat(all_gather(targets)).detach()

        metrics = metric(embeddings, targets)

        if self.distribution.has_confidences:
            confidences = self.distribution.confidences(embeddings)
            values = torch.sort(confidences)[0]
            # Compute filter-out metrics.
            for fraction in np.linspace(0, 0.9, self._filter_out_bins):
                name = "{:.3f}".format(fraction)
                th = values[int(round((len(values) - 1) * fraction))]
                mask = confidences >= th
                partial_metrics = metric(embeddings[mask], targets[mask])
                for k, v in partial_metrics.items():
                    metrics["filter-out/{}/{}".format(name, k)] = v
        return {self.prefix + k + self.suffix: v for k, v in metrics.items()}


class NearestMetricCallback(LoaderMetricCallback):
    """Callback for MAP@R or GAP@R computation.

    Args:
        scorer: Scorer object.
        input_key: Embeddings key.
        target_key: Labels key.
    """

    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        super().__init__(
            metric=NearestMetric(config=config, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_key
        )

    def on_stage_start(self, runner: "IRunner"):
        model = get_attr(runner, key="model", inner_key="model")
        scorer = get_attr(runner, key="model", inner_key="scorer")
        assert scorer is not None
        self.metric.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.metric.distribution = distribution

    def on_stage_end(self, runner: "IRunner"):
        self.metric.scorer = None
        self.metric.distribution = None


class ScoresMetric(ICallbackLoaderMetric):
    """Positive scores statistics computation in classification pipeline."""

    def __init__(self, config: Dict = None,
                 compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self._config = config
        self._distribution = None
        self._scorer = None
        self.reset(None, None)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distribution):
        self._distribution = distribution

    @property
    def scorer(self):
        return self._scorer

    @scorer.setter
    def scorer(self, scorer):
        self._scorer = scorer

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._positive_scores = []

    def update(self, embeddings: torch.Tensor, targets: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            embeddings: Tensor with embeddings distribution.
            targets: Tensor with target labels.
        """
        if self.scorer is None:
            raise RuntimeError("Scorer must be set on stage_start.")
        if self.distribution is None:
            raise RuntimeError("Distribution must be set on stage_start.")
        if targets.shape[-1] != self.distribution.num_parameters:
            # Target embeddings are not distributions and can't be matched.
            return {}
        assert embeddings.shape == targets.shape
        positive_scores = self.scorer(embeddings.detach(), targets.detach())  # (B).
        self._positive_scores.append(positive_scores)

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes nearest neighbours metrics."""
        positive_scores = torch.cat(self._positive_scores)  # (B).
        if self._is_ddp:
            positive_scores = torch.cat(all_gather(positive_scores)).detach()
        values = {
            "positive_scores/mean": positive_scores.mean(),
            "positive_scores/std": positive_scores.std()
        }
        return {self.prefix + k + self.suffix: v for k, v in values.items()}


class ScoresMetricCallback(LoaderMetricCallback):
    """Callback for positive scores statistics computation in classification pipeline."""

    def __init__(
        self, input_key: str, target_embeddings_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        super().__init__(
            metric=ScoresMetric(config=config, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key=target_embeddings_key
        )

    def on_stage_start(self, runner: "IRunner"):
        model = get_attr(runner, key="model", inner_key="model")
        scorer = get_attr(runner, key="model", inner_key="scorer")
        assert scorer is not None
        self.metric.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.metric.distribution = distribution

    def on_stage_end(self, runner: "IRunner"):
        self.metric.scorer = None
        self.metric.distribution = None


class QualityMetric(ICallbackLoaderMetric):
    """Compute sample quality estimation metrics."""

    def __init__(self, compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        self.reset(None, None)

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._confidences = []
        self._quality = []

    def update(self, confidences: torch.Tensor, quality: torch.Tensor) -> None:
        """Updates metric value with statistics for new data.

        Args:
            embeddings: Tensor with embeddings distribution.
            targets: Tensor with target labels.
        """
        self._confidences.append(confidences.detach())
        self._quality.append(quality.detach())

    def compute(self) -> Tuple[torch.Tensor, float, float, float]:
        """Computes nearest neighbours metrics."""
        metrics = self.compute_key_value()
        return [v for k, v in sorted(metrics.items())]

    def compute_key_value(self) -> Dict[str, float]:
        """Computes nearest neighbours metrics."""
        confidences = torch.cat(self._confidences)
        quality = torch.cat(self._quality)

        if self._is_ddp:
            confidences = torch.cat(all_gather(confidences)).detach()
            quality = torch.cat(all_gather(quality)).detach()

        values = {
            "quality_scc": spearmanr(quality.cpu().numpy(), confidences.cpu().numpy())[0]
        }
        return {self.prefix + k + self.suffix: v for k, v in values.items()}


class QualityMetricCallback(LoaderMetricCallback):
    """Callback for sample quality estimation metrics."""

    def __init__(
        self, input_key: str, target_key: str, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        if config:
            raise ConfigError("No config available for quality metrics.")
        if (input_key is not None) and (target_key is not None):
            super().__init__(
                metric=QualityMetric(prefix=prefix, suffix=suffix),
                input_key=input_key,
                target_key=target_key
            )
        else:
            super().__init__(
                metric=DummyMetric(),
                input_key={},
                target_key={}
            )
