from abc import abstractmethod, ABC
from collections import OrderedDict

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_curve

from ..config import prepare_config
from .knn import KNNIndex


def asarray(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
    return np.ascontiguousarray(x)


class NearestNeighboursBase(ABC):
    """Base class for all nearest neighbour metrics."""

    @property
    @abstractmethod
    def match_self(self):
        """Whether to compare each sample with self or not."""
        pass

    @property
    @abstractmethod
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        pass

    @property
    @abstractmethod
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        pass

    @abstractmethod
    def num_nearest(self, labels):
        """Get the number of required neighbours.

        Args:
            labels: Dataset labels.
        """
        pass

    @abstractmethod
    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.

        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores (optional): Similarity scores of elements with the same class (depends on match_self).
            confidences (optional): Confidence for each element of the batch with shape (B).

        Returns:
            Metric value.
        """
        pass


class RecallK(NearestNeighboursBase):
    """Recall@K metric."""
    def __init__(self, k):
        self._k = k

    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return False

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return False

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return False

    def num_nearest(self, labels):
        """Get the number of required neighbours.

        Args:
            labels: Dataset labels.
        """
        return self._k

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.

        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).

        Returns:
            Metric value.
        """
        mask = class_sizes > 1
        if mask.sum().item() == 0:
            return np.nan
        has_same, _ = nearest_same[mask, :self._k].max(1)
        return has_same.float().mean().item()


class ERCRecallK(NearestNeighboursBase):
    """Error-versus-Reject-Curve based on Recall@K metric."""
    def __init__(self, k):
        self._k = k

    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return False

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return False

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return True

    def num_nearest(self, labels):
        """Get the number of required neighbours.

        Args:
            labels: Dataset labels.
        """
        return self._k

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.

        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).

        Returns:
            Metric value.
        """
        if confidences is None:
            raise ValueError("Can't compute ERC without confidences.")
        mask = class_sizes > 1
        if mask.sum().item() == 0:
            return np.nan
        recalls, _ = nearest_same[mask, :self._k].max(1)
        errors = 1 - recalls.float()
        confidences = confidences[mask]

        b = len(errors)
        order = torch.argsort(confidences, descending=True)
        errors = errors[order]  # High confidence first.
        mean_errors = errors.cumsum(0) / torch.arange(1, b + 1, device=errors.device)
        return mean_errors.mean().cpu().item()


class ConfidenceRecallAccuracy(NearestNeighboursBase):
    """Compute maximum accuracy for R@1 prediction from confidence.

    NOTE: Decision threshold is adjusted using testset.
    """
    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return False

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return False

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return True

    def num_nearest(self, labels):
        """Get the number of required neighbours.

        Args:
            labels: Dataset labels.
        """
        return 1

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores=None, confidences=None):
        """Compute metric value.

        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Class size for each element.
            positive_scores: Similarity scores of elements with the same class.
            confidences: Confidence for each element of the batch with shape (B).

        Returns:
            Metric value.
        """
        assert confidences is not None
        mask = class_sizes > 1
        if mask.sum().item() == 0:
            return np.nan
        predictions = confidences[mask]
        targets = nearest_same[mask, 0]
        assert targets.ndim == 1
        pr = targets.float().mean().item()
        fprs, tprs, ths = roc_curve(targets.cpu().numpy(), predictions.cpu().numpy(), drop_intermediate=False)
        accuracy = np.max(pr * tprs + (1 - pr) * (1 - fprs))
        return accuracy


class ATRBase(NearestNeighboursBase):
    """Base class for @R metrics.

    All @R metrics search for the number of neighbours equal to class size.

    Args:
        match_self: Whether to compare each sample with self or not.

    Inputs:
        - parameters: Embeddings distributions tensor with shape (B, P).
        - labels: Label for each embedding with shape (B).

    Outputs:
        - Metric value.
    """

    def __init__(self, match_self=False):
        super().__init__()
        self._match_self = match_self

    @property
    @abstractmethod
    def oversample(self):
        """Sample times more nearest neighbours."""
        pass

    @abstractmethod
    def _aggregate(self, nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores, confidences=None):
        """Compute metric value.

        Args:
            nearest_same: Matching labels for nearest neighbours with shape (B, R).
                Matches are coded with 1 and mismatches with 0.
            nearest_scores: Score for each neighbour with shape (B, R).
            num_nearest: Number of nearest neighbours for each element of the batch with shape (B).
            class_sizes: Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        """
        pass

    @property
    def match_self(self):
        """Whether to compare each sample with self or not."""
        return self._match_self

    @property
    def need_positives(self):
        """Whether metric requires positive scores or not."""
        return True

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return False

    def num_nearest(self, labels):
        """Get maximum number of required neighbours.

        Args:
            labels: Dataset labels.
        """
        max_r = torch.bincount(labels).max().item()
        max_r *= self.oversample
        return max_r

    def __call__(self, nearest_same, nearest_scores, class_sizes, positive_scores, confidences=None):
        """Compute metric value.

        Args:
            nearset_same: Binary labels of nearest neighbours equal to 1 iff class is equal to the query.
            nearest_scores: Similarity scores of nearest neighbours.
            class_sizes: Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).

        Returns:
            Metric value.
        """
        num_positives = class_sizes if self.match_self else class_sizes - 1
        num_nearest = torch.clip(num_positives * self.oversample, max=nearest_same.shape[1])
        return self._aggregate(nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores,
                               confidences=confidences)


class MAPR(ATRBase):
    """MAP@R metric.

    See "A Metric Learning Reality Check" (2020) for details.
    """

    @property
    def oversample(self):
        """Sample times more nearest neighbours."""
        return 1

    def _aggregate(self, nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores, confidences=None):
        """Compute MAP@R.

        Args:
            nearest_same: Matching labels for nearest neighbours with shape (B, R).
                Matches are coded with 1 and mismatches with 0.
            nearest_scores: (unused) Score for each neighbour with shape (B, R).
            num_nearest: Number of nearest neighbours for each element of the batch with shape (B).
            class_sizes: (unused) Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        """
        b, r = nearest_same.shape
        device = nearest_same.device
        range = torch.arange(1, r + 1, device=device)  # (R).
        count_mask = range[None].tile(b, 1) <= num_nearest[:, None]  # (B, R).
        precisions = count_mask * nearest_same * torch.cumsum(nearest_same, dim=1) / range[None]  # (B, R).
        maprs = precisions.sum(-1) / torch.clip(num_nearest, min=1)  # (B).
        return maprs.mean()


class ERCMAPR(ATRBase):
    """ERC curve for MAP@R metric."""

    @property
    def need_confidences(self):
        """Whether metric requires confidences or not."""
        return True

    @property
    def oversample(self):
        """Sample times more nearest neighbours."""
        return 1

    def _aggregate(self, nearest_same, nearest_scores, num_nearest, class_sizes, positive_scores, confidences=None):
        """Compute MAP@R ERC.

        Args:
            nearest_same: Matching labels for nearest neighbours with shape (B, R).
                Matches are coded with 1 and mismatches with 0.
            nearest_scores: (unused) Score for each neighbour with shape (B, R).
            num_nearest: Number of nearest neighbours for each element of the batch with shape (B).
            class_sizes: (unused) Number of elements in the class for each element of the batch.
            positive_scores: Similarity scores of elements with the same class.
            confidences (optional): Confidence for each element of the batch with shape (B).
        """
        if confidences is None:
            raise ValueError("Can't compute ERC without confidences.")
        b, r = nearest_same.shape
        device = nearest_same.device
        range = torch.arange(1, r + 1, device=device)  # (R).
        count_mask = range[None].tile(b, 1) <= num_nearest[:, None]  # (B, R).
        precisions = count_mask * nearest_same * torch.cumsum(nearest_same, dim=1) / range[None]  # (B, R).
        maprs = precisions.sum(-1) / torch.clip(num_nearest, min=1)  # (B).
        errors = 1 - maprs.float()

        b = len(errors)
        order = torch.argsort(confidences, descending=True)
        errors = errors[order]  # High confidence first.
        mean_errors = errors.cumsum(0) / torch.arange(1, b + 1, device=errors.device)
        return mean_errors.mean().cpu().item()


class NearestNeighboursMetrics:
    """Metrics based on nearest neighbours search.

    Args:
        distribution: Distribution object.
        scorer: Scorer object.

    Inputs:
        - parameters: Embeddings distributions tensor with shape (B, P).
        - labels: Label for each embedding with shape (B).

    Outputs:
        - Metrics values.
    """

    METRICS = {
        "recall": RecallK,
        "erc-recall@1": lambda: ERCRecallK(1),
        "confidence-accuracy": ConfidenceRecallAccuracy,
        "mapr": MAPR,
        "erc-mapr": ERCMAPR,
        "mapr-ms": lambda: MAPR(match_self=True),
    }

    @staticmethod
    def get_default_config(backend="torch", broadcast_backend="torch", metrics=None, prefetch_factor=2, recall_k_values=(1,)):
        """Get metrics parameters.

        Args:
            backend: KNN search engine ("faiss", "torch" or "numpy").
            broadcast_backend: Torch doesn't support broadcast for gather method.
              We can emulate this behaviour with Numpy ("numpy") or tiling ("torch").
            metrics: List of metric names to compute ("recall", "mapr", "mapr-nms").
                By default compute all available metrics.
            prefetch_factor: Nearest neighbours number scaler for presampling.
            recall_k_values: List of K values to compute recall at.
        """
        return OrderedDict([
            ("backend", backend),
            ("broadcast_backend", broadcast_backend),
            ("metrics", metrics),
            ("prefetch_factor", prefetch_factor),
            ("recall_k_values", recall_k_values)
        ])

    def __init__(self, distribution, scorer, *, config=None):
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._scorer = scorer

        self._metrics = OrderedDict()
        metric_names = self._config["metrics"] if self._config["metrics"] is not None else list(self.METRICS)
        for name in metric_names:
            if name == "recall":
                for k in self._config["recall_k_values"]:
                    k = int(k)
                    self._metrics["{}@{}".format(name, k)] = self.METRICS[name](k)
            else:
                metric = self.METRICS[name]()
                if self._distribution.has_confidences or not metric.need_confidences:
                    self._metrics[name] = metric

    def __call__(self, parameters, labels):
        if parameters.ndim != 2:
            raise ValueError("Expected parameters matrix.")
        if len(labels) != len(parameters):
            raise ValueError("Batch size mismatch between labels and parameters.")
        parameters = parameters.detach()  # (B, P).
        labels = labels.detach()  # (B).

        need_confidences = any([metric.need_confidences for metric in self._metrics.values()])
        confidences = self._distribution.confidences(parameters) if need_confidences else None  # (B) or None.

        # Find desired nearest neighbours number for each sample and total.
        label_counts = torch.bincount(labels)  # (L).
        class_sizes = label_counts[labels]  # (B).
        num_nearest = max(metric.num_nearest(labels) + int(not metric.match_self) for metric in self._metrics.values())
        num_nearest = min(num_nearest, len(labels))

        # Gather nearest neighbours (sorted in score descending order).
        nearest, scores = self._find_nearest(parameters, num_nearest)  # (B, R), (B, R).
        num_nearest = torch.full((len(nearest),), num_nearest, device=labels.device)
        nearest_labels = self._gather_broadcast(labels[None], 1, nearest, backend=self._config["broadcast_backend"])  # (B, R).
        nearest_same = nearest_labels == labels[:, None]  # (B, R).

        need_positives = any(metric.need_positives for metric in self._metrics.values())
        if need_positives:
            positive_scores, _, positive_same_mask = self._get_positives(parameters, labels)
        else:
            positive_scores, positive_same_mask = None, None

        need_nms = any(not metric.match_self for metric in self._metrics.values())
        if need_nms:
            no_self_mask = torch.arange(len(labels), device=parameters.device)[:, None] != nearest
            nearest_same_nms, _ = self._gather_mask(nearest_same, num_nearest, no_self_mask)
            scores_nms, num_nearest = self._gather_mask(scores, num_nearest, no_self_mask)
            if need_positives:
                positive_scores_nms, _ = self._gather_mask(positive_scores, class_sizes, ~positive_same_mask)
            else:
                positive_scores_nms = None

        metrics = OrderedDict()
        for name, metric in self._metrics.items():
            if metric.match_self:
                metrics[name] = metric(nearest_same, scores, class_sizes, positive_scores, confidences=confidences)
            else:
                metrics[name] = metric(nearest_same_nms, scores_nms, class_sizes, positive_scores_nms, confidences=confidences)
        return metrics

    def _find_nearest(self, parameters, max_nearest):
        """Find nearest neighbours for each element of the batch.

        Stage 1. Find elements close to query by L2. Nearest neighbours are searched
        for each distribution mode independently (in multi-modal setup).
        Stage 2. Remove duplicates caused by cross-modal mining in stage 1.
        Stage 3. Rescore nearest neighbours using scorer.
        """
        _, modes = self._distribution.modes(parameters)  # (B, C, D).
        b, c, d = modes.shape
        # Presample using simple L2/dot scoring.
        prefetch = min(max_nearest * self._config["prefetch_factor"], b)
        candidates_indices = self._multimodal_knn(modes, prefetch).reshape((b, -1))  # (B, C * R).
        candidates_indices = self._remove_duplicates(candidates_indices, max_nearest)
        # Rescore using scorer.
        candidates_parameters = self._gather_broadcast(parameters[None], 1, candidates_indices[..., None],
                                                       backend=self._config["broadcast_backend"])  # (B, C * R, P).
        with torch.no_grad():
            scores = self._scorer(parameters[:, None, :], candidates_parameters)  # (B, C * R * S).
        nearest_order = torch.argsort(scores, dim=1, descending=True)  # (B, C * R * S).
        nearest = torch.gather(candidates_indices, 1, nearest_order)  # (B, C * R * S), at least R * S unique indices for each query.
        nearest_scores = torch.gather(scores, 1, nearest_order)  # (B, C * R * S).
        return nearest, nearest_scores

    def _get_positives(self, parameters, labels):
        label_counts = torch.bincount(labels)
        num_labels = len(label_counts)
        max_label_count = label_counts.max().item()
        by_label = torch.full((num_labels, max_label_count), -1, dtype=torch.long)
        counts = np.zeros(num_labels, dtype=np.int64)
        for i, label in enumerate(labels.cpu().numpy()):
            by_label[label][counts[label]] = i
            counts[label] += 1
        by_label = by_label.to(labels.device)  # (L, C).
        indices = by_label[labels]  # (B, C).
        num_positives = torch.from_numpy(counts).long().to(labels.device)[labels]
        positive_parameters = self._gather_broadcast(parameters[None], 1, indices[..., None],
                                                     backend=self._config["broadcast_backend"])  # (B, C, P).
        with torch.no_grad():
            positive_scores = self._scorer(parameters[:, None, :], positive_parameters)  # (B, C).
        same_mask = indices == torch.arange(len(labels), device=indices.device)[:, None]
        # Sort first elements in each row according to counts.
        no_sort_mask = torch.arange(positive_scores.shape[1], device=parameters.device)[None] >= num_positives[:, None]
        positive_scores[no_sort_mask] = positive_scores.min() - 1
        positive_scores, order = torch.sort(positive_scores, dim=1, descending=True)
        same_mask = torch.gather(same_mask, 1, order)
        return positive_scores, num_positives, same_mask

    def _multimodal_knn(self, x, k):
        """Find nearest neighbours for multimodal queries.

        Args:
            x: Embeddings with shape (B, C, D) where C is the number of modalities.
            k: Number of nearest neighbours.

        Returns:
            Nearest neighbours indices with shape (B, C, K). Indices are in the range [0, B - 1].
        """
        b, c, d = x.shape
        if k > b:
            raise ValueError("Number of nearest neighbours is too large: {} for batch size {}.".format(k, b))
        x_flat = asarray(x).reshape((b * c, d))
        with KNNIndex(d, backend=self._config["backend"]) as index:
            index.add(x_flat)
            _, indices = index.search(x_flat, k)  # (B * C, K), indices are in [0, B * C - 1].
        indices //= c  # (B * C, K), indices are in [0, B - 1].
        return torch.from_numpy(indices.reshape((b, c, k))).long().to(x.device)

    @staticmethod
    def _remove_duplicates(indices, num_unique):
        """Take first n unique values from each row.

        Args:
            indices: Input indices with shape (B, K).
            num_unique: Number of unique indices in each row.

        Returns:
            Unique indices with shape (B, num_unique) and new scores if scores are provided.
        """
        b, k = indices.shape
        if k == 1:
            return indices
        sorted_indices, order = torch.sort(indices, dim=1, stable=True)
        mask = sorted_indices[:, 1:] != sorted_indices[:, :-1]  # (B, K - 1).
        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1)  # (B, K).
        mask = torch.gather(mask, 1, torch.argsort(order, dim=1))
        counts = torch.cumsum(mask, 1)  # (B, K).
        mask &= counts <= num_unique  # (B, K).

        # Some FAISS indices allow duplicates. In this case total number of unique elements is less than min_unique.
        # Add tail samples to get exact min_unique number.
        num_extra_zeros = torch.clip(num_unique - counts[:, -1], 0)
        counts = torch.cumsum(~mask, 1)
        sums = counts[:, -1].unsqueeze(-1)  # (B, 1).
        counts = torch.cat((sums, sums - counts[:, :-1]), dim=-1)  # (B, K).
        mask |= counts <= num_extra_zeros[:, None]

        unique = indices[mask].reshape(b, num_unique)  # (B, R), all indices are unique.
        return unique

    @staticmethod
    def _gather_mask(matrix, lengths, mask):
        b, n = matrix.shape
        device = matrix.device
        length_mask = torch.arange(n, device=device)[None].tile(b, 1) < lengths[:, None]  # (B, N).
        mask = mask & length_mask
        counts = mask.sum(1)  # (B).
        max_count = counts.max()
        padding = max_count - counts.min()
        if padding > 0:
            matrix = torch.cat((matrix, torch.zeros(b, padding, dtype=matrix.dtype, device=device)), dim=1)
            mask = torch.cat((mask, torch.ones(b, padding, dtype=torch.bool, device=device)), dim=1)
        mask &= torch.cumsum(mask, 1) <= max_count
        return matrix[mask].reshape(b, max_count), counts

    @staticmethod
    def _gather_broadcast(input, dim, index, backend="torch"):
        if backend == "torch":
            shape = np.maximum(np.array(input.shape), np.array(index.shape)).tolist()
            index[index < 0] += shape[dim]
            shape[dim] = input.shape[dim]
            input = input.broadcast_to(shape)
            shape[dim] = index.shape[dim]
            index = index.broadcast_to(shape)
            return input.gather(dim, index)
        elif backend == "numpy":
            result_array = np.take_along_axis(asarray(input),
                                              asarray(index),
                                              dim)
            result = torch.from_numpy(result_array).to(dtype=input.dtype, device=input.device)
            return result
        else:
            raise ValueError("Unknown broadcast backend: {}.".format(backend))
