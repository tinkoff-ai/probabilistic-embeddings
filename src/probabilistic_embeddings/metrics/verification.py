import os
from collections import OrderedDict
from typing import Tuple, Dict, Optional, Any, Union

import numpy as np
import torch
from catalyst.metrics._metric import ICallbackLoaderMetric
from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.utils.distributed import all_gather, get_rank
from sklearn import metrics

from ..config import prepare_config


def risk_coverage_curve(loss, confidence):
    """Compute risk-coverage curve.

    Returns:
      risk: Mean loss for each confidence threshold.
      coverage: Coverage for each confidence threshold.
      thresholds: Decreasing thresholds.
    """
    loss = np.asarray(loss)
    confidence = np.asarray(confidence)
    if len(loss) != len(confidence):
        raise ValueError("Size mismatch.")
    if len(loss) == 0:
        raise ValueError("Empty data.")
    if not np.all(loss >= 0):
        raise ValueError("Losses must be non-negative.")
    n = len(loss)
    ths = np.empty(n + 1)
    risk = np.empty(n + 1)
    coverage = np.arange(n + 1) / n

    order = np.flip(np.argsort(confidence))
    ths[1:] = confidence[order]
    ths[0] = ths[1] + 1
    risk[1:] = np.cumsum(loss[order]) / (np.arange(n) + 1)
    risk[0] = 0
    return risk, coverage, ths


class VerificationMetrics(ICallbackLoaderMetric):
    """Compute verification metrics.

    Available metrics:
      - pr: Fraction of positives in the dataset.
      - max_accuracy: Verification accuracy with best threshold.
      - auc: ROC AUC.
      - tpr: TPR for the requested FPR.
      - fpr: Actual FPR of the found threshold.
      - eer: Equal error rate.
    """

    @staticmethod
    def get_default_config(fpr=1e-3, roc_curve_dump_dir=None):
        """Get config.

        Args:
            fpr: Required FPR for TPR computation.
            roc_curve_dump_dir: If not None, saves ROC curve to `roc_curve_dump_dir`.
        """
        return OrderedDict([
            ("fpr", fpr),
            ("roc_curve_dump_dir", roc_curve_dump_dir)
        ])

    def __init__(self, config: Dict = None,
                 compute_on_call: bool = True, prefix: str = None, suffix: str = None):
        super().__init__(compute_on_call=compute_on_call, prefix=prefix, suffix=suffix)
        config = prepare_config(self, config)
        self._fpr = np.float(config["fpr"])
        self._scores = []
        self._confidences = []
        self._targets = []
        self._is_ddp = get_rank() > -1
        self._roc_dump = config["roc_curve_dump_dir"]
        if self._roc_dump and self._roc_dump != "":
            os.makedirs(self._roc_dump, exist_ok=True)
        self._dump_counter = 0

    def reset(self, num_batches, num_samples) -> None:
        """Resets all fields"""
        self._is_ddp = get_rank() > -1
        self._scores = []
        self._confidences = []
        self._targets = []

    def update(self, scores: torch.Tensor, targets: torch.Tensor, confidences: torch.Tensor = None) -> None:
        """Updates metric value with statistics for new data.

        Args:
            scores: Tensor with scores.
            targets: Tensor with targets.
        """
        self._scores.append(scores.cpu().detach())
        self._targets.append(self._normalize_targets(targets).cpu().detach())
        if confidences is not None:
            self._confidences.append(confidences.cpu().detach())

    def compute(self) -> Optional[Tuple[Any, Union[int, Any], float, Any, Any, Union[float, Any]]]:
        """Computes the AUC metric based on saved statistics."""
        scores = torch.cat(self._scores)
        targets = torch.cat(self._targets)

        if self._is_ddp:
            scores = torch.cat(all_gather(scores))
            targets = torch.cat(all_gather(targets))

        use_confidences = bool(self._confidences)
        if use_confidences:
            confidences = torch.cat(self._confidences)
            if self._is_ddp:
                confidences = torch.cat(all_gather(confidences))
            assert len(confidences) == len(scores)
        else:
            confidences = torch.zeros_like(scores)
        mask = scores.isfinite()
        scores = scores[mask]
        confidences = confidences[mask]
        targets = targets[mask]
        if len(scores) == 0:
            return None

        pr = targets.float().mean().item()

        fprs, tprs, ths = metrics.roc_curve(targets.numpy(), scores.numpy(), drop_intermediate=False)
        auc = metrics.auc(fprs, tprs)

        if self._roc_dump and self._roc_dump != "":
            out_file = os.path.join(self._roc_dump, f"{self.suffix}_{self._dump_counter}")
            np.save(
                out_file,
                {
                    "tprs": tprs,
                    "fprs": fprs
                },
            )
            self._dump_counter += 1

        # Roc is sorted in increasing FPR order. Find minimum threshold (maximum fpr).
        fprs = fprs.astype(np.float)
        fpr_index, fpr = self._find_closest(fprs, self._fpr, last=True)
        tpr = tprs[fpr_index]

        # Choose between two thresholds.
        eer_index1, _ = self._find_closest(fprs, 1 - tprs, last=True)
        eer1 = 0.5 * (fprs[eer_index1] + 1 - tprs[eer_index1])
        eer_index2, _ = self._find_closest(fprs, 1 - tprs, last=False)
        eer2 = 0.5 * (fprs[eer_index2] + 1 - tprs[eer_index2])
        eer = min(eer1, eer2)

        accuracy_index = np.argmax(pr * tprs + (1 - pr) * (1 - fprs))
        max_accuracy = pr * tprs[accuracy_index] + (1 - pr) * (1 - fprs[accuracy_index])

        if use_confidences:
            # Compute Risk-Coverage and ROC.
            th = ths[accuracy_index]
            predictions = scores >= th
            correct = predictions == targets
            confidence_auroc = metrics.roc_auc_score(correct.numpy(), confidences.numpy())
            precisions, recalls, _ = metrics.precision_recall_curve(correct.numpy(), confidences.numpy())
            confidence_aupr = metrics.auc(recalls, precisions)
            risk, coverage, _ = risk_coverage_curve(1 - correct.numpy(), confidences.numpy())
            confidence_aurcc = metrics.auc(coverage, risk)
        else:
            confidence_auroc = None
            confidence_aupr = None
            confidence_aurcc = None
        return pr, max_accuracy, auc, tpr, fpr, eer, confidence_auroc, confidence_aupr, confidence_aurcc

    def compute_key_value(self) -> Dict[str, float]:
        """Computes the binary AUC metric based on saved statistics and returns key-value results."""
        names = ["pr", "max_accuracy", "auc", "tpr", "fpr", "eer",
                 "confidence_auroc", "confidence_aupr", "confidence_aurcc"]
        values = self.compute()
        if values is None:
            return {}
        assert len(names) == len(values)
        return {self.prefix + name + self.suffix: value for name, value in zip(names, values)
                if value is not None}

    @staticmethod
    def _normalize_targets(targets):
        targets = targets.long()
        if not torch.logical_or(targets == 1, targets == 0).all():
            raise ValueError("Expected boolean targets or {0, 1} targets.")
        return targets

    @staticmethod
    def _find_closest(array, value, last=False):
        deltas = np.abs(array - value)
        if last:
            deltas = deltas[::-1]
        index = np.argmin(deltas)
        if last:
            index = len(deltas) - 1 - index
        return index, array[index]


class VerificationMetricsCallback(LoaderMetricCallback):
    """Callback for verification metrics computation.

    Args:
        input_key: Pairwise scores key.
        target_key: Labels key.
    """

    def __init__(
        self, scores_key: str, targets_key: str, confidences_key: str = None, prefix: str = None, suffix: str = None,
        config: Dict = None
    ):
        input_key = {
            scores_key: "scores",
        }
        if confidences_key is not None:
            input_key[confidences_key] = "confidences"
        super().__init__(
            metric=VerificationMetrics(config=config, prefix=prefix, suffix=suffix),
            input_key=input_key,
            target_key={
                targets_key: "targets"
            }
        )
