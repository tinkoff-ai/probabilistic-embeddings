from collections import OrderedDict

from catalyst import dl

from ..config import prepare_config, ConfigError
from .verification import VerificationMetricsCallback
from .classification import NearestMetricCallback, ScoresMetricCallback, QualityMetricCallback


class Metrics:
    """Metrics and metric callbacks constructor.

    Args:
        num_classes: Number of classes in trainset.
        openset: Whether train and test have different sets of labels or not.
    """

    CLASSIFICATION_CALLBACKS = {
        "accuracy": lambda num_classes, labels_key, embeddings_key, target_embeddings_key, logits_key, confidences_key, quality_key, params: dl.AccuracyCallback(
            num_classes=num_classes,
            input_key=logits_key,
            target_key=labels_key,
            **params
        ),
        "nearest": lambda num_classes, labels_key, embeddings_key, target_embeddings_key, logits_key, confidences_key, quality_key, params: NearestMetricCallback(
            input_key=embeddings_key,
            target_key=labels_key,
            config=params
        ),
        "quality": lambda num_classes, labels_key, embeddings_key, target_embeddings_key, logits_key, confidences_key, quality_key, params: QualityMetricCallback(
            input_key=confidences_key,
            target_key=quality_key,
            config=params
        ),
        "scores": lambda num_classes, labels_key, embeddings_key, target_embeddings_key, logits_key, confidences_key, quality_key, params: ScoresMetricCallback(
            input_key=embeddings_key,
            target_embeddings_key=target_embeddings_key,
            config=params
        )
    }

    @staticmethod
    def get_default_config(verification_metrics_params=None,
                           train_classification_metrics=None,
                           test_classification_metrics=None):
        """Get metrics parameters.

        Args:
            verification_metrics_params: Parameters of :class:`VerificationMetricsCallback`.
            train_classification_metrics: List of {type, params} dicts or just metric names to compute on train.
                Default is to use accuracy for closedset and nothing for openset.
            test_classification_metrics: List of {type, params} dicts or just metric names to compute on test/val.
                Default is to use accuracy for closedset and nearest for openset.
        """
        return OrderedDict([
            ("verification_metrics_params", verification_metrics_params),
            ("train_classification_metrics", train_classification_metrics),
            ("test_classification_metrics", test_classification_metrics)
        ])

    def __init__(self, num_classes, openset, *, config=None):
        self._config = prepare_config(self, config)
        self._num_classes = num_classes
        self._openset = openset

        if self._config["test_classification_metrics"] is None:
            self._config["test_classification_metrics"] = ["nearest"] if openset else ["accuracy"]
        if self._config["train_classification_metrics"] is None:
            self._config["train_classification_metrics"] = [] if openset else ["accuracy"]

    def get_classification_callbacks(self, *, train, labels_key, embeddings_key,
                                     target_embeddings_key=None, logits_key=None,
                                     confidences_key=None, quality_key=None):
        """Get classification callbacks."""
        callbacks = OrderedDict()
        metrics = self._config["train_classification_metrics"] if train else self._config["test_classification_metrics"]
        for metric_dict in metrics:
            if isinstance(metric_dict, str):
                metric_dict = {"type": metric_dict}
            metric_type = metric_dict["type"]
            metric_params = metric_dict.get("params", {})
            if metric_type not in self.CLASSIFICATION_CALLBACKS:
                raise ConfigError("Unknown metric type.")
            if metric_type == "accuracy" and not logits_key:
                continue
            callbacks[metric_type] = self.CLASSIFICATION_CALLBACKS[metric_type](
                self._num_classes,
                labels_key,
                embeddings_key,
                target_embeddings_key,
                logits_key,
                confidences_key,
                quality_key,
                metric_params
            )
        return callbacks

    def get_verification_callbacks(self, *, train, labels_key, scores_key, confidences_key=None, suffix=None):
        """Get metrics callbacks for datasets dictionary."""
        callbacks = OrderedDict()
        callbacks["metrics"] = VerificationMetricsCallback(scores_key=scores_key,
                                                           targets_key=labels_key,
                                                           confidences_key=confidences_key,
                                                           config=self._config["verification_metrics_params"],
                                                           suffix=suffix)
        return callbacks
