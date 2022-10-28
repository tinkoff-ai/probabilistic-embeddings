from collections import OrderedDict

import torch
from catalyst import dl

from .._workarounds import OptimizerCallback
from ..config import prepare_config, ConfigError
from .gradient import GradientNormalizer
from .optimizer import SGDOptimizer, RMSpropOptimizer, AdamOptimizer, AdamWOptimizer, SamOptimizer
from .scheduler import StepScheduler, MultiStepScheduler, PlateauScheduler, WarmupScheduler, ExponentialScheduler
from .variance_scheduler import ExponentSTDSchedulerCallback


class Trainer:
    """Optimization pipeline."""

    OPTIMIZERS = {
        "sgd": SGDOptimizer,
        "rmsprop": RMSpropOptimizer,
        "adam": AdamOptimizer,
        "adamw": AdamWOptimizer,
        "sam": SamOptimizer
    }

    SCHEDULERS = {
        "step": StepScheduler,
        "multistep": MultiStepScheduler,
        "plateau": PlateauScheduler,
        "exponential": ExponentialScheduler
    }

    VARIANCE_SCHEDULERS = {
        "exponential": ExponentSTDSchedulerCallback
    }

    @staticmethod
    def get_default_config(num_epochs=16,
                           optimizer_type="sgd", optimizer_params=None, classifier_optimizer_params=None,
                           gradient_clipping=5, use_gradient_normalizer=False, gradient_normalizer_params=None,
                           scheduler_type=None, scheduler_params=None,
                           variance_scheduler_type=None, variance_scheduler_params=None,
                           warmup_epochs=0,
                           selection_dataset="train", selection_metric="loss", selection_minimize=True,
                           early_stop_patience=None, early_stop_epsilon=1e-3):
        """Get trainer parameters.

        Args:
            num_epochs: Number of training epochs.
            optimizer_type: One of `sgd` and `adam`.
            optimizer_params: Parameters of optimizer class.
            classifier_optimizer_params: Parameters of classifier optimizer. If not provided, same as optimizer_params.
            gradient_clipping: Size of gradient clipping.
            use_gradient_normalizer: Normalize gradient using moving norm.
            gradient_normalizer_params: Parameters of gradient normalizer.
            scheduler_type: One of `None` and `multistep`.
            scheduler_params: Parameters of :class:`LRScheduler`.
            variance_scheduler_type: One of `None` and `linear`.
            variance_scheduler_params: Parameters of the classifier variance scheduler.
            selection_dataset: Dataset used for checkpoint selection and early stopping.
            selection_metric: Metric used for checkpoint selection and early stopping.
            selection_minimize: Whether to minimize metric or maximize.
            early_stop_patience: Number of epochs without improvement for early stopping.
              Use None to disable early stopping.
            early_stop_epsilon: Improvement threshold for early stopping.

        """
        return OrderedDict([
            ("num_epochs", num_epochs),
            ("optimizer_type", optimizer_type),
            ("optimizer_params", optimizer_params),
            ("classifier_optimizer_params", classifier_optimizer_params),
            ("gradient_clipping", gradient_clipping),
            ("use_gradient_normalizer", use_gradient_normalizer),
            ("gradient_normalizer_params", gradient_normalizer_params),
            ("scheduler_type", scheduler_type),
            ("scheduler_params", scheduler_params),
            ("variance_scheduler_type", variance_scheduler_type),
            ("variance_scheduler_params", variance_scheduler_params),
            ("warmup_epochs", warmup_epochs),
            ("selection_dataset", selection_dataset),
            ("selection_metric", selection_metric),
            ("selection_minimize", selection_minimize),
            ("early_stop_patience", early_stop_patience),
            ("early_stop_epsilon", early_stop_epsilon)
        ])

    def __init__(self, *, config=None):
        self._config = prepare_config(self, config)
        if self._config["use_gradient_normalizer"]:
            if self._config["gradient_clipping"] is not None:
                raise ConfigError("Gradient clipping and gradient normalization are mutually exclusive.")
            self._gradient_normalizer = GradientNormalizer(**(self._config["gradient_normalizer_params"] or {}))

    def get_num_epochs(self):
        return self._config["num_epochs"]

    def get_optimizer(self, model):
        optimizer_cls = self.OPTIMIZERS[self._config["optimizer_type"]]
        param_groups = []
        embedder_params = [p for p in model.embedder.parameters() if p.requires_grad]
        if embedder_params:
            param_groups.append({"params": embedder_params})
        scorer_params = [p for p in model.scorer.parameters() if p.requires_grad]
        if scorer_params:
            param_groups.append({"params": scorer_params,
                                 **(self._config["classifier_optimizer_params"] or {})})
        classifier_params = [p for p in model.classifier.parameters() if p.requires_grad] if model.classification else []
        if classifier_params:
            param_groups.append({"params": classifier_params,
                                 **(self._config["classifier_optimizer_params"] or {})})
        total_parameters = sum([len(group["params"]) for group in param_groups])
        required_parameters = len([p for p in model.parameters() if p.requires_grad])
        assert total_parameters == required_parameters
        optimizer = optimizer_cls(param_groups, config=self._config["optimizer_params"])
        return optimizer

    def get_scheduler(self, optimizer):
        scheduler = None
        if self._config["scheduler_type"] is not None:
            scheduler_cls = self.SCHEDULERS[self._config["scheduler_type"]]
            scheduler = scheduler_cls(optimizer,
                                      minimize_metric=self._config["selection_minimize"],
                                      num_epochs=self.get_num_epochs(),
                                      config=self._config["scheduler_params"])
        if self._config["warmup_epochs"] > 0:
            scheduler = WarmupScheduler(scheduler, warmup_epochs=self._config["warmup_epochs"])
        return scheduler

    def get_callbacks(self, checkpoints_path, loss_key):
        if self._config["gradient_clipping"] is not None:
            grad_clip_kwargs = {
                "grad_clip_fn": torch.nn.utils.clip_grad_norm_,
                "grad_clip_params": {"max_norm": self._config["gradient_clipping"], "error_if_nonfinite": False}
            }
        elif self._config["use_gradient_normalizer"]:
            grad_clip_kwargs = {
                "grad_clip_fn": self._gradient_normalizer,
                "grad_clip_params": {}
            }
        else:
            grad_clip_kwargs = {}
        callbacks = {
            "optimizer": OptimizerCallback(metric_key=loss_key, model_key="model", **grad_clip_kwargs),
            "checkpoint": dl.CheckpointCallback(
                logdir=checkpoints_path,
                loader_key=self._config["selection_dataset"],
                metric_key=self._config["selection_metric"],
                minimize=self._config["selection_minimize"]
            )
        }
        if self._config["scheduler_type"] is not None:
            callbacks["scheduler"] = dl.SchedulerCallback(loader_key=self._config["selection_dataset"],
                                                          metric_key=self._config["selection_metric"])
        if self._config["variance_scheduler_type"] is not None:
            callbacks["variance_scheduler"] = self.VARIANCE_SCHEDULERS[self._config["variance_scheduler_type"]](
                self._config["num_epochs"], config=self._config["variance_scheduler_params"])
        if self._config["early_stop_patience"] is not None:
            callbacks["early_stop"] = dl.EarlyStoppingCallback(
                patience=self._config["early_stop_patience"],
                loader_key=self._config["selection_dataset"],
                metric_key=self._config["selection_metric"],
                min_delta=self._config["early_stop_epsilon"],
                minimize=self._config["selection_minimize"]
            )
        return callbacks
