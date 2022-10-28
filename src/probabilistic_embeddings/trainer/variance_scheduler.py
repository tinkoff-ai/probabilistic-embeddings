import math
from collections import OrderedDict

from catalyst.core.callback import Callback, CallbackNode, CallbackOrder

from ..config import prepare_config


class ExponentSTDSchedulerCallback(Callback):
    """Increases maximum STD during training."""

    @staticmethod
    def get_default_config(min_variance=0.01):
        """Get scheduler parameters."""
        return OrderedDict([
            ("min_variance", min_variance)
        ])

    def __init__(self, num_epochs, *, config=None):
        super().__init__(order=CallbackOrder.scheduler, node=CallbackNode.all)
        self._config = prepare_config(self, config)
        self._num_epochs = num_epochs
        self._classifier = None
        self._max_logvariance = None

    def on_stage_start(self, runner):
        self._classifier = runner.model["model"].classifier
        if not self._classifier.has_variance:
            raise ValueError("Classifier doesn't have variance.")
        self._max_logvariance = self._classifier.variance.log().item()
        self._epoch = 0

    def on_stage_end(self, runner):
        self._classifier = None
        self._max_logvariance = None

    def on_epoch_start(self, runner):
        min_logvariance = math.log(self._config["min_variance"])
        new_variance = math.exp(self._max_logvariance + (min_logvariance - self._max_logvariance) * self._epoch / self._num_epochs)
        self._classifier.set_variance(new_variance)
        self._epoch += 1
