from collections import OrderedDict

import torch

from ..config import prepare_config


class StepScheduler(torch.optim.lr_scheduler.StepLR):
    """Configurable LR scheduler."""

    @staticmethod
    def get_default_config(step=10, gamma=0.1):
        """Get scheduler parameters."""
        return OrderedDict([
            ("step", step),
            ("gamma", gamma)
        ])

    def __init__(self, optimizer, num_epochs, *, minimize_metric=True, config=None):
        config = prepare_config(self, config)
        super().__init__(optimizer,
                         step_size=config["step"],
                         gamma=config["gamma"])


class MultiStepScheduler(torch.optim.lr_scheduler.MultiStepLR):
    """Configurable LR scheduler."""

    @staticmethod
    def get_default_config(milestones=[9, 14], gamma=0.1):
        """Get scheduler parameters."""
        return OrderedDict([
            ("milestones", milestones),
            ("gamma", gamma)
        ])

    def __init__(self, optimizer, num_epochs, *, minimize_metric=True, config=None):
        config = prepare_config(self, config)
        super().__init__(optimizer,
                         milestones=config["milestones"],
                         gamma=config["gamma"])


class PlateauScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Configurable LR scheduler."""

    @staticmethod
    def get_default_config(patience=10, factor=0.1):
        """Get scheduler parameters."""
        return OrderedDict([
            ("patience", patience),
            ("factor", factor)
        ])

    def __init__(self, optimizer, num_epochs, *, minimize_metric=True, config=None):
        config = prepare_config(self, config)
        super().__init__(optimizer,
                         mode=("min" if minimize_metric else "max"),
                         patience=config["patience"],
                         factor=config["factor"])


class ExponentialScheduler(torch.optim.lr_scheduler.ExponentialLR):
    """Configurable LR scheduler."""

    @staticmethod
    def get_default_config(lr_at_last_epoch=0.0001):
        """Get scheduler parameters."""
        return OrderedDict([
            ("lr_at_last_epoch", lr_at_last_epoch),
        ])

    def __init__(self, optimizer, num_epochs, *, minimize_metric=True, config=None):
        config = prepare_config(self, config)
        lr_0 = optimizer.param_groups[0]["lr"]
        lr_t = config["lr_at_last_epoch"]
        t = num_epochs
        gamma = (lr_t / lr_0) ** (1 / t)
        super().__init__(optimizer,
                         gamma=gamma)


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Add warmup steps to LR scheduler."""
    def __init__(self, scheduler, warmup_epochs=1):
        self._scheduler = scheduler
        self._warmup_epochs = warmup_epochs
        super().__init__(optimizer=scheduler.optimizer,
                         last_epoch=scheduler.last_epoch,
                         verbose=scheduler.verbose)

    def state_dict(self):
        state = self._scheduler.state_dict() if self._scheduler is not None else {}
        state["warmup_epochs"] = self._warmup_epochs
        return state

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        self._warmup_epochs = state_dict.pop("warmup_epochs")
        if self._scheduler is not None:
            self._scheduler.load_state_dict(state_dict)
        super().load_state_dict(state_dict)

    def get_lr(self):
        lr = self._scheduler.get_last_lr() if self._scheduler is not None else self.base_lrs
        if self.last_epoch <= self._warmup_epochs:
            lr = [0.0] * len(lr)
        elif self.last_epoch == self._warmup_epochs + 1:
            lr = self.base_lrs
        return lr

    def step(self):
        if self.last_epoch > 0:
            self._scheduler.step()
        super().step()
