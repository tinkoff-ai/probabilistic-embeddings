import torch
import torch.nn.functional as F
import wandb
import catalyst.callbacks.checkpoint
from catalyst import dl
from catalyst.loggers.wandb import WandbLogger
from catalyst.callbacks.control_flow import LOADERS


# FIX lambda pickle in distributed mode.

class _filter_fn_from_loaders:
    def __init__(self, loaders: LOADERS, reverse_condition: bool):
        assert reverse_condition is False
        assert isinstance(loaders, str)
        self._loader = loaders

    def __call__(self, stage, epoch, loader):
        return loader == self._loader


catalyst.callbacks.control_flow._filter_fn_from_loaders = _filter_fn_from_loaders


# Create Wandb logger after fork.

class AfterForkWandbLogger(WandbLogger):
    def __init__(self, project=None, name=None, entity=None, **kwargs):
        self.project = project
        self.name = name
        self.entity = entity
        self.run = None
        self.kwargs = kwargs

    def init(self):
        self.run = wandb.init(
            project=self.project, name=self.name, entity=self.entity, allow_val_change=True, tags=[],
            **self.kwargs
        )

    def log_hparams(
        self,
        hparams,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        if (self.run is None) and (scope == "stage"):
            self.init()
        if self.run is not None:
            super().log_hparams(hparams, scope, run_key, stage_key)


# Catalyst seed can't be 0.

# Catalyst Optimizer doesn't make gradient clipping right with AMP.
# Catalyst Optimizer doesn't support closure.
class ClosureOptimizer:
    def __init__(self, optimizer, closure):
        self._optimizer = optimizer
        self._closure = closure

    def step(self):
        self._optimizer.step(closure=self._closure)

    @property
    def param_groups(self):
        return self._optimizer.param_groups


class OptimizerCallback(dl.OptimizerCallback):
    def on_batch_end(self, runner):
        """Event handler."""
        if runner.is_train_loader:
            if self.accumulation_steps != 1:
                raise NotImplementedError("Doesn't support closure with accumulation_steps.")
            self._accumulation_counter += 1
            need_gradient_step = self._accumulation_counter % self.accumulation_steps == 0

            loss = runner.batch_metrics[self.metric_key]
            runner.engine.backward_loss(loss, self.model, self.optimizer)
            self._apply_gradnorm(runner)

            if need_gradient_step:
                runner.engine.optimizer_step(loss, self.model, ClosureOptimizer(self.optimizer, lambda: self._closure(runner)))
                runner.engine.zero_grad(loss, self.model, self.optimizer)

            runner.batch_metrics.update(self._get_lr_momentum_stats())
            if hasattr(runner.engine, "scaler"):
                scaler_state = runner.engine.scaler.state_dict()
                runner.batch_metrics["gradient/scale"] = scaler_state["scale"] or 1.0
                runner.batch_metrics["gradient/growth_tracker"] = scaler_state["_growth_tracker"]

    def _apply_gradnorm(self, runner):
        if self.grad_clip_fn is not None:
            if hasattr(runner.engine, "scaler"):
                runner.engine.scaler.unscale_(self.optimizer)
            norm = self.grad_clip_fn(self.model.parameters())
        else:
            parameters = [p for p in self.model.parameters() if p.grad is not None]
            device = parameters[0].grad.device
            norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]))
        runner.batch_metrics["gradient/norm"] = norm.item()

    def _closure(self, runner):
        """Forward-backward pass used in multi-step optimizers."""
        runner._handle_train_batch((runner.batch["images"], runner.batch["labels"]))
        runner.batch = runner.engine.sync_device(runner.batch)
        runner.callbacks["criterion"].on_batch_end(runner)
        loss = runner.batch_metrics[self.metric_key]
        runner.engine.zero_grad(loss, self.model, self.optimizer)
        runner.engine.backward_loss(loss, self.model, self.optimizer)
        self._apply_gradnorm(runner)
        return loss


# FIX Catalyst inference with ArcFace and CosFace.
class ArcFace(catalyst.contrib.nn.ArcFace):
    def forward(self, input: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))

        if target is None:
            return cos_theta * self.s

        theta = torch.acos(torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps))

        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        mask = torch.where(theta > self.threshold, torch.zeros_like(one_hot), one_hot)

        logits = torch.cos(torch.where(mask.bool(), theta + self.m, theta))
        logits *= self.s

        return logits


class CosFace(catalyst.contrib.nn.CosFace):
    def forward(self, input: torch.Tensor, target: torch.LongTensor = None) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m

        if target is None:
            return cosine * self.s

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1).long(), 1)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s

        return logits
