import math
import torch


class GradientNormalizer(torch.nn.Module):
    """Normalize gradient value using running gradient norm mean.

    Outputs:
        Normalized gradient norm.
    """

    def __init__(self, clip=1e-2, momentum=0.9):
        super().__init__()
        self._momentum = momentum
        self._clip = clip
        self.register_buffer("is_first", torch.ones(1, dtype=torch.bool))
        self.register_buffer("moving_norm", torch.zeros([]))

    def forward(self, parameters):
        parameters = list(parameters)
        with torch.no_grad():
            norm = self._compute_grad_norm(parameters)
            momentum = 0 if self.is_first else self._momentum
            self.moving_norm.fill_(self.moving_norm * momentum + (1 - momentum) * norm)
            mean = self.moving_norm.clip(min=self._clip).item()
            self._mul_grad(parameters, 1 / mean)
            self.is_first.fill_(False)
        return norm / mean

    @staticmethod
    def _compute_grad_norm(parameters):
        parameters = [p for p in parameters if p.grad is not None]
        device = parameters[0].grad.device
        norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in parameters]))
        return norm

    @staticmethod
    def _mul_grad(parameters, alpha):
        for p in parameters:
            if (not p.requires_grad) or (p.grad is None):
                continue
            p.grad *= alpha
