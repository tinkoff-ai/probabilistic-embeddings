import torch


class PowerPooling2d(torch.nn.Module):
    def __init__(self, power):
        super().__init__()
        self._power = power

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError("Expected tensor with shape (b, c, h, w).")
        x.pow(self._power).sum(dim=(2, 3), keepdim=True).pow(1 / self._power)
        return x


class MultiPool2d(torch.nn.Module):
    """Combines average, power and max poolings.

    Args:
        mode: Combination of "a", "m", and digits to describe poolings used.
            For example "am3" means average, maximum and power-3 poolings.
        aggregate: Either "sum" or "cat".
    """
    def __init__(self, mode="am", aggregate="sum"):
        super().__init__()
        if aggregate not in ["sum", "cat"]:
            raise ValueError("Unknown aggrageation: {}.".format(aggregate))
        self._aggregate = aggregate
        self._poolings = []
        for m in mode:
            if m == "a":
                self._poolings.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
            elif m == "m":
                self._poolings.append(torch.nn.AdaptiveMaxPool2d(output_size=(1, 1)))
            else:
                try:
                    power = int(m)
                except Exception:
                    raise ValueError("Unknown pooling: {}.".format(m))
                self._poolings.append(PowerPooling2d(power))
        for i, module in enumerate(self._poolings):
            setattr(self, "pool{}".format(i), module)

    @property
    def channels_multiplier(self):
        return len(self._poolings) if self._aggregate == "cat" else 1

    def forward(self, x):
        results = [pooling(x) for pooling in self._poolings]
        if self._aggregate == "sum":
            result = torch.stack(results).sum(dim=0)
        else:
            assert self._aggregate == "cat"
            result = torch.cat(results, dim=-1)
        return result
