import torch


class Parametrization:
    """Mapping from real numbers to non-negative ones and vise-versa.

    Args:
        type: Type of parametrization (`exp`, `invlin`, `abs` or `sigmoid`).
        min: Minimum positive value.
        max: Maximum value for sigmoid parametrization.
        center: Shift values prior to positive transform.
        scale: Scale tangent slop at the center.
    """

    def __init__(self, type, min=0, max=None, center=0, scale=1):
        if type not in {"exp", "invlin", "abs", "sigmoid"}:
            raise ValueError("Unknown parametrization: {}.".format(type))
        if (max is not None) and (type != "sigmoid"):
            raise ValueError("Maximum is supported for sigmoid parametrization only.")
        if (max is None) and (type == "sigmoid"):
            raise ValueError("Maximum value must be provided for sigmoid parametrization.")
        self._type = type
        self._min = min
        self._max = max
        self._center = center
        self._scale = scale

    def positive(self, x):
        """Smooth mapping from real to positive numbers."""
        x = self._linear(x)
        if self._type == "exp":
            return self._exp(x, min=self._min)
        elif self._type == "invlin":
            return self._invlin(x, min=self._min)
        elif self._type == "sigmoid":
            return self._sigmoid(x, min=self._min, max=self._max)
        elif self._type == "abs":
            return self._abs(x, min=self._min)
        else:
            assert False

    def log_positive(self, x):
        """Logarithm of positive function."""
        x = self._linear(x)
        if self._type == "exp":
            return self._log_exp(x, min=self._min)
        elif self._type == "invlin":
            return self._log_invlin(x, min=self._min)
        elif self._type == "sigmoid":
            return self._log_sigmoid(x, min=self._min, max=self._max)
        elif self._type == "abs":
            return self._log_abs(x, min=self._min)
        else:
            assert False

    def ipositive(self, x):
        """Inverse of positive function."""
        if self._type == "exp":
            x = self._iexp(x, min=self._min)
        elif self._type == "invlin":
            x=  self._iinvlin(x, min=self._min)
        elif self._type == "sigmoid":
            x = self._isigmoid(x, min=self._min, max=self._max)
        elif self._type == "abs":
            x=  self._iabs(x, min=self._min)
        else:
            assert False
        x = self._ilinear(x)
        return x

    def _linear(self, x):
        if self._scale != 1:
            x = x / self._scale
        if self._center != 0:
            x = x - self._center
        return x

    def _ilinear(self, x):
        if self._center != 0:
            x = x + self._center
        if self._scale != 1:
            x = x * self._scale
        return x

    @staticmethod
    def _exp(x, min=0):
        """Smooth mapping from real to positive numbers."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        result = x.exp()
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_exp(x, min=0):
        """Logarithm of exponential function with min."""
        result = x
        if min > 0:
            min = torch.tensor(min, dtype=x.dtype, device=x.device)
            result = torch.logaddexp(x, min.log())
        return result

    @staticmethod
    def _iexp(x, min=0):
        """Inverse of exp function with min."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        if min > 0:
            x = x - min
        return x.log()

    @staticmethod
    def _invlin(x, min=0):
        """Smooth mapping from real to positive numbers.

        Inverse function for x < 0 and linear for x > 0.
        """
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        result = torch.where(x < 0, 1 / (1 - x.clip(max=0)), 1 + x)  # Clip max to prevent NaN gradient for x = 1.
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_invlin(x, min=0):
        """Logarithm of invlin function."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        is_negative = x < 0
        nxp1 = 1 - x
        xp1 = 1 + x
        if min > 0:
            xp1 = xp1 + min
        result = torch.where(is_negative, -nxp1.log(), xp1.log())
        if min > 0:
            nxp1ge1 = torch.clip(nxp1, min=1)
            result = result + is_negative * (1 + min * nxp1ge1).log()
        return result

    @staticmethod
    def _iinvlin(x, min=0):
        """Inverse of invlin."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        if min > 0:
            x = x - min
        return torch.where(x < 1, 1 - 1 / x, x - 1)

    @staticmethod
    def _abs(x, min=0):
        """Mapping from real to positive numbers."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        result = x.abs()
        if min > 0:
            result = result + min
        return result

    @staticmethod
    def _log_abs(x, min=0):
        """Logarithm of abs function."""
        return Parametrization._abs(x, min=min).log()

    @staticmethod
    def _iabs(x, min=0):
        """Inverse of abs (true inverse for positives only)."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        if min > 0:
            x = x - min
        return x

    @staticmethod
    def _sigmoid(x, min=0, max=1):
        """Smooth mapping from real to positive numbers."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        if min >= max:
            raise ValueError("Minimum must be less than maximum.")
        result = torch.sigmoid(x) * (max - min) + min
        return result

    @staticmethod
    def _log_sigmoid(x, min=0, max=1):
        """Logarithm of sigmoid function."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        if min >= max:
            raise ValueError("Minimum must be less than maximum.")
        result = torch.log(torch.sigmoid(x) * (max - min) + min)
        return result

    @staticmethod
    def _isigmoid(x, min=0, max=1):
        """Inverse sigmoid."""
        if min < 0:
            raise ValueError("Only non-negative minimum is supported.")
        if min >= max:
            raise ValueError("Minimum must be less than maximum.")
        result = torch.logit((x - min) / (max - min), eps=1-6)
        return result
