from abc import ABC, abstractmethod

import torch


def auto_matmul(A, B):
    """Broadcastable matrix multiplication which uses fastest backend possible."""
    # For (*, 1, a, k) x (1, ..., 1, b, k, 1) use torch.nn.functional.linear.
    if (A.ndim >= 3 and B.ndim >= 3 and
        A.shape[-3] == 1 and B.shape[-1] == 1 and
        all(dim == 1 for dim in B.shape[:-3])):
        b, k = B.shape[-3:-1]
        A = A.squeeze(-3)  # (*, a, k).
        B = B.reshape(b, k)  # (b, k).
        result = torch.nn.functional.linear(A, B)  # (*, a, b).
        result = result.transpose(-1, -2).unsqueeze(-1)  # (*, b, a, 1).
        return result
    # TODO: Use torch.bmm when the number of elements in prefixes matches.
    # Otherwise use torch.matmul.
    return torch.matmul(A, B)


class DistributionBase(ABC):
    """Base class for all distribution models."""
    @property
    @abstractmethod
    def dim(self):
        """Point dimension."""
        pass

    @property
    @abstractmethod
    def is_spherical(self):
        """Whether distribution is on sphere or R^n."""
        pass

    @property
    @abstractmethod
    def has_confidences(self):
        """Whether distribution has builtin confidence estimation or not."""
        pass

    @property
    @abstractmethod
    def num_parameters(self):
        """Number of distribution parameters."""
        pass

    @abstractmethod
    def unpack_parameters(self, parameters):
        """Returns dict with distribution parameters."""
        pass

    @abstractmethod
    def pack_parameters(self, parameters):
        """Returns vector from parameters dict."""
        pass

    @abstractmethod
    def make_normalizer(self):
        """Create and return normalization layer."""
        pass

    @abstractmethod
    def sample(self, parameters, size=None):
        """Sample from distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).
            size: Sample size (output shape without dimension). Parameters must be broadcastable to the given size.
              If not provided, output shape will be consistent with parameters.

        Returns:
            Tuple of:
                - Samples with shape (..., D).
                - Choosen mixture components with shape (...).
        """
        pass

    @abstractmethod
    def mean(self, parameters):
        """Extract mean for each distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Distribution means with shape (..., D).
        """
        pass

    @abstractmethod
    def modes(self, parameters):
        """Get modes of distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Tuple of mode log probabilities with shape (..., C) and modes with shape (..., C, D).
        """
        pass

    def mode(self, parameters):
        """Get mode of distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Modes with shape (..., D).
        """
        _, modes = self.modes(parameters)  # (..., C, D).
        modes_logpdfs = self.logpdf(parameters.unsqueeze(-2), modes)  # (..., C).
        max_modes = modes_logpdfs.argmax(-1, keepdim=True)  # (..., 1).
        modes = modes.take_along_dim(max_modes.unsqueeze(-1), -2).squeeze(-2)  # (..., D).
        return modes

    @abstractmethod
    def confidences(self, parameters):
        """Get confidence score for each element of the batch.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Confidences with shape (...).
        """
        pass

    @abstractmethod
    def prior_kld(self, parameters):
        """Get KL-divergence between distributions and prior."""
        pass

    @abstractmethod
    def logpdf(self, parameters, points):
        """Compute log density for all points.

        Args:
            parameters: Distribution parameters with shape (..., K).
            points: Points for density evaluation with shape (..., D).

        Returns:
            Log probabilities with shape (...).
        """
        pass

    @abstractmethod
    def logmls(self, parameters1, parameters2):
        """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.

        Args:
            parameters1: Distribution parameters with shape (..., K).
            parameters2: Distribution parameters with shape (..., K).

        Returns:
            Log MLS scores with shape (...).
        """
        pass

    @abstractmethod
    def pdf_product(self, parameters1, paramaters2):
        """Compute product of two densities.

        Returns:
            Tuple of new distribution class and it's parameters.
        """
        pass

    @abstractmethod
    def statistics(self, parameters):
        """Compute useful statistics for logging.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Dictionary with floating-point statistics values.
        """
        pass


class BatchNormNormalizer(torch.nn.BatchNorm1d):
    """Applies batchnorm to the part of input tensor. Useful for centroids normalization.

    Suppose input tensor has shape (..., K), where K is the number of parameters.
    First slice (..., begin:end) is extracted, then it is reshaped to (..., C, D), where
    C is the number of components. Batchnorm is applied along last dimension D.
    """
    def __init__(self, num_parameters, begin=0, end=None, train_scale=False, train_bias=True):
        begin = begin % num_parameters
        end = end % (num_parameters + 1) if end is not None else num_parameters
        if begin >= end:
            raise ValueError("Begin is greater or equal to end")
        super().__init__(end - begin)
        if not train_scale:
            torch.nn.init.constant_(self.weight, 1.0)
            self.weight.requires_grad = False
        if not train_bias:
            torch.nn.init.constant_(self.bias, 0.0)
            self.bias.requires_grad = False
        self._num_parameters = num_parameters
        self._begin = begin
        self._end = end

    def forward(self, input):
        if input.shape[-1] != self._num_parameters:
            raise ValueError("Unexpected input dimension")
        subset = input[..., self._begin:self._end]  # (..., C * D).
        subset = subset.reshape(-1, self._end - self._begin)  # (N, C * D).
        normalized_subset = super().forward(subset)  # (N, C * D).
        normalized_subset = normalized_subset.reshape(*(list(input.shape)[:-1] + [-1]))  # (..., C * D).
        result = torch.cat([
            input[..., :self._begin],
            normalized_subset,
            input[..., self._end:]
        ], dim=-1)
        assert result.shape == input.shape
        return result
