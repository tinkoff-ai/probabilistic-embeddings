from collections import OrderedDict

import torch

from probabilistic_embeddings.config import prepare_config

from .common import DistributionBase, BatchNormNormalizer


class DiracDistribution(DistributionBase):
    """Single-point distribution with infinity density in one point and zero in others."""

    @staticmethod
    def get_default_config(dim=512, spherical=False):
        """Get distribution parameters.

        Args:
            dim: Point dimension.
            spherical: Whether distribution is on sphere or R^n.
        """
        return OrderedDict([
            ("dim", dim),
            ("spherical", spherical)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(self, config)

    @property
    def dim(self):
        """Point dimension."""
        return self._config["dim"]

    @property
    def is_spherical(self):
        """Whether distribution is on sphere or R^n."""
        return self._config["spherical"]

    @property
    def has_confidences(self):
        """Whether distribution has builtin confidence estimation or not."""
        return False

    @property
    def num_parameters(self):
        """Number of distribution parameters."""
        return self._config["dim"]

    def unpack_parameters(self, parameters):
        """Returns dict with distribution parameters."""
        return {
            "mean": self.mean(parameters)
        }

    def pack_parameters(self, parameters):
        """Returns vector from parameters dict."""
        keys = {"mean"}
        if set(parameters) != keys:
            raise ValueError("Expected dict with keys {}.".format(keys))
        if parameters["mean"].shape[-1] != self.dim:
            raise ValueError("Parameters dim mismatch.")
        return parameters["mean"]

    def make_normalizer(self):
        """Create and return normalization layer."""
        return BatchNormNormalizer(self.num_parameters)

    def sample(self, parameters, size=None):
        """Sample from distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).
            size: Sample size (output shape without dimension). Parameters must be broadcastable to the given size.
              If not provided, output shape will be consistent with parameters.

        Returns:
            Tuple of:
                - Samples with shape (..., D).
                - Choosen components with shape (...).
        """
        if size is None:
            size = parameters.shape[:-1]
        means = self.mean(parameters)  # (..., D).
        means = means.broadcast_to(list(size) + [self.dim])
        components = torch.zeros(size, dtype=torch.long, device=parameters.device)
        return means, components

    def mean(self, parameters):
        """Extract mean for each distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Distribution means with shape (..., D).
        """
        if parameters.shape[-1] != self.num_parameters:
            raise ValueError("Unexpected number of parameters: {} != {}.".format(parameters.shape[-1], self.num_parameters))
        means = torch.nn.functional.normalize(parameters, dim=-1) if self._config["spherical"] else parameters
        return means

    def modes(self, parameters):
        """Get modes of distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Tuple of mode log probabilities with shape (..., C) and modes with shape (..., C, D).
        """
        modes = self.mean(parameters).unsqueeze(-2)  # (...., 1, D).
        log_probs = torch.zeros_like(modes[:-1])  # (..., 1).
        return log_probs, modes

    def confidences(self, parameters):
        """Get confidence score for each element of the batch.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Confidences with shape (...).
        """
        raise RuntimeError("Dirac distribution doesn't have confidence.")

    def prior_kld(self, parameters):
        """Get KL-divergence between distributions and prior.

        Is not defined for dirac.
        """
        raise RuntimeError("KLD is meaningless for dirac distribution.")

    def logpdf(self, parameters, x):
        """Compute log density for all points.

        Args:
            parameters: Distribution parameters with shape (..., K).
            points: Points for density evaluation with shape (..., D).

        Returns:
            Log probabilities with shape (...).
        """
        raise RuntimeError("Logpdf can't be estimated for Dirac density since it can be infinity.")

    def logmls(self, parameters1, parameters2):
        """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.

        Args:
            parameters1: Distribution parameters with shape (..., K).
            parameters2: Distribution parameters with shape (..., K).

        Returns:
            MLS scores with shape (...).
        """
        raise RuntimeError("MLS can't be estimated for Dirac density since it can be infinity.")

    def pdf_product(self, parameters1, paramaters2):
        """Compute product of two densities.

        Returns:
            Tuple of new distribution class and it's parameters.
        """
        raise RuntimeError("PDF product can't be estimated for Dirac density since it is unstable.")

    def statistics(self, parameters):
        """Compute useful statistics for logging.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}
