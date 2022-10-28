import math
from collections import OrderedDict
from numbers import Number

import torch

from probabilistic_embeddings.config import prepare_config, ConfigError

from ..parametrization import Parametrization
from .common import DistributionBase, BatchNormNormalizer
from .common import auto_matmul


class NormalDistribution(DistributionBase):
    """Normal distribution.

    Variances are parametrized as input of :meth:`positive` function.
    """

    @staticmethod
    def get_default_config(dim=512, spherical=False, covariance="spherical",
                           parametrization="invlin", min_logivar=None, max_logivar=10):
        """Get Normal distribution parameters.

        Args:
            dim: Point dimension.
            spherical: Whether distribution is on sphere or R^n.
            covariance: Type of covariance matrix (`diagonal`, `spherical` or number).
            parametrization: Type of parametrization (`exp` or `invlin`).
            min_logivar: Minimum value of log inverse variance (log concentration).
            max_logivar: Maximum value of log inverse variance (log concentration).
        """
        return OrderedDict([
            ("dim", dim),
            ("spherical", spherical),
            ("covariance", covariance),
            ("parametrization", parametrization),
            ("min_logivar", min_logivar),
            ("max_logivar", max_logivar)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(self, config)
        if ((self._config["covariance"] not in ["diagonal", "spherical"]) and
            (not isinstance(self._config["covariance"], Number))):
            raise ConfigError("Unknown covariance type: {}".format(self._config["covariance"]))
        if self._config["max_logivar"] is None:
            min_var = 0
        else:
            min_var = math.exp(-self._config["max_logivar"])
        if self._config["min_logivar"] is None:
            max_var = None
        else:
            max_var = math.exp(-self._config["min_logivar"])
        self._parametrization = Parametrization(self._config["parametrization"], min=min_var, max=max_var)

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
        return True

    @property
    def num_parameters(self):
        """Number of distribution parameters."""
        mean_parameters = self._config["dim"]
        if isinstance(self._config["covariance"], Number):
            cov_parameters = 0
        elif self._config["covariance"] == "spherical":
            cov_parameters = 1
        elif self._config["covariance"] == "diagonal":
            cov_parameters = self._config["dim"]
        else:
            assert False
        return mean_parameters + cov_parameters

    def unpack_parameters(self, parameters):
        """Returns dict with distribution parameters."""
        log_probs, means, hidden_vars = self.split_parameters(parameters)
        return {
            "log_probs": log_probs,
            "mean": means,
            "covariance": self._parametrization.positive(hidden_vars)
        }

    def pack_parameters(self, parameters):
        """Returns vector from parameters dict."""
        keys = {"log_probs", "mean", "covariance"}
        if set(parameters) != keys:
            raise ValueError("Expected dict with keys {}.".format(keys))
        hidden_vars = self._parametrization.ipositive(parameters["covariance"])
        return self.join_parameters(parameters["log_probs"], parameters["mean"], hidden_vars)

    def make_normalizer(self):
        """Create and return normalization layer."""
        dim = self._config["dim"]
        return BatchNormNormalizer(self.num_parameters, begin=0, end=dim)

    def split_parameters(self, parameters, normalize=True):
        """Extract component log probs, means and hidden variances from parameters."""
        if parameters.shape[-1] != self.num_parameters:
            raise ValueError("Wrong number of parameters: {} != {}.".format(
                parameters.shape[-1], self.num_parameters))
        dim = self._config["dim"]
        dim_prefix = list(parameters.shape)[:-1]
        scaled_log_probs = torch.zeros(*(dim_prefix + [1]), dtype=parameters.dtype, device=parameters.device)
        means_offset = 0
        means = parameters[..., means_offset:means_offset + dim].reshape(*(dim_prefix + [1, dim]))
        if isinstance(self._config["covariance"], Number):
            with torch.no_grad():
                hidden_covariance = self._parametrization.ipositive(torch.tensor([self._config["covariance"]])).item()
            hidden_vars = torch.full_like(parameters[..., :1], hidden_covariance)
        else:
            hidden_vars = parameters[..., means_offset + dim:]
        hidden_vars = hidden_vars.reshape(*(dim_prefix + [1, -1]))

        if normalize:
            log_probs = scaled_log_probs - torch.logsumexp(scaled_log_probs, dim=-1, keepdim=True)
            means = self._normalize(means)
            return log_probs, means, hidden_vars
        else:
            return scaled_log_probs, means, hidden_vars

    def join_parameters(self, log_probs, means, hidden_vars):
        """Join different GMM parameters into vectors."""
        dim_prefix = list(torch.broadcast_shapes(
            log_probs.shape[:-1],
            means.shape[:-2],
            hidden_vars.shape[:-2]
        ))
        log_probs = log_probs.broadcast_to(*(dim_prefix + list(log_probs.shape[-1:])))
        means = means.broadcast_to(*(dim_prefix + list(means.shape[-2:])))
        flat_parts = []
        flat_parts.extend([means.reshape(*(dim_prefix + [-1]))])
        if isinstance(self._config["covariance"], Number):
            with torch.no_grad():
                hidden_covariance = self._parametrization.ipositive(torch.tensor([self._config["covariance"]],
                                                                                 dtype=hidden_vars.dtype,
                                                                                 device=hidden_vars.device))
            if not torch.allclose(hidden_vars, hidden_covariance):
                raise ValueError("Covariance value changed: {} != {}.".format(
                    self._parametrization.positive(hidden_vars),
                    self._parametrization.positive(hidden_covariance)
                ))
        else:
            hidden_vars = hidden_vars.broadcast_to(*(dim_prefix + list(hidden_vars.shape[-2:])))
            flat_parts.extend([hidden_vars.reshape(*(dim_prefix + [-1]))])
        return torch.cat(flat_parts, dim=-1)

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
        parameters = parameters.reshape(list(parameters.shape[:-1]) + [1] * (len(size) - len(parameters.shape[:-1])) + [parameters.shape[-1]])
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, D).

        # Sample components.
        probs = log_probs.exp().broadcast_to(list(size) + [1])  # (..., C).
        components = torch.multinomial(probs.reshape(-1, 1), 1).reshape(*size)  # (...).
        broad_components = components.unsqueeze(-1).unsqueeze(-1).broadcast_to(list(size) + [1, self.dim])  # (..., 1, D).
        means = means.broadcast_to(list(size) + [1, self.dim])
        means = torch.gather(means, -2, broad_components).squeeze(-2)  # (..., D).
        hidden_vars = hidden_vars.broadcast_to(list(size) + [1, self.dim])
        hidden_vars = torch.gather(hidden_vars, -2, broad_components).squeeze(-2)  # (..., D).

        # Sample from components.
        normal = torch.randn(*(list(size) + [self.dim]), dtype=parameters.dtype, device=parameters.device)  # (..., D).
        stds = self._parametrization.positive(hidden_vars).sqrt()  # (..., D).
        samples = normal * stds + means  # (..., D).
        return samples, components

    def mean(self, parameters):
        """Extract mean for each distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Distribution means with shape (..., D).
        """
        log_probs, means, _ = self.split_parameters(parameters)  # (..., C), (..., C, D).
        means = means.squeeze(-2)
        return means

    def modes(self, parameters):
        """Get modes of distributions.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Tuple of mode log probabilities with shape (..., C) and modes with shape (..., C, D).
        """
        log_probs, means, _ = self.split_parameters(parameters)  # (..., C), (..., C, D).
        return log_probs, means

    def confidences(self, parameters):
        """Get confidence score for each element of the batch.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Confidences with shape (...).
        """
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., 1), (..., 1, D), (..., 1, D).
        logvars = self._parametrization.log_positive(hidden_vars)  # (..., 1, D).
        # Proportional to log entropy.
        return -logvars.mean((-1, -2))  #  (...).

    def prior_kld(self, parameters):
        """Get KL-divergence between distributions and standard normal distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            KL-divergence of each distribution with shape (...).
        """
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., 1), (..., 1, D), (..., 1, D).
        vars = self._parametrization.positive(hidden_vars)  # (..., 1, D).
        logvars = self._parametrization.log_positive(hidden_vars)  # (..., 1, D).
        # There is error in original DUL formula for KLD. Below is true KLD.
        if self._config["covariance"] == "spherical":
            assert logvars.shape[-1] == 1
            logdet = logvars[..., 0] * self.dim  # (..., 1).
            trace = vars[..., 0] * self.dim  # (..., 1).
        else:
            assert self._config["covariance"] == "diagonal"
            assert logvars.shape[-1] == self.dim
            logdet = logvars.sum(dim=-1)  # (..., 1).
            trace = vars.sum(dim=-1)  # (..., 1).
        means_sqnorm = means.square().sum(dim=-1)  # (..., 1).
        kld = 0.5 * (-logdet - self.dim + trace + means_sqnorm)  # (..., 1).
        return kld.squeeze(-1)  # (...).

    def logpdf(self, parameters, points):
        """Compute log density for all points.

        Args:
            parameters: Distribution parameters with shape (..., K).
            points: Points for density evaluation with shape (..., D).

        Returns:
            Log probabilities with shape (...).
        """
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, D).
        vars = self._parametrization.positive(hidden_vars)
        logivars = -self._parametrization.log_positive(hidden_vars)
        c = -self._config["dim"] / 2 * math.log(2 * math.pi)

        points = self._normalize(points)

        # Compute L2 using dot product and torch.nn.linear for better memory usage
        # during broadcasting.
        means_sq_norms = (means.square() / vars).sum(-1)  # (..., C).
        products = auto_matmul(means / vars, points.unsqueeze(-1)).squeeze(-1)  # (..., C).
        if ((self._config["covariance"] == "spherical") or isinstance(self._config["covariance"], Number)):
            assert logivars.shape[-1] == 1
            logidet = logivars[..., 0] * self.dim  # (..., C).
            points_sq_norms = points.unsqueeze(-2).square().sum(-1) / vars.squeeze(-1)  # (..., C).
        else:
            assert self._config["covariance"] == "diagonal"
            assert logivars.shape[-1] == self.dim
            logidet = logivars.sum(dim=-1)  # (..., C).
            points_sq_norms = auto_matmul(1 / vars, points.square().unsqueeze(-1)).squeeze(-1)  # (..., C).
        logexp = products - 0.5 * (means_sq_norms + points_sq_norms)  # (..., C).
        return torch.logsumexp(log_probs + c + 0.5 * logidet + logexp, dim=-1)  # (...).

    def logmls(self, parameters1, parameters2):
        """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.


        Args:
            parameters1: Distribution parameters with shape (..., K).
            parameters2: Distribution parameters with shape (..., K).

        Returns:
            MLS scores with shape (...).
        """
        log_probs1, means1, hidden_vars1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, D).
        log_probs2, means2, hidden_vars2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, D).
        logvars1 = self._parametrization.log_positive(hidden_vars1)
        logvars2 = self._parametrization.log_positive(hidden_vars2)
        pairwise_logmls = self._normal_logmls(
            means1=means1[..., :, None, :],  # (..., C, 1, D).
            logvars1=logvars1[..., :, None, :],  # (..., C, 1, D).
            means2=means2[..., None, :, :],  # (..., 1, C, D).
            logvars2=logvars2[..., None, :, :]  # (..., 1, C, D).
        )  # (..., C, C).
        pairwise_logprobs = log_probs1[..., :, None] + log_probs2[..., None, :]  # (..., C, C).
        dim_prefix = list(pairwise_logmls.shape)[:-2]
        logmls = torch.logsumexp((pairwise_logprobs + pairwise_logmls).reshape(*(dim_prefix + [-1])), dim=-1)  # (...).
        return logmls

    def pdf_product(self, parameters1, parameters2):
        """Compute product of two densities.

        Returns:
            Tuple of new distribution class and it's parameters.
        """

        # Init output distribution type.
        new_config = self._config.copy()
        if isinstance(self._config["covariance"], Number):
            new_config["covariance"] = "spherical"
        new_distribution = NormalDistribution(new_config)

        # Parse inputs.
        log_probs1, means1, hidden_vars1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, D).
        log_probs2, means2, hidden_vars2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, D).
        log_probs1 = log_probs1.unsqueeze(-1)  # (..., C, 1).
        log_probs2 = log_probs2.unsqueeze(-2)  # (..., 1, C).
        means1 = means1.unsqueeze(-2)  # (..., C, 1, D).
        means2 = means2.unsqueeze(-3)  # (..., 1, C, D).
        vars1 = self._parametrization.positive(hidden_vars1).unsqueeze(-2)  # (..., C, 1, D).
        vars2 = self._parametrization.positive(hidden_vars2).unsqueeze(-3)  # (..., 1, C, D).

        # Compute distribution parameters.
        vars_sum = vars1 + vars2  # (..., C, C, D)
        norm_config = self._config.copy()
        if isinstance(self._config["covariance"], Number):
            norm_config["covariance"] = "spherical"
        norm_distribution = NormalDistribution(norm_config)
        norm_means = means1 - means2  # (..., C, C, D).
        norm_parameters = norm_distribution.join_parameters(
            torch.zeros_like(vars_sum[..., :1]),  # (..., C, C).
            norm_means.unsqueeze(-2),  # (..., C, C, 1, D).
            self._parametrization.ipositive(vars_sum).unsqueeze(-2)  # (..., C, C, 1, D).
        )  # (..., C, C).
        new_log_probs = (log_probs1 + log_probs2) + norm_distribution.logpdf(norm_parameters, torch.zeros_like(norm_means))  # (..., C, C).
        new_vars = vars1 / vars_sum * vars2  # (..., C, C, D).
        new_hidden_vars = self._parametrization.ipositive(new_vars)  # (..., C, C, D).
        new_means = vars2 / vars_sum * means1 + vars1 / vars_sum * means2  # (..., C, C, D).
        prefix = tuple(new_means.shape[:-3])
        new_parameters = new_distribution.join_parameters(
            new_log_probs.reshape(*(prefix + (1,))),
            new_means.reshape(*(prefix + (1, -1))),
            new_hidden_vars.reshape(*(prefix + (1, -1)))
        )  # (..., P).
        return new_distribution, new_parameters

    def statistics(self, parameters):
        """Compute useful statistics for logging.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Dictionary with floating-point statistics values.
        """
        parameters = parameters.reshape(-1, parameters.shape[-1])
        log_probs, means, hidden_vars = self.split_parameters(parameters)  # (N, C), (N, D), (N, D).
        stds = self._parametrization.positive(hidden_vars).sqrt()
        return {
            "gmm_std/mean": stds.mean(),
            "gmm_std/std": stds.std()
        }

    def _normal_logmls(self, means1, logvars1, means2, logvars2):
        """Compute Log MLS for unimodal distributions.

        For implementation details see "Probabilistic Face Embeddings":
        https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf
        """
        c = -0.5 * self._config["dim"] * math.log(2 * math.pi)
        delta2 = torch.square(means1 - means2)  # (..., D).
        covsum = logvars1.exp() + logvars2.exp()   # (..., D).
        logcovsum = torch.logaddexp(logvars1, logvars2)  # (..., D).
        mls = c - 0.5 * (delta2 / covsum + logcovsum).sum(-1)  # (...).
        return mls

    def _normalize(self, points):
        return torch.nn.functional.normalize(points, dim=-1) if self.is_spherical else points
