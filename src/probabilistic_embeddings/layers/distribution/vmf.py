import math
from collections import OrderedDict
from numbers import Number

import numpy as np
import scipy
import scipy.special
import torch

from probabilistic_embeddings.config import prepare_config

from ...third_party import sample_vmf
from ..parametrization import Parametrization
from .common import DistributionBase, BatchNormNormalizer
from .common import auto_matmul


K_SEPARATE = "separate"
K_NORM = "norm"


class IveSCLFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, v, z): # computing I_v(z)
        if not isinstance(v, (int, float)):
            raise ValueError("Order must be number, got {}".format(type(v)))
        if v < 0:
            raise NotImplementedError("Negative order: {}.".format(v))

        self.save_for_backward(z)
        self.v = v
        z_cpu = z.data.cpu().numpy()

        if np.isclose(v, 0):
            output = scipy.special.i0e(z_cpu, dtype=z_cpu.dtype)
        elif np.isclose(v, 1):
            output = scipy.special.i1e(z_cpu, dtype=z_cpu.dtype)
        else:
            output = scipy.special.ive(v, z_cpu, dtype=z_cpu.dtype)

        return torch.Tensor(output).to(z.device)

    @staticmethod
    def backward(self, grad_output):
        z = self.saved_tensors[-1]
        return None, grad_output * (IveSCLFunction.apply(self.v - 1, z) - IveSCLFunction.apply(self.v, z) * (self.v + z) / z)


def logiv_scl(v, z, eps=1e-6):
    """Compute log IV using SCL implementation."""
    log_ive = torch.log(eps + IveSCLFunction.apply(v, z))
    log_iv = log_ive + z
    return log_iv


class LogIvFunction(torch.autograd.Function):
    """Differentiable logarithm of modified Bessel function of the first kind.

    Internal computations are done in double precision.

    Inputs:
        - v: Scalar order. Only non-negative values (>= 0) are supported.
        - z: Arguments tensor. Only positive values (> 0) are supported.

    Outputs:
        - Logarithm of modified Bessel function result the same shape as `z`.
    """
    EPS = 1e-16

    @staticmethod
    def forward(ctx, v, z):
        if not isinstance(v, (int, float)):
            raise ValueError("Order must be number, got {}".format(type(v)))
        if v < 0:
            raise NotImplementedError("Negative order.")
        z_numpy = z.double().detach().cpu().numpy()
        ive = torch.from_numpy(scipy.special.ive(v, z_numpy)).to(z.device)
        ctx.saved_v = v
        ctx.saved_z = z_numpy
        ctx.save_for_backward(z, ive)
        logiv = ive.log().to(z.dtype) + z
        logiv_small = -scipy.special.loggamma(v + 1) - v * math.log(2) + v * z.log()
        return torch.maximum(logiv, logiv_small)

    @staticmethod
    def backward(ctx, grad_output):
        v, z_numpy = ctx.saved_v, ctx.saved_z
        z, ive = ctx.saved_tensors
        # d logIV / dz = IVE(v + 1, z) / IVE(v, z) + v / z.
        ive_shifted = torch.from_numpy(scipy.special.ive(v + 1, z_numpy)).to(grad_output.device).to(grad_output.dtype)
        ratio = ive_shifted / ive
        ratio[ratio.isnan()] = 0
        scale = ratio + v / z  # z > 0.
        return None, grad_output * scale


logiv = LogIvFunction.apply


class VMFDistribution(DistributionBase):
    """Von Mises-Fisher Mixture Model.

    For MLS implemenation details see "Spherical Confidence Learning
    for Face Recognition":
    https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spherical_Confidence_Learning_for_Face_Recognition_CVPR_2021_paper.pdf

    Layer supports diffent types of k parametrization. Use "separate"
    to encode k as separate encoder output. Use "norm" to extract k
    from embedding L2 norm. You can also provide fixed k value, which
    will not be changed during training.

    """

    LOGIV = {
        "default": logiv,
        "scl": logiv_scl
    }

    @staticmethod
    def get_default_config(dim=512, k="separate",
                           parametrization="invlin", max_logk=10,
                           logiv_type="default"):
        """Get vMF parameters.

        Args:
            dim: Point dimension.
            k: Type of k parametrization (`separate`, `norm` or number). See class documentation for details.
            parameterization: Type of parametrization (`exp` or `invlin`).
            max_logk: Maximum value of log concentration for "separate" parametrization.
            logiv_type: Algorithm used for log IV computation (`default` or `scl`).
        """
        return OrderedDict([
            ("dim", dim),
            ("k", k),
            ("parametrization", parametrization),
            ("max_logk", max_logk),
            ("logiv_type", logiv_type)
        ])

    def __init__(self, config=None):
        self._config = prepare_config(self, config)
        if self._config["dim"] < 2:
            raise ValueError("Feature space must have dimension >= 2, got {}.".format(self._config["dim"]))
        if ((self._config["k"] not in [K_SEPARATE, K_NORM]) and
            (not isinstance(self._config["k"], Number))):
            raise ValueError("Unknow type of k parametrization: {}.".format(self._config["k"]))

        if self._config["k"] != K_SEPARATE:
            min_ik = 0
        elif self._config["max_logk"] is None:
            min_ik = 0
        else:
            min_ik = math.exp(-self._config["max_logk"])
        self._parametrization = Parametrization(self._config["parametrization"], min=min_ik)
        self._logiv_fn = self.LOGIV[self._config["logiv_type"]]

    @property
    def dim(self):
        """Point dimension."""
        return self._config["dim"]

    @property
    def is_spherical(self):
        """Whether distribution is on sphere or R^n."""
        return True

    @property
    def has_confidences(self):
        """Whether distribution has builtin confidence estimation or not."""
        return True

    @property
    def num_parameters(self):
        """Number of distribution parameters."""
        mean_parameters = self._config["dim"]
        k_parameters = 1 if self._config["k"] == K_SEPARATE else 0
        return mean_parameters + k_parameters

    def unpack_parameters(self, parameters):
        """Returns dict with distribution parameters."""
        log_probs, means, hidden_ik = self.split_parameters(parameters)
        return {
            "log_probs": log_probs,
            "mean": means,
            "k": 1 / self._parametrization.positive(hidden_ik)
        }

    def pack_parameters(self, parameters):
        """Returns vector from parameters dict."""
        keys = {"log_probs", "mean", "k"}
        if set(parameters) != keys:
            raise ValueError("Expected dict with keys {}.".format(keys))
        hidden_ik = self._parametrization.ipositive(1 / parameters["k"])
        return self.join_parameters(parameters["log_probs"], parameters["mean"], hidden_ik)

    def make_normalizer(self):
        """Create and return normalization layer."""
        dim = self._config["dim"]
        if self._config["k"] == K_NORM:
            normalizer = None
        else:
            normalizer = BatchNormNormalizer(self.num_parameters, begin=0, end=dim)
        return normalizer

    def split_parameters(self, parameters, normalize=True):
        """Extract log probs, means and inverse k from parameters."""
        if parameters.shape[-1] != self.num_parameters:
            raise ValueError("Wrong number of parameters: {} != {}.".format(
                parameters.shape[-1], self.num_parameters))
        dim = self._config["dim"]
        dim_prefix = list(parameters.shape)[:-1]
        scaled_log_probs = torch.zeros(*(dim_prefix + [1]), dtype=parameters.dtype, device=parameters.device)
        means_offset = 0
        scaled_means = parameters[..., means_offset:means_offset + 1 * dim].reshape(*(dim_prefix + [1, dim]))
        if isinstance(self._config["k"], Number):
            ik = torch.full((dim_prefix + [1, 1]), 1 / self._config["k"], dtype=parameters.dtype, device=parameters.device)
            hidden_ik = self._parametrization.ipositive(ik)
        elif self._config["k"] == K_SEPARATE:
            hidden_ik = parameters[..., means_offset + 1 * dim:].reshape(*(dim_prefix + [1, 1]))
        else:
            assert self._config["k"] == K_NORM
            k = torch.linalg.norm(scaled_means, dim=-1, keepdim=True)  # (..., C, 1).
            hidden_ik = self._parametrization.ipositive(1 / k)

        if normalize:
            log_probs = scaled_log_probs - torch.logsumexp(scaled_log_probs, dim=-1, keepdim=True)
            means = self._normalize(scaled_means)
            return log_probs, means, hidden_ik
        else:
            return scaled_log_probs, scaled_means, hidden_ik

    def join_parameters(self, log_probs, means, hidden_ik):
        """Join different vMF parameters into vectors."""
        # Denormalize k. See class documentation.
        dim_prefix = list(torch.broadcast_shapes(
            log_probs.shape[:-1],
            means.shape[:-2],
            hidden_ik.shape[:-2]
        ))
        means = means.broadcast_to(*(dim_prefix + list(means.shape[-2:])))
        hidden_ik = hidden_ik.broadcast_to(*(dim_prefix + list(hidden_ik.shape[-2:])))
        flat_parts = []
        if isinstance(self._config["k"], Number):
            ik = self._parametrization.positive(hidden_ik)
            if not ((ik - 1 / self._config["k"]).abs() < 1e-6).all():
                raise ValueError("All k must be equal to {} for fixed k parametrization".format(self._config["k"]))
            flat_parts.append(means.reshape(*(dim_prefix + [-1])))
        elif self._config["k"] == K_SEPARATE:
            flat_parts.extend([means.reshape(*(dim_prefix + [-1])),
                               hidden_ik.reshape(*(dim_prefix + [-1]))])
        else:
            assert self._config["k"] == K_NORM
            scaled_means = torch.nn.functional.normalize(means, dim=-1) / self._parametrization.positive(hidden_ik)
            flat_parts.append(scaled_means.reshape(*(dim_prefix + [-1])))
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
                - Means with shape (...).
        """
        if size is None:
            size = parameters.shape[:-1]
        parameters = parameters.reshape(list(parameters.shape[:-1]) + [1] * (len(size) - parameters.ndim + 1) + [parameters.shape[-1]])
        log_probs, means, hidden_ik = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, 1).
        probs = log_probs.exp().broadcast_to(list(size) + [1])  # (..., C).
        components = torch.multinomial(probs.reshape(-1, 1), 1).reshape(*size)  # (...).
        broad_components = components.unsqueeze(-1).unsqueeze(-1).broadcast_to(list(size) + [1, self.dim])  # (..., 1, D).
        means = means.broadcast_to(list(size) + [1, self.dim])
        means = torch.gather(means, -2, broad_components).squeeze(-2)  # (..., D).
        hidden_ik = hidden_ik.broadcast_to(list(size) + [1, 1])
        hidden_ik = torch.gather(hidden_ik, -2, broad_components[..., :1]).squeeze(-2)  # (..., D).
        k = 1 / self._parametrization.positive(hidden_ik)  # (..., D).
        samples = sample_vmf(means, k, size)
        return samples, components

    def mean(self, parameters):
        """Extract mean for each distribution.

        Args:
            parameters: Distribution parameters with shape (..., K).

        Returns:
            Distribution means with shape (..., D).
        """
        log_probs, means, hidden_ik = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, 1).
        k = 1 / self._parametrization.positive(hidden_ik)
        half_dim = self._config["dim"] / 2
        component_means = means * (self._logiv_fn(half_dim, k) - self._logiv_fn(half_dim - 1, k)).exp()  # (..., C, D).
        means = component_means.squeeze(-2)
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
        log_priors, means, hidden_ik = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, 1).
        logik = self._parametrization.log_positive(hidden_ik)
        return -logik.mean((-1, -2))  #  (...).

    def prior_kld(self, parameters):
        """Get KL-divergence between distributions and prior.

        Warning: This is not true KLD, but just simple regularizer
        on concentration parameter of vMF distribution.
        """
        log_priors, means, hidden_ik = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, 1).
        assert hidden_ik.shape[-1] == 1
        k = 1 / self._parametrization.positive(hidden_ik)  # (..., 1, D).
        logk = -self._parametrization.log_positive(hidden_ik)  # (..., 1, D).
        kld = k + self._vmf_logc(k, logk=logk) - self._vmf_logc(1e-6)
        return kld.squeeze(-1)  # (...).

    def logpdf(self, parameters, points):
        """Compute log density for all points after normalization.

        Args:
            parameters: Distribution parameters with shape (..., K).
            points: Points for density evaluation with shape (..., D).

        Returns:
            Log probabilities with shape (...).
        """
        log_priors, means, hidden_ik = self.split_parameters(parameters)  # (..., C), (..., C, D), (..., C, 1).
        k = 1 / self._parametrization.positive(hidden_ik)
        logk = -self._parametrization.log_positive(hidden_ik)
        points = self._normalize(points)  # (..., D).
        logc = self._vmf_logc(k, logk=logk)  # (..., C, 1).
        scaled_means = k * means  # (..., C, D).
        logexp = auto_matmul(scaled_means, points.unsqueeze(-1)).squeeze(-1)  # (..., C).
        return torch.logsumexp(log_priors + logc.squeeze(-1) + logexp, dim=-1)  # (...).

    def logmls(self, parameters1, parameters2):
        """Compute Log Mutual Likelihood Score (MLS) for pairs of distributions.


        Args:
            parameters1: Distribution parameters with shape (..., K).
            parameters2: Distribution parameters with shape (..., K).

        Returns:
            MLS scores with shape (...).
        """
        log_probs1, means1, hidden_ik1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, 1).
        log_probs2, means2, hidden_ik2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, 1).
        pairwise_logmls = self._vmf_logmls(
            means1=means1[..., :, None, :],  # (..., C, 1, D).
            hidden_ik1=hidden_ik1[..., :, None, :],  # (..., C, 1, 1).
            means2=means2[..., None, :, :],  # (..., 1, C, D).
            hidden_ik2=hidden_ik2[..., None, :, :]  # (..., 1, C, 1).
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
        new_distribution = VMFDistribution(new_config)

        # Parse inputs.
        log_probs1, means1, hidden_ik1 = self.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, D).
        log_probs2, means2, hidden_ik2 = self.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, D).
        log_probs1 = log_probs1.unsqueeze(-1)  # (..., C, 1).
        log_probs2 = log_probs2.unsqueeze(-2)  # (..., 1, C).
        means1 = means1.unsqueeze(-2)  # (..., C, 1, D).
        means2 = means2.unsqueeze(-3)  # (..., 1, C, D).
        ik1 = self._parametrization.positive(hidden_ik1).unsqueeze(-2)  # (..., C, 1, D).
        ik2 = self._parametrization.positive(hidden_ik2).unsqueeze(-3)  # (..., 1, C, D).

        # Compute distribution parameters.
        new_means = means1 / ik1 + means2 / ik2  # (..., C, C, D).
        new_k = torch.linalg.norm(new_means, dim=-1, keepdim=True)  # (..., C, C, 1).
        new_means = new_means / new_k
        log_norms = (self._vmf_logc(1 / ik1) + self._vmf_logc(1 / ik2) - self._vmf_logc(new_k)).squeeze(-1)  # (..., C, C).
        new_log_probs = (log_probs1 + log_probs2) + log_norms  # (..., C, C).
        new_hidden_ik = self._parametrization.ipositive(1 / new_k)  # (..., C, C, D).
        prefix = tuple(new_means.shape[:-3])
        new_parameters = self.join_parameters(
            new_log_probs.reshape(*(prefix + (1,))),
            new_means.reshape(*(prefix + (1, -1))),
            new_hidden_ik.reshape(*(prefix + (1, -1)))
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
        log_priors, means, hidden_ik = self.split_parameters(parameters)  # (N, C), (N, D), (N, D).
        sqrt_ik = self._parametrization.positive(hidden_ik).sqrt()
        return {
            "vmf_sqrt_inv_k/mean": sqrt_ik.mean(),
            "vmf_sqrt_inv_k/std": sqrt_ik.std()
        }

    def _normalize(self, points):
        """Project points to sphere."""
        result = torch.nn.functional.normalize(points, dim=-1)
        return result

    def _vmf_logc(self, k, logk=None):
        if isinstance(k, (float, np.floating)):
            return self._vmf_logc(torch.full((1,), k))[0].item()
        if k.ndim == 0:
            return self._vmf_logc(k[None])[0]
        if logk is None:
            logk = k.log()
        half_dim = self._config["dim"] / 2
        lognum = (half_dim - 1) * logk
        logden = half_dim * math.log(2 * math.pi) + self._logiv_fn(half_dim - 1, k)
        small_mask = torch.logical_or(lognum.isneginf(), logden.isneginf())
        logc_small = torch.tensor(-self._log_unit_area()).to(k.dtype).to(k.device)
        return torch.where(small_mask, logc_small, lognum - logden)

    def _vmf_logmls(self, means1, hidden_ik1, means2, hidden_ik2):
        """Compute Log MLS for unimodal distributions."""
        k1 = 1 / self._parametrization.positive(hidden_ik1)
        k2 = 1 / self._parametrization.positive(hidden_ik2)
        logk1 = -self._parametrization.log_positive(hidden_ik1)
        logk2 = -self._parametrization.log_positive(hidden_ik2)
        k = torch.linalg.norm(k1 * means1 + k2 * means2, dim=-1, keepdim=True)  # (..., 1).
        logc1 = self._vmf_logc(k1, logk=logk1)
        logc2 = self._vmf_logc(k2, logk=logk2)
        logc = self._vmf_logc(k)
        return (logc1 + logc2 - logc).squeeze(-1)

    def _log_unit_area(self):
        """Logarithm of the unit sphere area."""
        dim = self._config["dim"]
        return math.log(2) + dim / 2 * math.log(math.pi) - scipy.special.loggamma(dim / 2)
