from collections import OrderedDict
import math

import torch

from .._workarounds import ArcFace, CosFace
from ..config import prepare_config, ConfigError
from .distribution import NormalDistribution, VMFDistribution
from .parametrization import Parametrization


def get_log_priors(num_classes, priors=None):
    """Create new log priors tensor.

    Args:
        num_classes: Numder of classes.
        priors: Initial value for priors.

    Returns:
        Parameter if trainable is True and Tensor otherwise.
    """
    if priors is not None:
        if not isinstance(priors, torch.Tensor):
            priors = torch.tensor(priors)
        if priors.shape != (num_classes,):
            raise ValueError("Expected initial priors with shape ({},), got: {}.".format(num_classes, priors.shape))
        log_priors = priors.float().log()
    else:
        log_priors = torch.zeros(num_classes)
    return log_priors


def additive_margin(logits, labels=None, margin=0):
    """Add margin if labels are provided."""
    if (margin != 0) and (labels is not None):
        one_hot = torch.zeros_like(logits)  # (..., K).
        one_hot.scatter_(-1, labels.unsqueeze(-1).long(), 1)
        logits = logits - one_hot * margin
    return logits


class LinearClassifier(torch.nn.Linear):
    """Simple classification head based on linear layer.

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.
        priors (unused): Precomputed class priors. Priors can be learned on-line if not provided.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: Unused.
        - scorer: Unused.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    @staticmethod
    def get_default_config(sample=True, use_bias=True, initial_scale=1,
                           normalize_weights=False,
                           use_variance=False, initial_variance=1, variance_parametrization="exp",
                           freeze_variance=False, variance_center=0, variance_scale=1):
        """Get classifier config.

        Args:
            sample: If True, sample from distribution. Use distribution mean otherwise.
            use_bias: Whether to use bias in linear layer or not.
            initial_scale: Scale parameters during initialization.
            normalize_weights: Normalize weights before applying.
            use_variance: Whether to add trainable embeddings variance or not.
            initial_variance: Initial value of the variance.
            variance_parametrization: Type of variance coding ("exp" or "invlin").
            freeze_variance: Don't train variance parameter.
            variance_center: Parametrization center.
            variance_scale: Parametrization scale.
        """
        return OrderedDict([
            ("sample", sample),
            ("use_bias", use_bias),
            ("initial_scale", initial_scale),
            ("normalize_weights", normalize_weights),
            ("use_variance", use_variance),
            ("initial_variance", initial_variance),
            ("variance_parametrization", variance_parametrization),
            ("freeze_variance", freeze_variance),
            ("variance_center", variance_center),
            ("variance_scale", variance_scale)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        config = prepare_config(self, config)
        super().__init__(distribution.dim, num_classes, bias=config["use_bias"])
        self._config = config
        self._distribution = distribution
        self._num_classes = num_classes
        if self._config["initial_scale"] != 1:
            self.weight.data *= self._config["initial_scale"]
            if self._config["use_bias"]:
                self.bias.data *= self._config["initial_scale"]
        if self._config["use_variance"]:
            self._variance_parametrization = Parametrization(self._config["variance_parametrization"],
                                                             center=self._config["variance_center"],
                                                             scale=self._config["variance_scale"])
            initial_variance = float(self._config["initial_variance"])
            initial_hidden_variance = self._variance_parametrization.ipositive(torch.full([], initial_variance)).item()
            self.hidden_variance = torch.nn.Parameter(torch.full([], initial_hidden_variance, dtype=torch.float),
                                                      requires_grad=not self._config["freeze_variance"])

    @property
    def has_weight(self):
        return True

    @property
    def has_bias(self):
        return self._config["use_bias"]

    @property
    def has_variance(self):
        return self._config["use_variance"]

    @property
    def variance(self):
        return self._variance_parametrization.positive(self.hidden_variance)

    def clip_variance(self, max):
        max_hidden = self._variance_parametrization.ipositive(torch.tensor(max)).item()
        self.hidden_variance.data.clip_(max=max_hidden)

    def set_variance(self, value):
        hidden = self._variance_parametrization.ipositive(torch.tensor(value)).item()
        self.hidden_variance.data.fill_(hidden)

    def forward(self, parameters, labels=None, scorer=None):
        if self._config["sample"]:
            embeddings, _ = self._distribution.sample(parameters)  # (..., D).
        else:
            embeddings = self._distribution.mean(parameters)  # (..., D).
        if self._config["normalize_weights"]:
            weight = self.weight / torch.linalg.norm(self.weight.flatten())
            bias = self.bias
        else:
            weight, bias = self.weight, self.bias
        logits = torch.nn.functional.linear(embeddings, weight, bias)
        return logits

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )


class ArcFaceClassifier(ArcFace):
    """ArcFace classification head with trainable target classes centers.

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.
        priors (unused): Precomputed class priors. Priors can be learned on-line if not provided.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: If provided, used for ArcFace logit correction. Compute cosine otherwise.
        - scorer: Unused.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    @staticmethod
    def get_default_config(sample=True, scale=64.0, margin=0.5):
        """Get classifier config.

        Args:
            sample: If True, sample from distribution. Use distribution mean otherwise.
            scale: Output scale (number or "trainable").
            margin: ArcFace margin.
        """
        return OrderedDict([
            ("sample", sample),
            ("scale", scale),
            ("margin", margin)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        if not distribution.is_spherical:
            raise ValueError("Spherical distrubution is expected.")
        config = prepare_config(self, config)
        scale = torch.nn.Parameter(torch.ones([])) if config["scale"] == "trainable" else config["scale"]
        super().__init__(distribution.dim, num_classes, m=config["margin"], s=scale)
        self._config = config
        self._distribution = distribution
        self._num_classes = num_classes

    @property
    def has_weight(self):
        return True

    @property
    def has_bias(self):
        return False

    @property
    def has_variance(self):
        return False

    def forward(self, parameters, labels=None, scorer=None):
        if self._config["sample"]:
            embeddings, _ = self._distribution.sample(parameters)  # (..., D).
        else:
            embeddings = self._distribution.mean(parameters)  # (..., D).
        dim_prefix = list(parameters.shape)[:-1]
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        labels = labels.flatten() if labels is not None else None
        logits = super().forward(embeddings, target=labels)
        return logits.reshape(*[dim_prefix + [self._num_classes]])

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        scale = self.s.item() if self._config["scale"] == "trainable" else self.s
        return {"scale": scale}

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )


class CosFaceClassifier(CosFace):
    """CosFace classification head with trainable target classes centers.

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.
        priors (unused): Precomputed class priors. Priors can be learned on-line if not provided.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: If provided, used for logit correction. Compute cosine otherwise.
        - scorer: Unused.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    @staticmethod
    def get_default_config(scale=64.0, margin=0.35, symmetric=False):
        """Get classifier config.

        Args:
            scale: Output scale.
            margin: CosFace margin.
            symmetric: If true, add margin to negatives (useful for Proxy-Anchor loss).
        """
        return OrderedDict([
            ("scale", scale),
            ("margin", margin),
            ("symmetric", symmetric)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        if not distribution.is_spherical:
            raise ValueError("Spherical distrubution is expected.")
        config = prepare_config(self, config)
        super().__init__(distribution.dim, num_classes, m=config["margin"], s=config["scale"])
        self._config = config
        self._distribution = distribution
        self._num_classes = num_classes

    @property
    def has_weight(self):
        return True

    @property
    def has_bias(self):
        return False

    @property
    def has_variance(self):
        return False

    def forward(self, parameters, labels=None, scorer=None):
        dim_prefix = list(parameters.shape)[:-1]
        embeddings, _ = self._distribution.sample(parameters)  # (..., D).
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        labels = labels.flatten() if labels is not None else None
        logits = super().forward(embeddings, target=labels)
        if self._config["symmetric"]:
            logits += 0.5 * self._config["margin"] * self._config["scale"]
        return logits.reshape(*[dim_prefix + [self._num_classes]])

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )


class LogLikeClassifier(torch.nn.Module):
    """Contains target centroids and performs log likelihood estimation.

    Layer can add prior correction in different forms. If "pretrained"
    is used, log priors from training set are added to logits. If
    "trainable" is used, bias vector is trained for output logits. By
    default prior correction is turned off.

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.
        priors: Precomputed class priors. Priors can be learned on-line if not provided.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: Positive labels used for margin with shape (...).
        - scorer: Unused.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    TARGET_DISTRIBUTIONS = {
        "gmm": NormalDistribution,
        "vmf": VMFDistribution
    }

    @staticmethod
    def get_default_config(priors=None, margin=0,
                           target_distribution=None, target_distribution_params=None):
        """Get classifier config.

        Args:
            priors: Type of prior correction used (one of `pretrained`, `trainable` and `none`).
              See description above. By default turned off.
            margin: Log probability subtracted from positive logit.
            target_distribution: Compute likelihood of the prediction using target distributions.
              Default is to compute likelihood of the target using predicted distribution.
        """
        return OrderedDict([
            ("priors", priors),
            ("margin", margin),
            ("target_distribution", target_distribution),
            ("target_distribution_params", target_distribution_params)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._num_classes = num_classes

        if self._config["target_distribution"] is not None:
            self._target_distribution = self.TARGET_DISTRIBUTIONS[self._config["target_distribution"]](
                config=self._config["target_distribution_params"])
            if self._target_distribution.dim != distribution.dim:
                raise ConfigError("Predicted and target embeddings size mismatch: {} != {}.".format(
                    distribution.dim, self._target_distribution.dim
                ))
            if self._target_distribution.is_spherical != distribution.is_spherical:
                raise ConfigError("Predicted and target embeddings normalization mismatch")
            self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, self._target_distribution.num_parameters))
        else:
            self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, distribution.dim))
        torch.nn.init.xavier_uniform_(self.weight)

        if self._config["priors"] in [None, "none"]:
            self.bias = None
        else:
            with torch.no_grad():
                log_priors = get_log_priors(num_classes, priors)
            if self._config["priors"] == "pretrained":
                if priors is None:
                    raise ValueError("Need dataset priors for pretrained mode")
                trainable = False
            elif self._config["priors"] != "trainable":
                trainable = True
            else:
                raise ConfigError("Unknown priors mode: {}.".format(self._config["priors"]))
            self.bias = torch.nn.Parameter(log_priors, requires_grad=trainable)

    @property
    def has_weight(self):
        return True

    @property
    def has_bias(self):
        return self.bias is not None

    @property
    def has_variance(self):
        return False

    def forward(self, parameters, labels=None, scorer=None):
        if (labels is not None) and (labels.shape != parameters.shape[:-1]):
            raise ValueError("Parameters and labels shape mismatch: {}, {}".format(
                parameters.shape, labels.shape))
        dim_prefix = list(parameters.shape)[:-1]
        targets = self.weight.reshape(*([1] * len(dim_prefix) + list(self.weight.shape)))  # (..., C, D or P).
        if self._config["target_distribution"] is None:
            parameters = parameters.unsqueeze(-2)  # (..., 1, K).
            logits = self._distribution.logpdf(parameters, targets)  # (..., C).
        else:
            embeddings = self._distribution.sample(parameters)[0].unsqueeze(-2)  # (..., 1, D).
            logits = self._target_distribution.logpdf(targets, embeddings)  # (..., C).
        if self.bias is not None:
            log_priors = self.bias - torch.logsumexp(self.bias, 0)  # (C).
            logits = log_priors + logits
        logits = additive_margin(logits, labels, self._config["margin"])
        return logits

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        result = {}
        if self._config["target_distribution"] is not None:
            confidences = self._target_distribution.confidences(self.weight)
            result["target_confidence/mean"] = confidences.mean()
            result["target_confidence/std"] = confidences.std()
        return result

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )


class VMFClassifier(torch.nn.Module):
    """Contains target centroids distribution and evaluates expected log likelihood.

    Implementation is based on "Von Mises–Fisher Loss:An Exploration of Embedding Geometries for Supervised Learning." (2021).

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: Positive labels used for margin with shape (...).
        - scorer: Unused.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    @staticmethod
    def get_default_config(scale="trainable", initial_log_scale=2.773, kappa_confidence=0.7, sample_size=10,
                           approximate_logc=True, deterministic_target=False):
        """Get classifier config.

        Args:
            scale: Output scale (number or "trainable").
            initial_log_scale: Initial logarithm of scale value when scale is trainable.
            kappa_confidence: Hyperparameter used for initialization and scoring.
            sample_size: Number of samples for probability estimation.
            approximate_logc: Use approximation from the paper to speedup training.
            deterministic_target: Use a variation of vMF-loss with deterministic target embeddings.
        """
        return OrderedDict([
            ("scale", scale),
            ("initial_log_scale", initial_log_scale),
            ("kappa_confidence", kappa_confidence),
            ("sample_size", sample_size),
            ("approximate_logc", approximate_logc),
            ("deterministic_target", deterministic_target)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        if not isinstance(distribution, VMFDistribution):
            raise ValueError("Expected vMF distribution for vMF loss.")
        super().__init__()
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._num_classes = num_classes

        l = self._config["kappa_confidence"]
        dim = distribution.dim
        if self._config["deterministic_target"]:
            means = torch.randn(num_classes, dim) * l / (1 - l * l) * (dim - 1) / math.sqrt(dim)
            self.weight = torch.nn.Parameter(means)
        else:
            means = torch.randn(num_classes, 1, dim) * l / (1 - l * l) * (dim - 1) / math.sqrt(dim)
            self.weight = torch.nn.Parameter(distribution.join_parameters(
                log_probs=torch.zeros(num_classes, 1),
                means=means,
                hidden_ik=distribution._parametrization.ipositive(1 / torch.linalg.norm(means, dim=-1, keepdim=True))
            ))
        self.log_scale = (torch.nn.Parameter(torch.full([], self._config["initial_log_scale"]))
                          if self._config["scale"] == "trainable"
                          else math.log(self._config["scale"]))

    @property
    def has_weight(self):
        return True

    @property
    def has_bias(self):
        return False

    @property
    def has_variance(self):
        return False

    @property
    def kappa_confidence(self):
        """Get lambda parameter of vMF-loss."""
        return self._config["kappa_confidence"]

    def forward(self, parameters, labels=None, scorer=None):
        if (labels is not None) and (labels.shape != parameters.shape[:-1]):
            raise ValueError("Parameters and labels shape mismatch: {}, {}".format(
                parameters.shape, labels.shape))
        dtype = parameters.dtype
        device = parameters.device
        b = len(parameters)
        k = self._config["sample_size"]
        c = self._num_classes
        scale = self._get_scale()
        sample, _ = self._distribution.sample(parameters, list(parameters.shape[:-1]) + [k])  # (B, K, D).
        sample = torch.nn.functional.normalize(sample, dim=-1)
        if (labels is not None) and (not self._config["deterministic_target"]):
            # Train, probabilistic target.
            sample_parameters = self._distribution.join_parameters(
                log_probs=torch.zeros(b, k, 1, dtype=dtype, device=device),
                means=sample.unsqueeze(-2),
                hidden_ik=self._distribution._parametrization.ipositive(torch.ones(b, k, 1, 1, dtype=dtype, device=device) / scale)
            )  # (B, K, P).
            logmls = self._logmls(sample_parameters.reshape(b, k, 1, -1), self.weight.reshape(1, 1, c, -1))  # (B, K, C).
            means = self._distribution.mean(parameters)  # (B, D).
            target_means = self._distribution.mean(self.weight[labels])  # (B, D).
            neg_lognum = scale * (means * target_means).sum(dim=-1)  # (B).
            neg_logden = torch.logsumexp(logmls, dim=2) - self._distribution._vmf_logc(scale)  # (B, K).
            losses = neg_logden.mean(1) - neg_lognum  # (B).
            logits = torch.empty(b, c, dtype=dtype, device=device)
            logits.scatter_(1, labels.reshape(b, 1), -losses.reshape(b, 1))
        elif labels is not None:
            # Train, deterministic target.
            assert self._config["deterministic_target"]
            nweight = torch.nn.functional.normalize(self.weight, dim=-1)  # (C, D).
            means = self._distribution.mean(parameters)  # (B, D).
            neg_lognum = scale * (means * nweight[labels]).sum(dim=-1)  # (B).
            products = scale * (nweight[None, None, :, :] * sample[:, :, None, :]).sum(-1)  # (B, K, C).
            neg_logden = torch.logsumexp(products, dim=2)  # (B, K).
            losses = neg_logden.mean(1) - neg_lognum  # (B).
            logits = torch.empty(b, c, dtype=dtype, device=device)
            logits.scatter_(1, labels.reshape(b, 1), -losses.reshape(b, 1))
        else:
            # Test.
            if self._config["deterministic_target"]:
                target_sample = self.weight[:, None, :]  # (C, 1, D).
            else:
                target_sample, _ = self._distribution.sample(self.weight, [self._num_classes, k])  # (C, K, D).
            tk = target_sample.shape[1]
            target_sample = torch.nn.functional.normalize(target_sample, dim=-1)  # (C, TK, D).
            cosines = torch.nn.functional.linear(sample.reshape(b * k, -1), target_sample.reshape(c * tk, -1))  # (B * K, C * TK).
            cosines = cosines.reshape(b, k, c, tk).permute(0, 2, 1, 3).reshape(b, c, k * tk)  # (B, C, KxTK).
            scores = scale * cosines  # (B, C, KxTK).
            probs = torch.nn.functional.softmax(scores, dim=1).mean(-1)  # (B, C).
            logits = probs.log()
        return logits

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        result = {
            "scale": self._get_scale()
        }
        if not self._config["deterministic_target"]:
            target_hidden_ik = self._distribution.split_parameters(self.weight)[2].squeeze(-1)
            target_sqrt_ik = self._distribution._parametrization.positive(target_hidden_ik).sqrt()
            result["target_sqrt_inv_k/mean"] = target_sqrt_ik.mean()
            result["target_sqrt_inv_k/std"] = target_sqrt_ik.std()
        return result

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )

    def _get_scale(self):
        return self.log_scale.exp() if self._config["scale"] == "trainable" else math.exp(self.log_scale)

    def _vmf_logc(self, k):
        dim = self._distribution.dim
        nm14 = (dim - 1) / 4
        nm12 = (dim - 1) / 2
        np12 = (dim + 1) / 2
        nm12sq = nm12 ** 2
        np12sq = np12 ** 2
        ksq = k ** 2
        sqrtm = (nm12sq + ksq).sqrt()
        sqrtp = (np12sq + ksq).sqrt()
        return (nm14 * ((nm12 + sqrtm).log() + (nm12 + sqrtp).log())
                - 0.5 * (sqrtm + sqrtp))

    def _vmf_logmls(self, means1, hidden_ik1, means2, hidden_ik2):
        """Compute Log MLS for unimodal distributions."""
        k1 = 1 / self._distribution._parametrization.positive(hidden_ik1)
        k2 = 1 / self._distribution._parametrization.positive(hidden_ik2)
        k = torch.linalg.norm(k1 * means1 + k2 * means2, dim=-1, keepdim=True)  # (..., 1).
        logc1 = self._vmf_logc(k1)
        logc2 = self._vmf_logc(k2)
        logc = self._vmf_logc(k)
        return (logc1 + logc2 - logc).squeeze(-1)

    def _logmls(self, parameters1, parameters2):
        if not self._config["approximate_logc"]:
            return self._distribution.logmls(parameters1, parameters2)
        log_probs1, means1, hidden_ik1 = self._distribution.split_parameters(parameters1)  # (..., C), (..., C, D), (..., C, 1).
        log_probs2, means2, hidden_ik2 = self._distribution.split_parameters(parameters2)  # (..., C), (..., C, D), (..., C, 1).
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


class SPEClassifier(torch.nn.Module):
    """Extracts target centroids from elements of the same batch and computes Stochastic Prototype Embeddings logits.

    See "Stochastic Prototype Embeddings." (2019) for details.

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: Positive labels used for margin with shape (...).
        - scorer: Unused.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    LOG_EPS = -100.0

    @staticmethod
    def get_default_config(train_epsilon=True, sample_size=16):
        """Get classifier config.

        Args:
            train_epsilon: Whether to use trainable addition to the variance or not.
            sample_size: Number of samples used for integral evaluation. Zero to disable sampling and use distribution mean.
        """
        return OrderedDict([
            ("train_epsilon", train_epsilon),
            ("sample_size", sample_size)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        if not isinstance(distribution, NormalDistribution):
            raise ValueError("Expected GMM distribution for SPE loss.")
        super().__init__()
        self._config = prepare_config(self, config)
        self._distribution = distribution
        self._num_classes = num_classes
        if self._config["train_epsilon"]:
            self.hidden_epsilon = torch.nn.Parameter(torch.full([], 0.01 ** (2 / distribution.dim), dtype=torch.float))

    @property
    def has_weight(self):
        return False

    @property
    def has_bias(self):
        return False

    @property
    def has_variance(self):
        return self._config["train_epsilon"]

    @property
    def variance(self):
        if self._config["train_epsilon"]:
            return self.hidden_epsilon.exp()
        else:
            return 0

    def forward(self, parameters, labels=None, scorer=None):
        if labels is None:
            # Inference.
            return torch.zeros(*(list(parameters.shape[:-1]) + [self._num_classes]),
                               dtype=parameters.dtype, device=parameters.device)
        if parameters.ndim != 2:
            raise NotImplementedError("Expected embeddings with shape (B, N), got: {}".format(parameters.shape))
        if labels.shape != parameters.shape[:-1]:
            raise ValueError("Parameters and labels shape mismatch: {}, {}".format(
                parameters.shape, labels.shape))
        by_class, order, label_map = self._group_by_class(parameters, labels)  # (B, L', P), (B, L'), (L').
        k = len(by_class) // 2
        logits1 = self._compute_logits(by_class[:k], by_class[k:])  # (B1, L', L').
        logits2 = self._compute_logits(by_class[k:], by_class[:k])  # (B2, L', L').
        logits = torch.cat([logits1, logits2], dim=0)  # (B, L', L').
        all_logits = torch.full([logits.shape[0], logits.shape[1], self._num_classes], self.LOG_EPS,
                                device=logits.device, dtype=logits.dtype)  # (B, L', L).
        indices = label_map[None, None].tile(logits.shape[0], logits.shape[1], 1)  # (B, L', L').
        all_logits.scatter_(2, indices, logits)  # (B, L', L).
        all_logits = all_logits.reshape(len(labels), self._num_classes)  # (B, L).
        all_logits = all_logits.take_along_dim(torch.argsort(order.flatten()).reshape(-1, 1), 0)  # (B, L).
        return all_logits

    @staticmethod
    def _group_by_class(embeddings, labels):
        """Group embeddings into batch by label.

        Returns:
           A tuple of
               - grouped_embeddings with shape (B // L, L, P), where second dimension encodes label.
               - label_map with shape (L) which stores original label indices.
        """
        if embeddings.ndim != 2:
            raise ValueError("Expected tensor with shape (B, P).")
        counts = torch.bincount(labels)
        counts = counts[counts > 0]
        if (counts != counts[0]).any():
            raise RuntimeError("Need uniform balanced sampling: {}.".format(counts))
        unique_labels = torch.unique(labels)
        indices = torch.stack([torch.nonzero(labels == label).squeeze(-1) for label in unique_labels], dim=1)  # (B // L, L).
        by_class = torch.stack([embeddings[labels == label] for label in unique_labels], dim=1)  # (B // L, L, P).
        assert by_class.ndim == 3
        return by_class, indices, unique_labels

    def _compute_prototypes(self, embeddings):
        if embeddings.ndim != 3:
            raise ValueError("Expected grouped embeddings with shape (B, L, P).")
        logprobs, mean, hidden_var = self._distribution.split_parameters(embeddings)  # (B, L, 1), (B, L, 1, D), (B, L, 1, D).
        var = self.variance + self._distribution._parametrization.positive(hidden_var)  # (B, L, 1, D).
        new_var = 1 / (1 / var).sum(0)  # (L, 1, D).
        new_mean = new_var * (mean / var).sum(0)  # (L, 1, D).
        new_hidden_var = self._distribution._parametrization.ipositive(self.variance + new_var)
        prototypes = self._distribution.join_parameters(
            logprobs[0],
            new_mean,
            new_hidden_var
        )  # (L, P).
        return prototypes

    def _compute_logits(self, query, support):
        """Compute SPE logits.

        Args:
            - query: Queries with shape (B, L, P) to compute logits for.
            - support: Embeddings used for prototype computation with shape (B', L, P).
        Returns:
            SPE logits with shape (B, L).
        """
        prototypes = self._compute_prototypes(support)  # (L, P).
        prod_distribution, prod_parameters = self._distribution.pdf_product(query[:, :, None, :], prototypes[None, None])  # (B, L, L, P).
        if self._config["sample_size"] > 0:
            b, l, _ = query.shape
            s = self._config["sample_size"]
            sample, _ = prod_distribution.sample(prod_parameters[:, :, None, :, :], [b, l, s, l])  # (B, L, S, L, D).
        else:
            s = 1
            sample = prod_distribution.mean(prod_parameters).unsqueeze(-3)  # (B, L, S, L, D).
        logmls = self._distribution.logmls(query[:, :, None, :], prototypes[None, None])  # (B, L, L).
        target_logpdfs = self._distribution.logpdf(prototypes[None, None, None], sample)  # (B, L, S, L).
        logdenum = torch.logsumexp(target_logpdfs, dim=-1, keepdim=True)  # (B, L, S, 1).
        logits = logmls + torch.logsumexp(-logdenum, dim=-2) - math.log(s)  # (B, L, L).
        return logits

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        return {}

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )


class ScorerClassifier(torch.nn.Linear):
    """Classify using scores.

    Args:
        distribution: Distribution used in the model.
        num_classes: Number of output classes.
        priors (unused): Precomputed class priors. Priors can be learned on-line if not provided.

    Inputs:
        - parameters: Distribution parameters with shape (..., K).
        - labels: Unused.
        - scorer: Scorer used for logits computation.

    Outputs:
        - logits: Class logits with shape (..., C).

    """

    @staticmethod
    def get_default_config(use_bias=True):
        """Get classifier config."""
        return OrderedDict([
            ("use_bias", use_bias)
        ])

    def __init__(self, distribution, num_classes, *, priors=None, config=None):
        config = prepare_config(self, config)
        super().__init__(distribution.num_parameters, num_classes, bias=config["use_bias"])
        self._config = config
        self._distribution = distribution
        self._num_classes = num_classes

    @property
    def has_weight(self):
        return True

    @property
    def has_bias(self):
        return self._config["use_bias"]

    @property
    def has_variance(self):
        return False

    def forward(self, parameters, labels=None, scorer=None):
        prefix = tuple(parameters.shape[:-1])
        target_distributions = self.weight.reshape(*([1] * len(prefix) + list(self.weight.shape)))  # (..., C, P).
        logits = scorer(parameters.unsqueeze(-2), target_distributions)  # (..., C).
        return logits

    def statistics(self):
        """Compute useful statistics for logging.

        Returns:
            Dictionary with floating-point statistics values.
        """
        parameters = self._distribution.unpack_parameters(self.weight)
        if "covariance" in parameters:
            key = "std"
            value = parameters["covariance"].detach()
        elif "k" in parameters:
            key = "vmf_sqrt_inv_k"
            value = 1 / parameters["k"].detach().sqrt()
        else:
            return {}
        return {
            "target_{}/mean".format(key): value.mean(),
            "target_{}/std".format(key): value.std()
        }

    def extra_repr(self):
        return "distribution={}, num_classes={}, config={}".format(
            self._distribution, self._num_classes, self._config
        )
