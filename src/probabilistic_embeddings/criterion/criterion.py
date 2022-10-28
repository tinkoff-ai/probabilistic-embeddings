from collections import OrderedDict

import torch
from catalyst import dl
from catalyst.utils.misc import get_attr

from ..config import prepare_config
from ..torch import get_base_module, disable_amp
from .multisim import MultiSimilarityLoss
from .proxynca import ProxyNCALoss


class Criterion(torch.nn.Module):
    """Combination of crossentropy and KL-divergence regularization.

    PFE loss is described in Probabilistic Face Embeddings:
      https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf

    HIB loss is described in Modeling Uncertainty with Hedged Instance Embedding:
      https://arxiv.org/pdf/1810.00319.pdf
    """

    @staticmethod
    def get_default_config(use_softmax=True, xent_weight=1.0, xent_smoothing=0.0,
                           hinge_weight=0.0, hinge_margin=1.0,
                           proxy_archor_weight=0.0, proxy_nca_weight=0.0,
                           multi_similarity_weight=0.0, multi_similarity_params=None,
                           prior_kld_weight=0.0, pfe_weight=0.0, pfe_match_self=True, hib_weight=0.0):
        """Get optimizer parameters."""
        return OrderedDict([
            ("use_softmax", use_softmax),
            ("xent_weight", xent_weight),
            ("xent_smoothing", xent_smoothing),
            ("hinge_weight", hinge_weight),
            ("hinge_margin", hinge_margin),
            ("proxy_anchor_weight", proxy_archor_weight),
            ("proxy_nca_weight", proxy_nca_weight),
            ("multi_similarity_weight", multi_similarity_weight),
            ("multi_similarity_params", multi_similarity_params),
            ("prior_kld_weight", prior_kld_weight),
            ("pfe_weight", pfe_weight),
            ("pfe_match_self", pfe_match_self),
            ("hib_weight", hib_weight),
        ])

    def __init__(self, *, config=None):
        super().__init__()
        self._config = prepare_config(self, config)
        if self._config["multi_similarity_weight"] > 0:
            self._multi_similarity_loss = MultiSimilarityLoss(config=self._config["multi_similarity_params"])
        if self._config["proxy_nca_weight"] > 0:
            self._proxy_nca_loss = ProxyNCALoss()
        self.distribution = None
        self.scorer = None

    def __call__(self, embeddings, labels, logits=None, target_embeddings=None,
                 final_weights=None, final_bias=None, final_variance=None):
        loss = 0
        if self._config["xent_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for Xent loss.")
            loss = loss + self._config["xent_weight"] * self._xent_loss(logits, labels)
        if self._config["hinge_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for Hinge loss.")
            loss = loss + self._config["hinge_weight"] * self._hinge_loss(logits, labels)
        if self._config["proxy_anchor_weight"] != 0:
            if logits is None:
                raise ValueError("Need logits for Proxy-Anchor loss.")
            loss = loss + self._config["proxy_anchor_weight"] * self._proxy_anchor_loss(logits, labels)
        if self._config["proxy_nca_weight"] != 0:
            if self.scorer is None:
                raise ValueError("Need scorer for Proxy-NCA loss.")
            if final_weights is None:
                raise ValueError("Need final weights for Proxy-NCA loss.")
            if final_bias is not None:
                raise ValueError("Final bias is redundant for Proxy-NCA loss.")
            loss = loss + self._config["proxy_nca_weight"] * self._proxy_nca_loss(embeddings, labels, final_weights, self.scorer)
        if self._config["multi_similarity_weight"] > 0:
            if self.scorer is None:
                raise ValueError("Need scorer for Multi-similarity loss.")
            loss = loss + self._config["multi_similarity_weight"] * self._multi_similarity_loss(embeddings, labels, self.scorer)
        if self._config["prior_kld_weight"] != 0:
            loss = loss + self._config["prior_kld_weight"] * self._prior_kld_loss(embeddings)
        if self._config["pfe_weight"] != 0:
            loss = loss + self._config["pfe_weight"] * self._pfe_loss(embeddings, labels)
        if self._config["hib_weight"] != 0:
            loss = loss + self._config["hib_weight"] * self._hib_loss(embeddings, labels)
        return loss

    def _xent_loss(self, logits, labels):
        if self._config["use_softmax"]:
            kwargs = {}
            if self._config["xent_smoothing"] > 0:
                # Old PyTorch (below 1.10) doesn't support label_smoothing.
                kwargs["label_smoothing"] = self._config["xent_smoothing"]
            return torch.nn.functional.cross_entropy(logits, labels, **kwargs)
        else:
            return torch.nn.functional.nll_loss(logits, labels)

    def _hinge_loss(self, logits, labels):
        """Compute Hinge loss.

        Args:
            logits: Logits tensor with shape (*, N).
            labels: Integer labels with shape (*).

        Returns:
            Loss value.
        """
        n = logits.shape[-1]
        gt_logits = logits.take_along_dim(labels.unsqueeze(-1), -1)  # (*, 1).
        alt_mask = labels.unsqueeze(-1) != torch.arange(n, device=logits.device)  # (*, N).
        loss = (self._config["hinge_margin"] - gt_logits + logits).clip(min=0)[alt_mask].mean()
        return loss

    def _proxy_anchor_loss(self, logits, labels):
        """See Proxy Anchor Loss for Deep Metric Learning (2020):
        https://arxiv.org/pdf/2003.13911.pdf
        """
        b, c = logits.shape
        one_hot = torch.zeros_like(logits)  # (B, C).
        one_hot.scatter_(-1, labels.unsqueeze(-1).long(), 1)  # (B, C).
        num_positives = one_hot.sum(0)  # (C).
        ninf = -1.0e10
        positive = (-logits + (1 - one_hot) * ninf)[:, num_positives > 0].logsumexp(0)  # (P).
        positive = torch.nn.functional.softplus(positive).mean()
        negative = (logits + one_hot * ninf)[:, num_positives < b].logsumexp(0)  # (N).
        negative = torch.nn.functional.softplus(negative).mean()
        return positive + negative

    def _prior_kld_loss(self, distributions):
        return self.distribution.prior_kld(distributions).mean()

    def _pfe_loss(self, distributions, labels):
        pair_mls = self.distribution.logmls(distributions[None], distributions[:, None])
        same_mask = labels[None] == labels[:, None]  # (B, B).
        if not self._config["pfe_match_self"]:
            same_mask.fill_diagonal_(False)
        same_mls = pair_mls[same_mask]
        return -same_mls.mean()

    def _hib_loss(self, distributions, labels):
        same_probs = self.scorer(distributions[None], distributions[:, None])  # (B, B).
        same_mask = labels[None] == labels[:, None]  # (B, B).
        positive_probs = same_probs[same_mask]
        negative_probs = same_probs[~same_mask]
        positive_xent = torch.nn.functional.binary_cross_entropy(positive_probs, torch.ones_like(positive_probs))
        negative_xent = torch.nn.functional.binary_cross_entropy(negative_probs, torch.zeros_like(negative_probs))
        return 0.5 * (positive_xent + negative_xent)


class CriterionCallback(dl.CriterionCallback):
    """Compute criterion in FP32 and pass distribution and scorer to criterion."""
    def __init__(self, *args, **kwargs):
        amp = kwargs.pop("amp", False)
        super().__init__(*args, **kwargs)
        self._amp = amp

    def _metric_fn(self, *args, **kwargs):
        with disable_amp(not self._amp):
            return self.criterion(*args, **kwargs)

    def on_stage_start(self, runner: "IRunner"):
        super().on_stage_start(runner)
        model = get_attr(runner, key="model", inner_key="model")
        scorer = get_attr(runner, key="model", inner_key="scorer")
        assert scorer is not None
        self.criterion.scorer = scorer
        distribution = get_base_module(model).distribution
        assert distribution is not None
        self.criterion.distribution = distribution

    def on_stage_end(self, runner: "IRunner"):
        super().on_stage_end(runner)
        self.criterion.scorer = None
        self.criterion.distribution = None
