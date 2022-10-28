import math
from unittest import TestCase, main

import numpy as np
import torch

from probabilistic_embeddings.layers.distribution import NormalDistribution, VMFDistribution, DiracDistribution
from probabilistic_embeddings.layers.classifier import LogLikeClassifier, VMFClassifier, SPEClassifier


class TestLogLikeClassifier(TestCase):
    def test_margin(self):
        """Check margin in simple case."""
        distribution = VMFDistribution()
        classifier = LogLikeClassifier(distribution, 30)
        classifier_margin = LogLikeClassifier(distribution, 30, config={"margin": 3})
        classifier_margin.load_state_dict(classifier.state_dict())
        parameters = torch.randn(2, 1, distribution.num_parameters)
        points = torch.randn(2, 1, distribution.dim)
        labels = torch.tensor([[5], [1]]).long()
        delta_gt = np.zeros((2, 1, 30))
        delta_gt[0, 0, 5] = 3
        delta_gt[1, 0, 1] = 3
        with torch.no_grad():
            logits = classifier(parameters, labels)
            logits_margin = classifier_margin(parameters, labels)
            delta = (logits - logits_margin).numpy()
        self.assertTrue(np.allclose(delta, delta_gt))


class TestVMFClassifier(TestCase):
    def test_train(self):
        """Compare computation with formula-based."""
        distribution = VMFDistribution(config={"dim": 8})
        b = 3
        nc = 1
        k = 100000
        classifier = VMFClassifier(distribution, nc, config={"sample_size": k, "approximate_logc": False})
        with torch.no_grad():
            classifier.log_scale.copy_(torch.zeros([]) + 5)  # Larger than zero to ensure large loss variance.
        parameters = torch.randn(b, distribution.num_parameters)  # (B, P).
        labels = torch.arange(b) % nc
        logits = classifier(parameters, labels)  # (B, C).
        losses = -logits.gather(1, labels[:, None]).squeeze(1)  # NLL, shape (B).

        _, target_means, target_hidden_ik = distribution.split_parameters(classifier.weight)
        target_k = 1 / distribution._parametrization.positive(target_hidden_ik)  # (C, 1, 1).
        sample, _ = distribution.sample(parameters, (b, k))  # (B, K, D).
        b, _, d = sample.shape
        weighted = classifier.log_scale.exp() * sample.reshape(b, k, 1, d) + (target_means * target_k).reshape(1, 1, nc, d)  # (B, K, C, D).
        norms = torch.linalg.norm(weighted, dim=-1)  # (B, K, C).
        log_fractions = distribution._vmf_logc(target_k).reshape(1, 1, nc) - distribution._vmf_logc(norms)  # (B, K, C).
        expectation = log_fractions.logsumexp(dim=2).mean(1)  # (B).
        targets = classifier.weight[labels]  # (B, D).
        expected_target = distribution.mean(targets)  # (B, D).
        expected_sample = distribution.mean(parameters)  # (B, D).
        offset = -classifier.log_scale.exp() * (expected_target * expected_sample).sum(1)  # (B).
        losses_gt = expectation + offset
        self.assertTrue(torch.allclose(losses, losses_gt, rtol=1e-3))

    def test_infer(self):
        """Test probability estimation with high concentration."""
        dim = 8
        distribution = VMFDistribution(config={"dim": dim, "max_logk": None})
        b = 3
        nc = 5
        k = 1000

        classifier = VMFClassifier(distribution, nc, config={"sample_size": k})
        cls_means = torch.nn.functional.normalize(torch.randn(nc, 1, dim), dim=-1)
        with torch.no_grad():
            classifier.weight.copy_(distribution.join_parameters(
                log_probs=torch.zeros(nc, 1),
                means=cls_means,
                hidden_ik=distribution._parametrization.ipositive(torch.full((nc, 1, 1), 1e-10))
            ))
            classifier.log_scale.copy_(torch.zeros([]) + 0)

        means = torch.nn.functional.normalize(torch.randn(b, 1, dim), dim=-1)
        parameters = distribution.join_parameters(
            log_probs=torch.zeros(b, 1),
            means=means,
            hidden_ik=distribution._parametrization.ipositive(torch.full((b, 1, 1), 1e-10))
        )
        probs = classifier(parameters).exp()  # (B, C).
        logits_gt = classifier.log_scale.exp() * (cls_means[None, :, 0] * means).sum(-1)
        probs_gt = torch.nn.functional.softmax(logits_gt, dim=1)  # (B, C).
        self.assertTrue(torch.allclose(probs, probs_gt, rtol=1e-3))


class TestSPEClassifier(TestCase):
    def test_utils(self):
        distribution = NormalDistribution(config={"dim": 2, "max_logivar": None, "parametrization": "exp"})
        classifier = SPEClassifier(distribution, 3, config={"train_epsilon": False, "sample_size": 0})
        v = 1
        logv = math.log(v)
        embeddings = torch.tensor([
            [0, 0, logv],
            [1, 0, logv],
            [0, 0, logv],
            [0, 1, logv]
        ]).float()  # (4, 2).
        labels = torch.tensor([0, 0, 2, 2])

        # Test grouping.
        by_class, indices, label_map = classifier._group_by_class(embeddings, labels)
        by_class_gt = torch.tensor([
            [0, 0, logv],
            [0, 0, logv],
            [1, 0, logv],
            [0, 1, logv]
        ]).float().reshape(2, 2, 3)
        self.assertTrue(by_class.allclose(by_class_gt))
        label_map_gt = torch.tensor([0, 2])
        self.assertTrue((label_map == label_map_gt).all())
        indices_gt = torch.tensor([[0, 2], [1, 3]])
        self.assertTrue((indices == indices_gt).all())

        # Test prototypes.
        self.assertTrue(classifier._compute_prototypes(embeddings[None]).allclose(embeddings))
        prototypes = classifier._compute_prototypes(by_class)
        prototypes_gt = torch.tensor([
            [0.5, 0, logv - math.log(2)],
            [0, 0.5, logv - math.log(2)]
        ]).float()
        self.assertTrue(prototypes.allclose(prototypes_gt))

    def test_logits(self):
        distribution = NormalDistribution(config={"dim": 1, "max_logivar": None, "parametrization": "exp"})
        classifier = SPEClassifier(distribution, 3, config={"train_epsilon": False, "sample_size": 0})
        v = 1
        logv = math.log(v)
        embeddings = torch.tensor([
            [0, logv],
            [0, logv],
            [1, logv],
            [1, logv]
        ]).float()  # (4, 2).
        labels = torch.tensor([2, 2, 0, 0])

        lp0 = -0.5 * math.log(2 * math.pi)
        lp05 = lp0 - 0.5 * 0.25
        lp1 = lp0 - 0.5
        lpsum = math.log(math.exp(lp0) + math.exp(lp05))
        mls0 = lp0 - 0.5 * math.log(2)
        mls1 = mls0 - 0.25
        logit0 = mls0 - lpsum
        logit1 = mls1 - lpsum
        logits_gt = torch.tensor([
            [logit1, classifier.LOG_EPS, logit0],
            [logit1, classifier.LOG_EPS, logit0],
            [logit0, classifier.LOG_EPS, logit1],
            [logit0, classifier.LOG_EPS, logit1]
        ])
        logits = classifier(embeddings, labels)
        self.assertTrue(logits.allclose(logits_gt))


if __name__ == "__main__":
    main()
