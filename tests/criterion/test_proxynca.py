#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from probabilistic_embeddings.criterion.proxynca import ProxyNCALoss
from probabilistic_embeddings.layers import DiracDistribution, NegativeL2Scorer


class TestProxyNCALoss(TestCase):
    def test_1d(self):
        distribution = DiracDistribution(config={"dim": 1})
        scorer = NegativeL2Scorer(distribution)
        proxy_nca_loss = ProxyNCALoss(aggregation="none")
        embeddings = torch.tensor([-1, 0, 2]).float().reshape(-1, 1)
        targets = torch.tensor([1, -0.5]).float().reshape(-1, 1)
        labels = torch.tensor([0, 1, 0])
        losses = proxy_nca_loss(embeddings, labels, targets, scorer).numpy().tolist()
        losses_gt = [3.75, -0.75, -5.25]
        self.assertEqual(len(losses), 3)
        for loss, loss_gt in zip(losses, losses_gt):
            self.assertAlmostEqual(loss, loss_gt)


if __name__ == "__main__":
    main()
