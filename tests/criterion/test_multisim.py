#!/usr/bin/env python3
import math
from unittest import TestCase, main

import torch

from probabilistic_embeddings.criterion.multisim import MultiSimilarityLoss
from probabilistic_embeddings.layers import DiracDistribution, NegativeL2Scorer


class TestMultiSimilarityLoss(TestCase):
    def test_1d(self):
        distribution = DiracDistribution(config={"dim": 1})
        scorer = NegativeL2Scorer(distribution)
        multi_similarity_loss = MultiSimilarityLoss(aggregation="none")
        embeddings = torch.tensor([-1, 0, 2]).float().reshape(-1, 1)
        labels = torch.tensor([0, 1, 0])
        losses = multi_similarity_loss(embeddings, labels, scorer).numpy().tolist()
        losses_gt = [1 / 2 * math.log(1 + math.exp(-2 * -9.5)) + 1 / 40 * math.log(1 + math.exp(40 * -1.5)),
                     0,
                     1 / 2 * math.log(1 + math.exp(-2 * -9.5)) + 1 / 40 * math.log(1 + math.exp(40 * -4.5))]
        self.assertEqual(len(losses), 3)
        for loss, loss_gt in zip(losses, losses_gt):
            self.assertAlmostEqual(loss, loss_gt)


if __name__ == "__main__":
    main()
