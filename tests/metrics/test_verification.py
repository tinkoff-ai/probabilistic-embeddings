#!/usr/bin/env python3
from unittest import TestCase, main

import numpy as np
import torch

from probabilistic_embeddings.metrics import VerificationMetrics


class TestVerificationMetrics(TestCase):
    def test_simple(self):
        """Test metrics in simple cases."""
        fpr = 1 / 3
        labels = [0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
        # Correct   =   T    T    T    F    T    T    F    T    T    T.
        confidences = [0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.9, 0.5, 0.5, 1.0]
        scores = np.arange(len(labels))
        permutation = np.arange(len(labels)) # np.random.permutation(len(labels))
        labels = np.array(labels)[permutation]
        confidences = np.array(confidences)[permutation]
        scores = scores[permutation]
        evaluator = VerificationMetrics(config={"fpr": fpr})
        evaluator.update(torch.from_numpy(scores), torch.from_numpy(labels), torch.from_numpy(confidences))
        pr, max_accuracy, auc, tpr, fpr, eer, confidence_auroc, confidence_aupr, confidence_aurcc = evaluator.compute()
        self.assertAlmostEqual(pr, 6 / 10)
        self.assertAlmostEqual(max_accuracy, 0.8)
        # No easy way to compute AUC.
        self.assertAlmostEqual(fpr, 1 / 4)
        self.assertAlmostEqual(tpr, 5 / 6)
        self.assertAlmostEqual(eer, 1 - 0.5 * (5 / 6 + 3 / 4))
        self.assertAlmostEqual(confidence_auroc, 1 / 8 / 2 + 1 / 2)
        self.assertAlmostEqual(confidence_aupr, 1 / 8 + 7 / 8 * (1 / 2 + 8 / 9) / 2)
        self.assertAlmostEqual(confidence_aurcc, 0.1 * (1 / 2 + 1 / 3 + 1 / 4 + 1 / 5 + 1 / 6 + 1 / 7 + 1 / 8 + 1 / 9 + 2 / 10 / 2))


if __name__ == "__main__":
    main()
