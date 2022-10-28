import torch
from unittest import TestCase, main

from probabilistic_embeddings.metrics.classification import QualityMetric


class TestSpearman(TestCase):
    def test_simple(self):
        m = QualityMetric()
        m.update(torch.tensor([106, 100, 86, 101]),
                 torch.tensor([7, 27, 2, 50, 28]))
        m.update(torch.tensor([99, 103, 97, 113, 112, 110]),
                 torch.tensor([29, 20, 12, 6, 17]))
        metrics = m.compute_key_value()
        self.assertAlmostEqual(metrics["quality_scc"], -0.17575757575757575)


if __name__ == "__main__":
    main()
