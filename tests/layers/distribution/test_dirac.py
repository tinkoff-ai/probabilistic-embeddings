import itertools
import math
from unittest import TestCase, main

import numpy as np
import torch

from probabilistic_embeddings.layers.distribution import DiracDistribution


class TestDiracDistribution(TestCase):
    def test_sampling(self):
        """Test MLS is equal to estimation by sampling."""
        distribution = DiracDistribution(config={"dim": 2})
        parameters = torch.randn((1, 1, 2))
        with torch.no_grad():
            means = distribution.mean(parameters)
            sample, _ = distribution.sample(parameters, [50, 10])
            delta = (sample - means).abs().max()
        self.assertAlmostEqual(delta, 0)


if __name__ == "__main__":
    main()
