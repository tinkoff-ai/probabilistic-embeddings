import itertools
import math
from unittest import TestCase, main

import numpy as np
import torch

from probabilistic_embeddings.layers.distribution.common import auto_matmul


class TestCommon(TestCase):
    def test_auto_matmul(self):
        """Check auto_matmul result equal to torch.matmul."""
        def _check_case(shape1, shape2):
            with torch.no_grad():
                m1 = torch.randn(*shape1)
                m2 = torch.randn(*shape2)
                gt = torch.matmul(m1, m2).numpy()
                result = auto_matmul(m1, m2).numpy()
            self.assertTrue(np.allclose(result, gt, atol=1e-6))
        # Zero dimension.
        _check_case([0, 1], [1, 2])
        # torch.matmul mode.
        _check_case([5, 1, 4, 1, 7], [2, 4, 7, 2])
        # Fast torch.nn.functional linear mode.
        _check_case([3, 2, 1, 4, 5], [1, 6, 5, 1])


if __name__ == "__main__":
    main()
