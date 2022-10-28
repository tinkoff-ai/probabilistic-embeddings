import itertools
import math
from unittest import TestCase, main

import numpy as np
import torch

from probabilistic_embeddings.layers.parametrization import Parametrization


class TestParametrization(TestCase):
    def test_log_positive(self):
        for type in ["exp", "invlin", "abs"]:
            for min in [0, 0.1, 1, 10]:
                for kwargs in [{"scale": 1, "center": 0}, {"scale": 0.3, "center": 5.4}]:
                    p = Parametrization(type=type, min=min, **kwargs)
                    xs = torch.linspace(-10, 10, 1001)
                    with torch.no_grad():
                        ys_gt = p.positive(xs).log()
                        ys = p.log_positive(xs)
                    self.assertTrue(np.allclose(ys.numpy(), ys_gt.numpy(), atol=1e-6))
        for max in [0, 0.1, 1, 10]:
            for min in [0, 0.1, 1, 10]:
                if min >= max:
                    continue
                p = Parametrization(type="sigmoid", min=min, max=max)
                xs = torch.linspace(-10, 10, 1001)
                with torch.no_grad():
                    ys_gt = p.positive(xs).log()
                    ys = p.log_positive(xs)
                self.assertTrue(np.allclose(ys.numpy(), ys_gt.numpy(), atol=1e-6))

    def test_ipositive(self):
        for type in ["exp", "invlin"]:
            for min in [0, 0.1, 1, 10]:
                for kwargs in [{"scale": 1, "center": 0}, {"scale": 0.3, "center": 0.9}]:
                    p = Parametrization(type=type, min=min, **kwargs)
                    xs_gt = torch.linspace(-5, 5, 1001).double()
                    with torch.no_grad():
                        ys = p.positive(xs_gt)
                        xs = p.ipositive(ys)
                    self.assertTrue((ys > 0).all())
                    self.assertTrue(np.allclose(xs.numpy(), xs_gt.numpy(), atol=1e-6))
        for max in [0, 0.1, 1, 10]:
            for min in [0, 0.1, 1, 10]:
                if min >= max:
                    continue
                p = Parametrization(type="sigmoid", min=min, max=max)
                xs_gt = torch.linspace(-10, 10, 1001).double()
                with torch.no_grad():
                    ys = p.positive(xs_gt)
                    xs = p.ipositive(ys)
                self.assertTrue((ys > 0).all())
                self.assertTrue(np.allclose(xs.numpy(), xs_gt.numpy(), atol=1e-6))
        for min in [0, 0.1, 1, 10]:
            p = Parametrization(type="abs", min=min)
            xs_gt = torch.linspace(0, 5, 1001).double()
            with torch.no_grad():
                ys = p.positive(xs_gt)
                xs = p.ipositive(ys)
            self.assertTrue((ys >= 0).all())
            self.assertTrue(np.allclose(xs.numpy(), xs_gt.numpy(), atol=1e-6))


if __name__ == "__main__":
    main()
