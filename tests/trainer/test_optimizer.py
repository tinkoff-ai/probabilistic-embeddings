import os
import tempfile
from collections import OrderedDict
from unittest import TestCase, main

from probabilistic_embeddings.trainer.optimizer import *


class TestOptimizer(TestCase):
    def test_sam_split(self):
        #parameters = [torch.randn([]), torch.randn(5, 3), torch.randn(5)]
        parameters = [torch.full([], 2.), torch.full([5, 3], 2.), torch.full([5], 2.)]
        groups = SamOptimizer._split_bias_and_bn_groups(parameters, {"adaptive": False})
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(len(groups[1]), 2)
        self.assertEqual(groups[0]["params"][0].ndim, 2)
        self.assertEqual(groups[1]["params"][0].ndim, 0)
        self.assertEqual(groups[1]["params"][1].ndim, 1)
        self.assertEqual(groups[1]["adaptive"], False)

        def closure(optimizer):
            for group in optimizer.param_groups:
                for p in group["params"]:
                    p.grad = torch.ones_like(p)

        optimizer = SamOptimizer([p.clone() for p in parameters], config={"adaptive_bias_and_bn": True})
        closure(optimizer)
        optimizer.first_step()
        self.assertFalse(optimizer.param_groups[0]["params"][0].allclose(parameters[0]))
        self.assertFalse(optimizer.param_groups[0]["params"][1].allclose(parameters[1]))
        self.assertFalse(optimizer.param_groups[0]["params"][2].allclose(parameters[2]))
        gt_update = optimizer.param_groups[0]["params"]

        optimizer = SamOptimizer([p.clone() for p in parameters], config={"adaptive_bias_and_bn": False})
        closure(optimizer)
        optimizer.first_step()
        # Scale lower, power greater.
        self.assertTrue((optimizer.param_groups[0]["params"][0] > gt_update[1]).all())
        self.assertTrue((optimizer.param_groups[1]["params"][0] < gt_update[0]).all())
        self.assertTrue((optimizer.param_groups[1]["params"][1] < gt_update[2]).all())


if __name__ == "__main__":
    main()
