#!/usr/bin/env python3
import os
import sys
import tempfile
from collections import OrderedDict
from unittest import TestCase, main

import numpy as np
import torch
import yaml
from torchvision import transforms

from probabilistic_embeddings import commands


class Namespace:
    ARGS = ["cmd", "data", "name", "logger", "config", "train_root", "checkpoint", "no_strict_init",
            "from_stage", "from_seed"]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, key):
        if key not in self.ARGS:
            raise AttributeError(key)
        return self.__dict__.get(key, None)


CONFIG = {
    "fp16": False,
    "dataset_params": {
        "name": "debug-openset",
        "batch_size": 4,
        "num_workers": 0,
        "num_valid_workers": 0,
        "num_validation_folds": 2
    },
    "model_params": {
        "embedder_params": {
            "pretrained": False,
            "model_type": "resnet18"
        }
    },
    "trainer_params": {
        "optimizer_params": {"lr": 1e-4},
        "num_epochs": 2
    },
    "num_evaluation_seeds": 2
}


class TestReproducibility(TestCase):
    def test_train(self):
        results = [self._run(seed=0) for _ in range(2)]
        for r in results[1:]:
            self.assertEqual(self._num_not_equal(r, results[0]), 0)
        results2 = [self._run(seed=1) for _ in range(2)]
        for r in results2[1:]:
            self.assertEqual(self._num_not_equal(r, results2[0]), 0)
        self.assertGreater(self._num_not_equal(results2[0], results[0]), 0)

    def _run(self, seed=None):
        with tempfile.TemporaryDirectory() as root:
            config = CONFIG.copy()
            if seed is not None:
                config["seed"] = seed
            config_path = os.path.join(root, "config.yaml")
            with open(config_path, "w") as fp:
                yaml.safe_dump(config, fp)
            args = Namespace(
                cmd="train",
                data=root,  # Unused.
                config=config_path,
                logger="tensorboard",
                train_root=root
            )
            commands.train(args)

            args = Namespace(
                cmd="test",
                checkpoint=os.path.join(root, "checkpoints", "best.pth"),
                data=root,  # Unused.
                config=config_path,
                logger="tensorboard"
            )
            metrics = commands.test(args)
            metrics["checkpoint_hash"] = np.mean([v.sum().item() for v in torch.load(args.checkpoint)["model_model_state_dict"].values()])
        return metrics

    def _num_not_equal(self, d1, d2):
        self.assertEqual(set(d1), set(d2))
        n = 0
        for k in d1:
            v1 = d1[k]
            v2 = d2[k]
            if isinstance(v1, (dict, OrderedDict)):
                n += self._num_not_equal(v1, v2)
            else:
                n += abs(v1 - v2) > 1e-6
        return n


if __name__ == "__main__":
    commands.setup()
    main()
