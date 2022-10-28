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
from probabilistic_embeddings.runner import Runner


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
    "dataset_params": {
        "name": "debug-openset",
        "batch_size": 4,
        "num_workers": 0,
        "num_validation_folds": 2
    },
    "model_params": {
        "distribution_type": "vmf",
        "distribution_params": {
            "k": "separate"
        },
        "embedder_params": {
            "pretrained": False,
            "model_type": "resnet18",
            "extra_head_dim": 1
        },
        "classifier_type": "loglike"
    },
    "trainer_params": {
        "num_epochs": 2
    },
    "num_evaluation_seeds": 2,
    "stages": [
        {
            "model_params": {
                "embedder_params": {
                    "freeze_extra_head": True
                }
            }
        },
        {
            "resume_prefixes": "_embedder.,_classifier.",
            "model_params": {
                "freeze_classifier": True,
                "embedder_params": {
                    "freeze_stem": True,
                    "freeze_head": True,
                    "freeze_normalizer": True
                }
            }
        },
    ]
}


class TestStages(TestCase):
    def test_train(self):
        with tempfile.TemporaryDirectory() as root:
            config = CONFIG.copy()
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

            runner = Runner(root, root, config=config)
            runner.evaluate()
            checkpoint0 = runner.model["model"].state_dict()
            checkpoint1 = self._load_checkpoint(os.path.join(root, "checkpoints", "train-0."))["model_model_state_dict"]
            checkpoint2 = self._load_checkpoint(os.path.join(root, "checkpoints", "train-1."))["model_model_state_dict"]
            self.assertTrue(self._is_equal(checkpoint0, checkpoint1, prefix="_embedder._extra_head."))
            self.assertFalse(self._is_equal(checkpoint0, checkpoint1, prefix="_embedder._stem."))
            self.assertFalse(self._is_equal(checkpoint0, checkpoint1, prefix="_embedder._head."))
            self.assertFalse(self._is_equal(checkpoint0, checkpoint1, prefix="_embedder._normalizer."))
            self.assertFalse(self._is_equal(checkpoint0, checkpoint1, prefix="_classifier."))
            self.assertFalse(self._is_equal(checkpoint1, checkpoint2, prefix="_embedder._extra_head."))
            self.assertTrue(self._is_equal(checkpoint1, checkpoint2, prefix="_embedder._stem."))
            self.assertTrue(self._is_equal(checkpoint1, checkpoint2, prefix="_embedder._head."))
            self.assertTrue(self._is_equal(checkpoint1, checkpoint2, prefix="_classifier."))
            self.assertTrue(self._is_equal(checkpoint1, checkpoint2, prefix="_embedder._normalizer."))

    def _is_equal(self, state_dict1, state_dict2, prefix=None):
        if prefix is not None:
            state_dict1 = {k: v for k, v in state_dict1.items() if k.startswith(prefix)}
            state_dict2 = {k: v for k, v in state_dict2.items() if k.startswith(prefix)}
            assert state_dict1 and state_dict2
        is_equal = True
        if set(state_dict1) != set(state_dict2):
            raise ValueError("Keys mismatch")
        for k in state_dict1:
            if not np.allclose(state_dict1[k].cpu().numpy(), state_dict2[k].cpu().numpy()):
                is_equal = False
                break
        return is_equal

    def _load_checkpoint(self, prefix):
        checkpoint = None
        for i in range(CONFIG["trainer_params"]["num_epochs"] + 1):
            path = prefix + str(i) + ".pth"
            if not os.path.exists(path):
                continue
            return torch.load(path, map_location="cpu")
        raise FileNotFoundError("No checkpoint for prefix {}.".format(prefix))


if __name__ == "__main__":
    main()
