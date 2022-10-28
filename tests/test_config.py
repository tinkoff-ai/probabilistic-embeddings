import os
import tempfile
from collections import OrderedDict
from unittest import TestCase, main

from probabilistic_embeddings.config import *


class SimpleModel(object):
    @staticmethod
    def get_default_config(model=None, model_config=None):
        return OrderedDict([
            ("model", model),
            ("model_config", model_config)
        ])

    def __init__(self, config=None):
        self.config = prepare_config(self, config)


class TestConfig(TestCase):
    def test_parser(self):
        config_orig = {
            "model": "some-model",
            "model_config": {"_type": "SimpleModel", "arg1": 5, "arg2": None}
        }
        config_gt = {
            "model": "some-model",
            "model_config": {"_type": "SimpleModel", "arg1": 5, "arg2": None}
        }
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "config.yaml")
            write_config(config_orig, path)
            config = read_config(path)
        self.assertEqual(config, config_gt)

    def test_types(self):
        config = {
            "model": "some-model",
            "model_config": {"_type": "SimpleModel", "arg1": 5, "arg2": None}
        }
        model = SimpleModel(config)
        self.assertEqual(model.config["model"], config["model"])
        self.assertEqual(model.config["model_config"]["arg1"], config["model_config"]["arg1"])
        self.assertEqual(model.config["model_config"]["arg2"], config["model_config"]["arg2"])

    def test_flat_nested(self):
        config = {
            "a": {
                "b": 4,
                "_hopt": {
                    "b": 5
                },
                "c": {
                    "d": 1
                }
            },
            "e": "aoeu",
            "f": [
                {"i": 9, "_hopt": {"g": 7}},
                {"_hopt": {"h": 8}}
            ]
        }
        flat_gt = {
            "a.b": 4,
            "a.c.d": 1,
            "e": "aoeu",
            "f.0.i": 9,
            "_hopt": {
                "a.b": 5,
                "f.0.g": 7,
                "f.1.h": 8
            }
        }
        self.assertTrue(has_hopts(config))
        flat = as_flat_config(config)
        self.assertEqual(flat, flat_gt)
        nested = as_nested_config(flat)
        self.assertEqual(nested, config)

    def test_optional_values(self):
        config = {
            "_hopt": {
                "b": 5
            }
        }

        default = {"a": 4, "b": 1}
        gt = {"a": 4, "b": 5}
        self.assertEqual(prepare_config(default, config), gt)

        default = {"a": 4}
        gt = {"a": 4}
        self.assertEqual(prepare_config(default, config), gt)

    def test_update_config(self):
        config = {
            "a": {
                "b": 2
            },
            "c": [
                {"d": 4},
                {"e": 5}
            ]
        }
        patch = {
            "a": {"b": 2.5},
            "f": 6,
            "c": [
                {},
                {"g": 7},
                {"h": 8}
            ]
        }
        gt = {
            "a": {
                "b": 2.5
            },
            "f": 6,
            "c": [
                {"d": 4},
                {"e": 5, "g": 7},
                {"h": 8}
            ]
        }
        self.assertEqual(update_config(config, patch), gt)


if __name__ == "__main__":
    main()
