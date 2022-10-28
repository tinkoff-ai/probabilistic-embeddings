import argparse
import io
import os
from collections import OrderedDict
from copy import deepcopy

import mxnet as mx
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from probabilistic_embeddings.config import update_config, as_flat_config, as_nested_config
from probabilistic_embeddings.io import read_yaml, write_yaml
from probabilistic_embeddings.dataset import DatasetCollection
from probabilistic_embeddings.dataset.common import DatasetWrapper
from probabilistic_embeddings.runner import Runner


def parse_arguments():
    parser = argparse.ArgumentParser("Generate configs for reality check from templates.")
    parser.add_argument("templates", help="Templates root.")
    parser.add_argument("dst", help="Target configs root.")
    parser.add_argument("--best", help="Best hopts root.")
    parser.add_argument("--embeddings-dims", help="Coma-separated list of required embeddings dimensions.",
                        default="128,512")
    return parser.parse_args()


def get_best_hopts(path):
    """Load best hyperparameters from wandb config.

    If file doesn't exists, returns empty dictionary.
    """
    if not path.exists():
        return {}
    print("Load best parameters from {}.".format(path))
    flat_config = {k: v["value"] for k, v in read_yaml(path).items()
                   if not k.startswith("wandb") and not k.startswith("_")
                   and not k.startswith("dataset_params")
                   and not k.startswith("metrics_params")}
    config = as_nested_config(flat_config)
    config.pop("git_commit", None)
    default_keys = set(Runner.get_default_config())
    for k in config:
        if k not in default_keys:
            raise RuntimeError("Unknown parameter: {}.".format(k))
    return config


def main(args):
    src = Path(args.templates)
    dst = Path(args.dst)
    best = Path(args.best) if args.best is not None else None
    filenames = {path.relative_to(src) for path in src.glob("reality-*.yaml")}
    for required in [Path("reality-base.yaml"), Path("reality-datasets.yaml")]:
        try:
            filenames.remove(required)
        except KeyError:
            raise FileNotFoundError("Need {} template.".format(required))
    template = read_yaml(src / "reality-base.yaml")
    datasets = read_yaml(src / "reality-datasets.yaml")
    dims = [int(s) for s in args.embeddings_dims.split(",")]
    for filename in filenames:
        print(filename.stem)
        _, pipeline = str(filename.stem).split("-", 1)
        pipeline_patch = read_yaml(src / filename)
        for dataset, dataset_patch in datasets.items():
            for dim in dims:
                filename =  "{}-{}-{}.yaml".format(pipeline, dim, dataset)
                config = update_config(template, dataset_patch)
                config = update_config(config, pipeline_patch)
                if best is not None:
                    config = update_config(config, get_best_hopts(best / filename))
                config = update_config(config, {"model_params": {"distribution_params": {"dim": dim}}})
                write_yaml(config, dst / filename)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
