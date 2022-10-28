import os
import sys
import tempfile
from collections import defaultdict, OrderedDict
from contextlib import contextmanager

import numpy as np
import wandb
from catalyst.utils import prepare_cudnn

from ..config import as_flat_config, as_nested_config
from ..io import write_yaml
from ..runner import parse_logger


def setup():
    """Setup global settings before run."""
    prepare_cudnn(benchmark=True)


@contextmanager
def folder_or_tmp(root=None):
    """Use temp directory if None is providided."""
    if root is None:
        with tempfile.TemporaryDirectory() as root:
            yield root
    else:
        yield root


def make_directory(root, subdir=None):
    """Create directories in a more safe way than os.makedirs.

    If root can't be created, exception is raised.

    """
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.isdir(root):
        raise FileNotFoundError("Can't create directory {}.".format(root))
    if subdir is not None:
        os.makedirs(os.path.join(root, subdir), exist_ok=True)


def patch_cmd(cmd, options):
    """Patch argv of the current run with given options."""
    argv = list(sys.argv[2:])
    to_del = [k for k, v in options.items() if v is None]
    options = {k: v for k, v in options.items() if v is not None}
    for k in to_del:
        try:
            index = argv.index(k)
            del argv[index + 1]
            del argv[index]
        except ValueError:
            pass
    for i, arg in enumerate(argv):
        if arg in options:
            argv[i + 1] = options.pop(arg)
    for k, v in options.items():
        argv.extend([k, v])
    return [sys.executable, "-m", "probabilistic_embeddings", cmd] + argv


def aggregate_metrics(*metrics):
    """Merge multiple evaluations and compute mean/std."""
    flat = defaultdict(list)
    for m in metrics:
        for k, v in as_flat_config(m).items():
            flat[k].append(v)
    aggregated_flat = defaultdict(OrderedDict)
    for k, vs in flat.items():
        aggregated_flat[k] = np.mean(vs)
        aggregated_flat[k + "_std"] = np.std(vs)
    aggregated = as_nested_config(aggregated_flat)
    return aggregated


def print_nested(nested):
    """Print nested dictionary in human-readable way."""
    write_yaml(nested, sys.stdout)


def log_wandb_metrics(metrics, logger):
    if isinstance(logger, str):
        logger_type, project, experiment, group = parse_logger(logger)
        if logger_type != "wandb":
            return
        logger = wandb.init(project=project, name=experiment, group=group)

    step = int(round(metrics.pop("epoch"))) if "epoch" in metrics else None
    for top_k, top_v in metrics.items():
        if not isinstance(top_v, (dict, OrderedDict)):
            logger.log({top_k: top_v}, step=step)
        else:
            for k, v in top_v.items():
                logger.log({"{}_epoch/{}".format(k, top_k): v}, step=step)
