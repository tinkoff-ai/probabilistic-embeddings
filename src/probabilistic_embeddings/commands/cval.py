import os
import subprocess

import torch

from ..config import read_config, write_config, prepare_config
from ..dataset import DatasetCollection
from ..io import read_yaml, write_yaml
from ..runner import parse_logger, Runner
from .common import aggregate_metrics, log_wandb_metrics, make_directory, patch_cmd, print_nested


def get_train_root(train_root, fold=None):
    parts = [train_root]
    if fold is not None:
        parts.append("fold-{}".format(fold))
    path = os.path.join(*parts)
    make_directory(path)
    return path


def get_gpus():
    if not torch.cuda.is_available():
        return []
    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if visible_gpus is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES required")
    gpus = list(map(int, visible_gpus.split(",")))
    return gpus


def patch_logger(logger, fold):
    logger_type, project, experiment, group = parse_logger(logger)
    if logger_type == "tensorboard":
        logger = "tensorboard"
    elif logger_type == "wandb":
        if group is None:
            group = experiment
        experiment = experiment + "-fold-{}".format(fold)
        logger = ":".join([logger_type, project, experiment, group])
    return logger


def get_train_cmd(args, fold):
    env = dict(os.environ)
    gpus = get_gpus()
    if gpus:
        gpu = gpus[fold % len(gpus)]
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    top_root = args.train_root
    train_root = get_train_root(top_root, fold)

    config = read_config(args.config) if args.config is not None else {}
    config = prepare_config(Runner, config)
    config["dataset_params"] = config["dataset_params"] or {}
    config["dataset_params"]["validation_fold"] = fold
    config["seed"] = config["seed"] + fold
    config_path = os.path.join(train_root, "config.yaml")
    write_config(config, config_path)

    new_args = {
        "--config": config_path,
        "--train-root": train_root,
        "--logger": patch_logger(args.logger, fold)
    }
    if args.checkpoint is not None:
        new_args["--checkpoint"] = args.checkpoint.replace("{fold}", str(fold))
    for key in ["WANDB_SWEEP_ID", "WANDB_RUN_ID", "WANDB_SWEEP_PARAM_PATH"]:
        env.pop(key, None)
    return env, patch_cmd("train", new_args)


def run_parallel(cmds):
    processes = []
    for env, cmd in cmds:
        processes.append(subprocess.Popen(cmd, env=env, cwd=os.getcwd()))
    for p in processes:
        p.wait()
        if p.returncode != 0:
            raise RuntimeError("Subprocess failed with code {}.".format(p.returncode))


def cval(args):
    """Train and eval multiple models using cross validation.

    For wandb logging, multiple runs are grouped together.

    """
    config = read_config(args.config) if args.config is not None else {}
    dataset_config = config.pop("dataset_params", None)
    dataset_config = prepare_config(DatasetCollection.get_default_config(), dataset_config)
    num_folds = dataset_config["num_validation_folds"]

    if not os.path.isdir(args.train_root):
        os.mkdir(args.train_root)

    # Run jobs in chunks to prevent gpu collisions (gpu index is a function of the fold number).
    num_parallel = max(len(get_gpus()), 1)
    for i in range(0, num_folds, num_parallel):
        cmds = [get_train_cmd(args, fold) for fold in range(i, min(num_folds, i + num_parallel))]
        run_parallel(cmds)

    # Aggregate and dump metrics.
    metrics = aggregate_metrics(*[read_yaml(os.path.join(get_train_root(args.train_root, fold), "metrics.yaml"))
                                  for fold in range(num_folds)])
    log_wandb_metrics(metrics, args.logger)
    metrics["num_folds"] = num_folds
    print_nested(metrics)
    write_yaml(metrics, os.path.join(args.train_root, "metrics.yaml"))
    return metrics
