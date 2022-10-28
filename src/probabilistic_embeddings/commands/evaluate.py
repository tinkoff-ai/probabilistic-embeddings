import os
import subprocess
from collections import OrderedDict

from ..config import read_config, write_config, prepare_config, as_flat_config, as_nested_config
from ..io import read_yaml, write_yaml
from ..runner import Runner, parse_logger
from .common import aggregate_metrics, log_wandb_metrics, make_directory, patch_cmd, print_nested


def get_train_root(train_root, seed=None):
    parts = [train_root]
    if seed is not None:
        parts.append("seed-{}".format(seed))
    path = os.path.join(*parts)
    make_directory(path)
    return path


def patch_logger(logger, seed):
    logger_type, project, experiment, group = parse_logger(logger)
    if logger_type == "tensorboard":
        logger = "tensorboard"
    elif logger_type == "wandb":
        if group is None:
            group = experiment
        experiment = experiment + "-seed-{}".format(seed)
        logger = ":".join([logger_type, project, experiment, group])
    return logger


def get_train_cmd(args, seed, run_cval=True):
    top_root = args.train_root
    train_root = get_train_root(top_root, seed)

    config = read_config(args.config) if args.config is not None else {}
    config["seed"] = seed
    config_path = os.path.join(train_root, "config.yaml")
    write_config(config, config_path)

    new_args = {
        "--config": config_path,
        "--train-root": train_root,
        "--logger": patch_logger(args.logger, config["seed"])
    }
    if args.checkpoint is not None:
        new_args["--checkpoint"] = args.checkpoint.replace("{seed}", str(seed))
    cmd = "cval" if run_cval else "train"
    return (os.environ, patch_cmd(cmd, new_args))


def read_metrics_no_std(path):
    metrics = read_yaml(path)
    flat = as_flat_config(metrics)
    flat_no_std = OrderedDict([(k, v) for k, v in flat.items() if "_std" not in k])
    metrics_no_std = as_nested_config(flat_no_std)
    return metrics_no_std


def evaluate(args, run_cval=True):
    """Train and eval multiple models using cross validation and multiple seeds."""
    config = read_config(args.config) if args.config is not None else {}
    config = prepare_config(Runner.get_default_config(), config)
    num_seeds = config["num_evaluation_seeds"]

    make_directory(args.train_root)

    for seed in range(args.from_seed, num_seeds):
        env, cmd = get_train_cmd(args, seed, run_cval=run_cval)
        subprocess.call(cmd, env=env, cwd=os.getcwd())

    # Aggregate and dump metrics. Exclude STD metrics to prevent ambiguity between STD of means and mean STD.
    metrics = aggregate_metrics(*[read_metrics_no_std(os.path.join(get_train_root(args.train_root, seed), "metrics.yaml"))
                                  for seed in range(num_seeds)])
    log_wandb_metrics(metrics, args.logger)
    metrics["num_seeds"] = num_seeds
    print_nested(metrics)
    write_yaml(metrics, os.path.join(args.train_root, "metrics.yaml"))
