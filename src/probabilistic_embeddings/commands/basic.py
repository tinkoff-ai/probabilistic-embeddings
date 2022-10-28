import copy
import os

import wandb

from ..io import write_yaml
from ..runner import Runner
from .common import folder_or_tmp, log_wandb_metrics, make_directory, parse_logger, print_nested, setup


def train(args):
    """Train single model and eval best checkpoint."""
    setup()
    if args.train_root is None:
        raise RuntimeError("Need training root path")
    logger_type, project, experiment, group = parse_logger(args.logger)
    make_directory(args.train_root)
    runner = Runner(args.train_root, args.data,
                    config=args.config, logger=args.logger,
                    initial_checkpoint=args.checkpoint,
                    no_strict_init=args.no_strict_init,
                    from_stage=args.from_stage)
    if (args.from_stage or 0) >= 0:
        if args.config is not None:
            print("Run training with config:")
            with open(args.config) as fp:
                print(fp.read())
        runner.train(verbose=True)
        epoch = runner.global_sample_step + 1 if logger_type == "wandb" else runner.global_epoch_step
    else:
        print("Skip training.")
        runner.on_experiment_start(runner)
        runner.stage_key = runner.STAGE_TEST
        runner.on_stage_start(runner)
        epoch = 0

    # Eval best checkpoint.
    test_args = copy.copy(args)
    test_args.checkpoint = os.path.join(args.train_root, "checkpoints", "best.pth")
    test_args.logger = "tensorboard"  # Don't create another Wandb logger.
    metrics = test(test_args)
    metrics["epoch"] = epoch
    if logger_type == "wandb":
        logger = wandb.init(project=project, name=experiment, group=group, resume=runner._wandb_id)
        log_wandb_metrics(metrics, logger)
    write_yaml(metrics, os.path.join(args.train_root, "metrics.yaml"))
    return metrics


def test(args):
    """Compute metrics for checkpoint."""
    setup()
    if args.checkpoint is None:
        raise RuntimeError("Need checkpoint for evaluation")
    with folder_or_tmp(args.train_root) as root:
        runner = Runner(root, args.data,
                        config=args.config, logger=args.logger,
                        initial_checkpoint=args.checkpoint,
                        no_strict_init=args.no_strict_init)
        metrics = runner.evaluate()
    print_nested(metrics)
    return metrics
