import copy
import os
import math
import random
import shutil
import sys
import tempfile
import traceback
from collections import OrderedDict

import optuna
import wandb

from ..config import read_config, write_config, prepare_config, update_config
from ..config import has_hopts, as_flat_config, as_nested_config, CONFIG_HOPT, ConfigError
from ..runner import Runner, parse_logger
from ..trainer import Trainer
from .common import make_directory
from .cval import cval
from .basic import train


def patch_logger(logger, logger_suffix, run_id):
    logger_type, project, experiment, group = parse_logger(logger)
    if logger_type == "tensorboard":
        return logger
    elif logger_type != "wandb":
        raise ValueError("Unknown logger: {}".format(logger_type))
    if group is None:
        group = experiment + "-" + logger_suffix
    experiment = experiment + "-" + logger_suffix + "-" + run_id
    logger = ":".join([logger_type, project, experiment, group])
    return logger


def suggest(trial, name, spec):
    if not isinstance(spec, (dict, OrderedDict)):
        raise ValueError("Dictionary HOPT specification is expected, got {} for {}.".format(spec, name))
    if "values" in spec:
        return trial.suggest_categorical(name, spec["values"])
    distribution = spec.get("distribution", "uniform")
    if distribution == "log_uniform":
        return trial.suggest_loguniform(name, math.exp(spec["min"]), math.exp(spec["max"]))
    if distribution != "uniform":
        raise ConfigError("Unknown distribution: {}.".format(distribution))
    if isinstance(spec["min"], float) or isinstance(spec["max"], float):
        return trial.suggest_uniform(name, spec["min"], spec["max"])
    else:
        return trial.suggest_int(name, spec["min"], spec["max"])


class HoptWorker:
    def __init__(self, args, logger_suffix, run_cval):
        self._args = args
        self._logger_suffix = logger_suffix
        self._run_cval = run_cval

    def run_worker(self, optuna_trial=None):
        random.seed()
        run_id = str(random.randint(0, 1e15))
        train_root = os.path.join(self._args.train_root, run_id)
        make_directory(train_root)

        logger = patch_logger(self._args.logger, self._logger_suffix, run_id)
        logger_type, project, experiment, group = parse_logger(logger)
        if optuna_trial is None:
            # Get config from WandB.
            wandb_logger = wandb.init(project=project, name=experiment, group=group)
            flat = dict(wandb_logger.config)
            config = as_nested_config(flat)
        else:
            if logger_type == "wandb":
                wandb.init(project=project, name=experiment, group=group, reinit=True)
            config = read_config(self._args.config)
            flat = as_flat_config(config)
            for name, spec in flat.pop(CONFIG_HOPT, {}).items():
                flat[name] = suggest(optuna_trial, name, spec)
            config = as_nested_config(flat)
        config["seed"] = random.randint(0, (1 << 16))
        config_path = os.path.join(train_root, "config.yaml")
        write_config(config, config_path)

        args = copy.copy(self._args)
        args.train_root = train_root
        args.config = config_path
        args.logger = logger
        if self._run_cval:
            metrics = cval(args)
        else:
            metrics = train(args)
        trainer_config = prepare_config(Trainer, config.get("trainer_params", None))
        metric = metrics[trainer_config["selection_dataset"]][trainer_config["selection_metric"]]
        if isinstance(metric, (dict, OrderedDict)) and ("mean" in metric):
            metric = metric["mean"]
        if args.clean:
            shutil.rmtree(train_root)
        return float(metric)

    def __call__(self, optuna_trial=None):
        try:
            return self.run_worker(optuna_trial)
        except Exception:
            print(traceback.print_exc(), file=sys.stderr)
            exit(1)


def make_sweep(project, experiment, config_path):
    config = read_config(config_path)
    if not has_hopts(config):
        raise RuntimeError("No hyper parameters to optimize")
    runner_config = prepare_config(Runner, config)
    trainer_config = prepare_config(Trainer, config["trainer_params"])
    flat_config = as_flat_config(config)
    # Replace {k: v} with {k: {"value": v}}, because WandB expects HOpt format even for constant values.
    flat_hopts = flat_config.pop(CONFIG_HOPT)
    flat_config = {k: {"value": v} for k, v in flat_config.items()}
    flat_config.update(flat_hopts)
    hopt_backend, hopt_method = runner_config["hopt_backend"].split("-")
    assert hopt_backend == "wandb"
    sweep_config = {
        "name": experiment,
        "method": hopt_method,
        "early_terminate": {
            "type": "hyperband",
            "min_iter": trainer_config["early_stop_patience"],
            "eta": trainer_config["early_stop_patience"]
        },
        "metric": {
            "name": "{}_epoch/{}".format(trainer_config["selection_metric"], trainer_config["selection_dataset"]),
            "goal": ("minimize" if trainer_config["selection_minimize"] else "maximize")
        },
        "parameters": dict(flat_config)
    }
    sweep_id = wandb.sweep(sweep_config, project=project)
    return sweep_id, runner_config["num_hopt_trials"]


def hopt_wandb(args, run_cval=True):
    logger_type, project, experiment, group = parse_logger(args.logger)
    if logger_type != "wandb":
        raise RuntimeError("Need wandb logger for wandb-based hyperparameter search")
    if experiment is None:
        raise RuntimeError("Need experiment name for hyperparameter search")
    if args.sweep_id is not None:
        sweep_id, count = args.sweep_id, None
    else:
        sweep_id, count = make_sweep(project, experiment, args.config)
    worker = HoptWorker(args, "sweep-" + sweep_id, run_cval)
    wandb.agent(sweep_id, function=worker, project=project, count=count)
    print("Finished sweep", sweep_id)


def hopt_optuna(args, run_cval=True):
    SAMPLERS = {
        "tpe": optuna.samplers.TPESampler
    }

    if args.sweep_id is not None:
        raise ValueError("Can't attach to sweep ID using Optuna.")
    config = read_config(args.config) if args.config is not None else {}
    if not has_hopts(config):
        raise RuntimeError("No hyper parameters to optimize")
    runner_config = prepare_config(Runner, config)
    trainer_config = prepare_config(Trainer, runner_config["trainer_params"])
    hopt_backend, hopt_method = runner_config["hopt_backend"].split("-")
    assert hopt_backend == "optuna"
    study = optuna.create_study(direction="minimize" if trainer_config["selection_minimize"] else "maximize",
                                sampler=SAMPLERS[hopt_method]())
    worker = HoptWorker(args, "optuna", run_cval)
    study.optimize(worker,
                   n_trials=runner_config["num_hopt_trials"])


BACKENDS = {
    "wandb-bayes": hopt_wandb,
    "wandb-random": hopt_wandb,
    "optuna-tpe": hopt_optuna
}


def hopt(args, run_cval=True):
    """Run hopt tuning."""
    make_directory(args.train_root)
    config = prepare_config(Runner, args.config)
    config = update_config(config, config["hopt_params"])
    del config["seed"]
    with tempfile.NamedTemporaryFile("w") as fp:
        write_config(config, fp)
        fp.flush()
        args = copy.deepcopy(args)
        args.config = fp.name
        BACKENDS[config["hopt_backend"]](args, run_cval=run_cval)
