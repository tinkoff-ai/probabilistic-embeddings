import os
from collections import OrderedDict

import torch
from catalyst import dl
from catalyst.loggers.tensorboard import TensorboardLogger
from git import Repo, InvalidGitRepositoryError

from ._workarounds import AfterForkWandbLogger as WandbLogger
from .config import prepare_config, update_config, as_flat_config, ConfigError
from .criterion import Criterion, CriterionCallback
from .dataset import DatasetCollection
from .initializer import Initializer
from .metrics import Metrics
from .model import Model
from .torch import get_base_module
from .trainer import Trainer


def parse_logger(logger):
    """Parse logger specification.

    Returns:
        Tuple of logger_type, project, experiment, group.
    """
    if logger is None:
        logger = "tensorboard"
    project = None
    experiment = None
    group = None
    tokens = logger.split(":")
    logger_type = tokens[0]
    if logger_type == "tensorboard":
        if len(tokens) != 1:
            raise ValueError("Bad tensorboard spec: {}.".format(logger))
    elif logger_type == "wandb":
        if len(tokens) == 3:
            logger_type, project, experiment = tokens
        elif len(tokens) == 4:
            logger_type, project, experiment, group = tokens
        else:
            raise ValueError("Bad wandb spec: {}".format(logger))
    else:
        raise ValueError("Bad logger spec: {}.".format(logger))
    return logger_type, project, experiment, group


def get_git_commit():
    try:
        repo = Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except InvalidGitRepositoryError:
        return None


class Runner(dl.IRunner):
    """Configurable runner class for training and evaluation.

    Args:
        root: Training folder (checkpoints, logs, etc.).
        data_root: Dataset root folder.
        config: Runner config dictionary or None to use default.
        logger: Logger to use ("tensorboard" or "wandb:<project-name>").
        initial_checkpoint: Path to checkpoint to resume training.
        no_strict_init: Skip checkpoint mismatch errors.
        from_stage: Start training from specified stage index.
    """

    STAGE_TRAIN = "train"
    STAGE_TEST = "test"

    @staticmethod
    def get_default_config(dataset_params=None,
                           model_params=None,
                           initializer_params=None,
                           criterion_params=None,
                           trainer_params=None,
                           metrics_params=None,
                           stages=None, stage_resume="best", resume_prefixes=None,
                           fp16=False, amp_head=False, initial_grad_scale=65536.0, grad_scale_growth_interval=2000,
                           seed=42,
                           num_evaluation_seeds=10,
                           num_hopt_trials=50,
                           hopt_backend="wandb-bayes",
                           hopt_params=None):
        """Get runner parameters.

        Args:
            dataset_params: Parameters of :class:`DatasetCollection`.
            model_params: Parameters of :class:`Model`.
            initializer_params: Parameters of :class:`Initializer`.
            criterion_params: Parameters of the loss function.
            trainer_params: Parameters of :class:`Trainer`.
            metrics_params: Parameters of :class:`Metrics`.
            stages: List of config patches for each stage. Train single stage by default.
            stage_resume: Type of model preloading between stages (one of "best", "last").
            resume_prefixes: Coma-separated list of parameter name prefixes for model preloading.
            fp16: Whether to use FP16 training or not.
            amp_head: Whether to use FP16 for classifier and criterion or not (when fp16=True).
            initial_grad_scale: Initial grad scale used for FP16 training.
            grad_scale_growth_interval: Number of batches without overflow before gradient scale growth.
            seed: Random seed.
            num_evaluation_seeds: Number of different seeds used for evaluation.
            num_hopt_trials: Number of runs used for hyperparameter tuning.
            hopt_backend: Type of hyperparameter search algorithm ("wandb-bayes", "wandb-random", "optuna-tpe").
            hopt_params: Config patch used during hyper-parameter tuning.
        """
        return OrderedDict([
            ("dataset_params", dataset_params),
            ("model_params", model_params),
            ("initializer_params", initializer_params),
            ("criterion_params", criterion_params),
            ("trainer_params", trainer_params),
            ("metrics_params", metrics_params),
            ("stages", stages),
            ("stage_resume", stage_resume),
            ("resume_prefixes", resume_prefixes),
            ("fp16", fp16),
            ("amp_head", amp_head),
            ("initial_grad_scale", initial_grad_scale),
            ("grad_scale_growth_interval", grad_scale_growth_interval),
            ("seed", seed),
            ("num_evaluation_seeds", num_evaluation_seeds),
            ("num_hopt_trials", num_hopt_trials),
            ("hopt_backend", hopt_backend),
            ("hopt_params", hopt_params)
        ])

    def __init__(self, root, data_root, *, config,
                 logger="tensorboard",
                 initial_checkpoint=None,
                 no_strict_init=False,
                 from_stage=None):
        super().__init__()
        self._base_config = prepare_config(self, config)

        self._root = root
        self._data_root = data_root
        self._logger = logger
        self._initial_checkpoint = initial_checkpoint
        self._no_strict_init = no_strict_init
        self._from_stage = from_stage

        for stage_config in self._base_config["stages"] or []:
            for key in ["stages", "fp16", "initial_grad_scale", "num_hopt_trials", "hopt_backend"]:
                assert key in self._base_config
                if key in stage_config:
                    raise ConfigError("Can't overwrite {} in a stage".format(key))

    def get_stage_config(self, stage):
        stages = self._base_config["stages"] or [{}]
        if stage == self.STAGE_TEST:
            stage_id = len(stages) - 1
        else:
            stage_id = int(stage.split("-")[1])
        stage_config = stages[stage_id]
        config = update_config(self._base_config, stage_config)
        config.pop("stages")
        return config

    def init_stage(self, stage):
        """Initialize config and base structures for the stage."""
        self._stage = self.STAGE_TEST if stage == self.STAGE_TEST else self.STAGE_TRAIN
        self._config = self.get_stage_config(stage)
        self._datasets = DatasetCollection(self._data_root, config=self._config["dataset_params"])
        self._metrics = Metrics(self._datasets.num_train_classes, self._datasets.openset, config=self._config["metrics_params"])
        self._loaders = None

    @property
    def datasets(self):
        return self._datasets

    @property
    def stages(self):
        if self._stage == self.STAGE_TEST:
            return [self.STAGE_TEST]
        assert self._stage == self.STAGE_TRAIN
        keys = [self.STAGE_TRAIN + "-" + str(i)
                for i in range(len(self._base_config["stages"] or [{}]))]
        if self._from_stage is not None:
            if self._from_stage >= len(keys):
                raise ConfigError("Can't start from stage {}. Total number of stages is {}.".format(
                    self._from_stage, len(keys)))
            keys = keys[self._from_stage:]
        return keys

    def get_stage_len(self, stage):
        if stage == self.STAGE_TEST:
            return 1
        return self._get_trainer().get_num_epochs()

    def on_stage_start(self, runner):
        self.init_stage(self.stage_key)
        super().on_stage_start(runner)

    def on_epoch_start(self, runner):
        super().on_epoch_start(runner)
        self.epoch_metrics["_epoch_"]["model_hash"] = sum([p.sum().item() for p in self.model["model"].parameters()])

    def on_epoch_end(self, runner):
        if self.stage_key != self.STAGE_TEST:
            self.epoch_metrics["_epoch_"]["stage"] = int(self.stage_key.split("-")[1])
        super().on_epoch_end(runner)

    @property
    def seed(self) -> int:
        return self._config["seed"] + 1

    def get_engine(self):
        if not torch.cuda.is_available():
            return dl.DeviceEngine()
        elif self._base_config["fp16"]:
            engine_cls = dl.DataParallelAMPEngine if torch.cuda.device_count() > 1 else dl.AMPEngine
            return engine_cls(scaler_kwargs={"init_scale": self._base_config["initial_grad_scale"],
                                             "growth_interval": self._base_config["grad_scale_growth_interval"]})
        else:
            engine_cls = dl.DataParallelEngine if torch.cuda.device_count() > 1 else dl.DeviceEngine
            return engine_cls()

    def get_model(self, stage):
        training = stage != self.STAGE_TEST
        stage_index = list(self.stages).index(stage)
        model = Model(self._datasets.num_train_classes,
                      priors=self._datasets.train_priors,
                      amp_classifier=self._config["amp_head"],
                      config=self._config["model_params"])
        print("Total model parameters:", model.num_parameters)
        if training and (stage_index == 0):
            initializer = Initializer(config=self._config["initializer_params"])
            initializer(model, train_loader=self.get_loaders(stage)["train"])
        checkpoint_path = self._initial_checkpoint
        if training and (stage_index > 0) and (self._config["stage_resume"] is not None):
            if self._config["stage_resume"] not in {"best", "last"}:
                raise ConfigError("Unexpected resume type: {}.".format(self._config["stage_resume"]))
            checkpoint_path = os.path.join(self._root, "checkpoints", self._config["stage_resume"] + ".pth")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError("Can't find checkpoint {}.".format(checkpoint_path))
        if checkpoint_path is not None:
            print("Load", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")["model_model_state_dict"]
            if training and self._config["resume_prefixes"]:
                new_checkpoint = {}
                for prefix in self._config["resume_prefixes"].split(","):
                    if not prefix:
                        raise ConfigError("Empty resume prefix.")
                    parameters = {k: v for k, v in checkpoint.items()
                                  if k.startswith(prefix)}
                    if not parameters:
                        raise ConfigError("Unknown prefix {}.".format(prefix))
                    new_checkpoint.update(parameters)
                checkpoint = new_checkpoint
                missing, unexpected = model.load_state_dict(checkpoint, strict=False)
                if unexpected:
                    raise RuntimeError("Unexpected state dict keys: {}.".format(unexpected))
            else:
                model.load_state_dict(checkpoint, strict=not self._no_strict_init)
        # Separate embedder from model to allow multi-gpu computation.
        return {"model": model, "embedder": model.embedder, "scorer": model.scorer}

    def get_criterion(self, stage):
        if stage == self.STAGE_TEST:
            return None
        return Criterion(config=self._config["criterion_params"])

    def get_optimizer(self, stage, model):
        if stage == self.STAGE_TEST:
            return None
        return self._get_trainer().get_optimizer(model["model"])

    def get_scheduler(self, stage, optimizer):
        if stage == self.STAGE_TEST:
            return None
        return self._get_trainer().get_scheduler(optimizer)

    def get_loaders(self, stage):
        return self._datasets.get_loaders(train=(stage != self.STAGE_TEST))

    def get_loggers(self):
        logger_type, project, experiment, group = parse_logger(self._logger)
        if logger_type == "tensorboard":
            logger = TensorboardLogger(logdir=self._root, use_logdir_postfix=True)
        elif logger_type == "wandb":
            kwargs = {}
            if group is not None:
                kwargs["group"] = group
            logger = WandbLogger(project=project, name=experiment, **kwargs)
            logger.init()
            self._wandb_id = logger.run.id
            logger.run.config.update(as_flat_config(self._base_config))
            logger.run.config.update({"git_commit": get_git_commit()})
        else:
            raise ValueError("Unknown logger: {}.".format(self._logger))
        loggers = {
            "_console": dl.ConsoleLogger(),
            "_csv": dl.CSVLogger(logdir=self._root, use_logdir_postfix=True),
            "main": logger
        }
        return loggers

    def get_callbacks(self, stage):
        callbacks = {}
        callbacks["verbose"] = dl.TqdmCallback()

        model = get_base_module(self.model["model"])

        if stage != self.STAGE_TEST:
            criterion = self.get_criterion(stage)
            criterion_inputs = {"embeddings": "embeddings"}
            criterion_outputs = {"labels": "labels"}
            if model.classification:
                criterion_inputs["logits"] = "logits"
                if model.has_final_weights:
                    criterion_outputs["final_weights"] = "final_weights"
                    criterion_outputs["target_embeddings"] = "target_embeddings"
                if model.has_final_bias:
                    criterion_outputs["final_bias"] = "final_bias"
                if model.has_final_variance:
                    criterion_outputs["final_variance"] = "final_variance"
            callbacks["criterion"] = dl.ControlFlowCallback(
                CriterionCallback(amp=self._config["amp_head"],
                                  input_key=criterion_inputs, target_key=criterion_outputs, metric_key="loss"),
                loaders="train"
            )
            callbacks.update(self._get_trainer().get_callbacks(
                checkpoints_path=os.path.join(self._root, "checkpoints"),
                loss_key="loss"
            ))

        datasets = self._datasets.get_datasets(train=(stage != self.STAGE_TEST), transform=False)
        for name, dataset in datasets.items():
            suffix = "_" + name if name != "train" else ""
            if dataset.classification:
                dataset_callbacks = self._metrics.get_classification_callbacks(
                    train=(name == "train"),
                    labels_key="labels" + suffix,
                    embeddings_key="embeddings" + suffix,
                    target_embeddings_key="target_embeddings" + suffix if model.classification else None,
                    logits_key="logits" + suffix if model.classification else None,
                    confidences_key="confidences" + suffix if model.distribution.has_confidences else None,
                    quality_key="quality" + suffix if dataset.has_quality else None
                )
            else:
                kwargs = {}
                if model.distribution.has_confidences:
                    kwargs["confidences_key"] = "confidences" + suffix
                dataset_callbacks = self._metrics.get_verification_callbacks(
                    train=(name == "train"),
                    labels_key="labels" + suffix,
                    scores_key="scores" + suffix,
                    **kwargs
                )
            for callback_name, callback in dataset_callbacks.items():
                callbacks[name + "_" + callback_name] = dl.ControlFlowCallback(callback, loaders=name)
        return callbacks

    def train(self, **kwargs):
        self._stage = self.STAGE_TRAIN
        self.run()

    def evaluate(self):
        self._stage = self.STAGE_TEST
        self.run()
        return self.epoch_metrics

    def get_loader_suffix(self):
        return "" if self.loader_key == "train" else "_" + self.loader_key


    def handle_batch(self, batch):
        images, labels = batch[:2]
        quality = batch[2] if len(batch) > 2 else None
        batch = {
            "labels": labels
        }
        if quality is not None:
            batch["quality"] = quality
        if isinstance(images, torch.Tensor):
            batch["images"] = images
            batch, metrics = self._handle_classification_batch(batch)
        else:
            batch["images1"], batch["images2"] = images
            batch, metrics = self._handle_verification_batch(batch)

        suffix = self.get_loader_suffix()
        self.batch = {k + suffix: v for k, v in batch.items()}
        self.batch_metrics.update({k + suffix: (v.item() if isinstance(v, torch.Tensor) else v)
                                   for k, v in metrics.items()})

    def _get_trainer(self):
        return Trainer(config=self._config["trainer_params"])

    def _handle_classification_batch(self, batch):
        is_train = self.loader_key == "train"
        results = self.model["model"](batch["images"], batch["labels"] if is_train else None)
        batch["embeddings"] = results["distributions"]

        model = get_base_module(self.model["model"])
        if model.classification:
            batch["logits"] = results["logits"]
            if model.has_final_weights:
                batch["final_weights"] = model.get_final_weights()
                if is_train or (not self.loader.dataset.openset):
                    batch["target_embeddings"] = model.get_target_embeddings(batch["labels"])
            if model.has_final_bias:
                batch["final_bias"] = model.get_final_bias()
            if model.has_final_variance:
                batch["final_variance"] = model.get_final_variance()
        if model.distribution.has_confidences:
            batch["confidences"] = model.distribution.confidences(batch["embeddings"])

        metrics = {}
        if is_train:
            with torch.no_grad():
                metrics = model.statistics(results)
                metrics.update(OrderedDict([
                    ("infnans", 1 - batch["embeddings"].isfinite().float().mean())
                ]))
        return batch, metrics

    def _handle_verification_batch(self, batch):
        batch["embeddings1"] = self.model["embedder"](batch["images1"])
        batch["embeddings2"] = self.model["embedder"](batch["images2"])
        batch["scores"] = self.model["scorer"](batch["embeddings1"], batch["embeddings2"])
        model = get_base_module(self.model["model"])
        if model.distribution.has_confidences:
            confidences1 = model.distribution.confidences(batch["embeddings1"])  # (B).
            confidences2 = model.distribution.confidences(batch["embeddings2"])  # (B).
            batch["confidences"] = torch.minimum(confidences1, confidences2)  # (B).
        metrics = {}
        return batch, metrics
