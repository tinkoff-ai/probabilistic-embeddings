import argparse
import tempfile
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from catalyst.utils.misc import maybe_recursive_call
from catalyst.utils.torch import any2device
from probabilistic_embeddings.config import read_config, update_config
from probabilistic_embeddings.runner import Runner


class NoLogRunner(Runner):
    def get_loggers(self):
        return {}


def parse_arguments():
    parser = argparse.ArgumentParser("Predict embeddings, logits or dump helper tensors. Run without `outputs` to list valid output keys.")
    parser.add_argument("data", help="Path to dataset root")
    parser.add_argument("--dataset", help="Name of the dataset. If not provided, list available datasets.")
    parser.add_argument("--config", help="Path to training config")
    parser.add_argument("--checkpoint", help="Path to initial checkpoint")
    parser.add_argument("--outputs", help="A list of tensor_key:filename with output files. If not provided, list valid keys.", nargs="+")
    parser.add_argument("--augment-train", help="Augment training set", action="store_true")
    parser.add_argument("--num-batches", help="Limit the number of batches to evaluate", type=int)
    return parser.parse_args()


def init_runner(root, args):
    is_train = args.dataset == "train"
    config = read_config(args.config) if args.config is not None else {}
    config["stage_resume"] = None
    patch = {
        "dataset_params": {
            "samples_per_class": None,
            "shuffle_train": False
        }
    }
    config = update_config(config, patch)
    runner = NoLogRunner(root=root, data_root=args.data, config=config)
    runner._stage = runner.STAGE_TRAIN if is_train else runner.STAGE_TEST
    runner.stage_key = runner.stages[-1]
    runner._run_event("on_experiment_start")
    runner._run_event("on_stage_start")
    runner._run_event("on_epoch_start")
    runner.loader_key = args.dataset
    loaders = runner.datasets.get_loaders(train=is_train, augment_train=args.augment_train)
    if (args.dataset is None) or (args.dataset not in loaders):
        loaders.update(runner.datasets.get_loaders(train=~is_train))
        raise ValueError("Available datasets are: {}.".format(args.dataset, list(loaders)))
    runner.loader = loaders[args.dataset]
    runner._run_event("on_loader_start")
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")["model_model_state_dict"]
        runner.model["model"].load_state_dict(checkpoint)
    runner.engine.sync_device(tensor_or_module=runner.model)
    maybe_recursive_call(runner.model, "train", mode=False)
    return runner


def use_grad(mode):
    if mode:
        return torch.enable_grad()
    else:
        return torch.no_grad()


def model_hash(runner):
    hash = 0
    for p in runner.model["model"].parameters():
        hash += p.sum().item()
    return hash


def main(args):
    is_train = args.dataset == "train"
    output_files = {}
    for output in args.outputs or []:
        key, filename = output.split(":")
        if key in output_files:
            raise ValueError("Multiple files for {}".format(key))
        output_files[key] = filename
    need_gradients = "gradnorms" in output_files
    with tempfile.TemporaryDirectory() as root:
        runner = init_runner(root, args)
        hash_before = model_hash(runner)
        runner.callbacks.pop("optimizer", None)
        outputs = defaultdict(list)
        key_suffix = runner.get_loader_suffix()
        with use_grad(need_gradients):
            for i, batch in tqdm(enumerate(runner.loader)):
                if (args.num_batches is not None) and (i >= args.num_batches):
                    break
                runner.batch = batch
                runner._run_batch()
                results = runner.batch_metrics
                results.update(runner.batch)
                distribution = runner.model["model"].distribution
                batch_size = len(runner.batch["labels" + key_suffix])
                if distribution.has_confidences:
                    for suffix in ["", "1", "2"]:
                        if "embeddings" + suffix + key_suffix in results:
                            results["confidences" + suffix + key_suffix] = distribution.confidences(results["embeddings" + suffix + key_suffix])
                if not output_files:
                    valid_keys = [key for key, value in results.items()
                        if (isinstance(value, torch.Tensor) and
                            (value.ndim > 0) and
                            (len(value) == batch_size))]
                    valid_keys.append("gradnorms")
                    print("Valid keys: {}".format(valid_keys))
                    return
                for key in output_files:
                    if key == "gradnorms":
                        loss = results["loss" + key_suffix]
                        runner.engine.backward_loss(loss, runner.model, runner.optimizer)
                        gradient_norm = torch.nn.utils.clip_grad_norm_(runner.model["model"].parameters(), 1e6)
                        outputs[key].append(torch.full((batch_size,), gradient_norm.item()))
                        runner.engine.zero_grad(loss, runner.model["model"], runner.optimizer)
                    elif key in results:
                        outputs[key].append(results[key].detach().cpu().numpy())
                    elif key + key_suffix in results:
                        outputs[key].append(results[key + key_suffix].detach().cpu().numpy())
                    else:
                        raise KeyError("Unknown key: {}".format(key))
        outputs = {key: np.concatenate(values, 0) for key, values in outputs.items()}
        assert abs(model_hash(runner) - hash_before) < 1e-6, "Model changed"
    for key, filename in output_files.items():
        value = outputs[key]
        print("Dump {} with shape {} to {}".format(key, value.shape, filename))
        np.save(filename, value)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
