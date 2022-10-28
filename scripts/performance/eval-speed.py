import argparse
import os
import tempfile
import time

import numpy as np
import torch
from tqdm import tqdm

from probabilistic_embeddings.runner import Runner


def parse_arguments():
    parser = argparse.ArgumentParser("Eval train/inference speed.")
    parser.add_argument("data", help="Path to dataset root")
    parser.add_argument("--config", help="Path to training config")
    return parser.parse_args()


def measure(title, fn, n=50):  # TODO: 50
    evals = []
    for _ in range(n):
        init = time.time()
        fn()
        evals.append(time.time() - init)
    print("{}: {:.2f} +- {:.2f} ms".format(title, 1000 * np.mean(evals), 1000 * np.std(evals)))


def main(args):
    with tempfile.TemporaryDirectory() as root:
        runner = Runner(root=root, data_root=args.data, config=args.config)
        runner.engine = runner.get_engine()

        runner._stage = runner.STAGE_TRAIN
        runner.stage_key = runner.stages[-1]
        runner._run_event("on_stage_start")
        runner.callbacks = {k: v for k, v in runner.callbacks.items()
                            if k in ["criterion", "optimizer"]}
        loader = runner.datasets.get_loaders(train=True)["train"]
        images, labels = next(iter(loader))
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        runner.model["embedder"].train(True)
        runner.model["embedder"](images).mean().backward()
        print("Memory usage (MB):", torch.cuda.max_memory_allocated() / 1024 ** 2)
        print("Memory usage per request (MB):", torch.cuda.max_memory_allocated() / 1024 ** 2 / len(images))

        def train_cnn():
            runner.model["embedder"](images).mean().backward()
        measure("Train CNN", train_cnn)

        def train_batch():
            runner.batch = (images, labels)
            runner.is_train_loader = True
            runner.loader_key = "train"
            runner._run_batch()
        #measure("Train full", train_batch)

        runner._stage = runner.STAGE_TEST
        runner.stage_key = runner.stages[0]
        runner._run_event("on_stage_start")
        runner.callbacks = {k: v for k, v in runner.callbacks.items()
                            if k in ["criterion"]}
        loader = runner.datasets.get_loaders(train=False)["valid"]
        images, labels = next(iter(loader))
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        def valid_cnn():
            with torch.no_grad():
                runner.model["embedder"](images.detach()).mean()
        measure("Inference CNN", valid_cnn)

        def valid_batch():
            runner.batch = (images, labels)
            runner.is_train_loader = False
            runner.loader_key = "valid"
            runner._run_batch()
        measure("Inference full", valid_batch)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
