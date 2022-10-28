import argparse
import copy
import os
import sys
import tempfile

from probabilistic_embeddings.runner import Runner
from probabilistic_embeddings.config import read_config, write_config, prepare_config, update_config


def parse_arguments():
    parser = argparse.ArgumentParser("Print configs for all training stages.")
    parser.add_argument("-c", "--config", help="Path to training config.")
    parser.add_argument("--hopt", help="Print config for hyper-parameter tuning.", action="store_true")
    return parser.parse_args()


def main(args):
    with tempfile.TemporaryDirectory() as root:
        config = args.config
        if args.hopt:
            config = prepare_config(Runner, args.config)
            config = update_config(config, config["hopt_params"])
        runner = Runner(root=root, data_root=None, config=config)
        runner._stage = runner.STAGE_TRAIN
        for stage in runner.stages:
            config = runner.get_stage_config(stage)
            print("=== STAGE {} ===".format(stage))
            write_config(config, sys.stdout)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
