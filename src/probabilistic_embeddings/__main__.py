import argparse
from .commands import train, test, cval, evaluate, hopt, trace_embedder


def parse_arguments():
    parser = argparse.ArgumentParser("Train classification model")
    parser.add_argument("cmd", help="Command to run (train, test, cval, evaluate, evaluate-cval, hopt or hopt-cval)",
                        choices=["train", "test", "cval", "evaluate", "evaluate-cval", "hopt", "hopt-cval", "trace-embedder"])
    parser.add_argument("data", help="Path to dataset root")
    parser.add_argument("--logger", help="Logger to use. Default is tensorboard. "
                                         "To use wandb, type 'wandb:<project-name>:<experiment-name>[:<group-name>]'")
    parser.add_argument("--config", help="Path to training config")
    parser.add_argument("--train-root", help="Training directory")
    parser.add_argument("--checkpoint", help="Path to initial checkpoint. Can contain mask with '{seed}' and '{fold}'.")
    parser.add_argument("--no-strict-init", help="Skip checkpoint mismatch errors", action="store_true")
    parser.add_argument("--trace-output", help="Path to traced model (for trace-embedder command).")
    parser.add_argument("--sweep-id", help="Connect to existing sweep id when hopt is called")
    parser.add_argument("--from-stage", help="Start training from specified stage. Use -1 to skip training.", type=int)
    parser.add_argument("--from-seed", help="Resume evaluation from given seed", type=int, default=0)
    parser.add_argument("--clean", help="Clean working directory after training (produce WandB logs only)",
                        action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    commands = {
        "train": train,
        "test": test,
        "cval": cval,
        "evaluate": lambda args: evaluate(args, run_cval=False),
        "evaluate-cval": lambda args: evaluate(args, run_cval=True),
        "hopt": lambda args: hopt(args, run_cval=False),
        "hopt-cval": lambda args: hopt(args, run_cval=True),
        "trace-embedder": trace_embedder
    }
    commands[args.cmd](args)
