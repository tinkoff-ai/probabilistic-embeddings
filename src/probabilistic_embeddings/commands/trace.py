import torch
import tempfile

from ..runner import Runner
from .common import setup


def trace_embedder(args):
    """Trace model embedder to output checkpoint."""
    setup()
    if args.checkpoint is None:
        raise ValueError("Input checkpoint path must be provided")
    if args.trace_output is None:
        raise ValueError("Output checkpoint path must be provided")
    with tempfile.TemporaryDirectory() as root:
        runner = Runner(root, args.data,
                        config=args.config, logger="tensorboard",
                        initial_checkpoint=args.checkpoint,
                        no_strict_init=args.no_strict_init)
        runner.init_stage(runner.STAGE_TEST)
        model = runner.get_model(runner.STAGE_TEST)["embedder"]
        model.eval()
        loader = next(iter(runner.get_loaders(runner.STAGE_TEST).values()))
        batch = next(iter(loader))[0]
        if not isinstance(batch, torch.Tensor):
            # Pairs dataset. Take first set of images.
            batch = batch[0]
        if torch.cuda.is_available():
            batch = batch.cuda()
            model = model.cuda()
        checkpoint = torch.jit.trace(model, batch)
        torch.jit.save(checkpoint, args.trace_output)
