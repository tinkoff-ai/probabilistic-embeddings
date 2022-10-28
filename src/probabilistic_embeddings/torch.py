import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

try:
    import torch.cuda.amp
    USE_AMP = True
except ImportError:
    USE_AMP = False


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


def enable_amp(enable=True):
    """Context to disable AMP."""
    if USE_AMP and enable:
        return torch.cuda.amp.autocast()
    else:
        return nullcontext()


def disable_amp(disable=True):
    """Context to disable AMP."""
    if USE_AMP and disable:
        return torch.cuda.amp.autocast(enabled=False)
    else:
        return nullcontext()


@contextmanager
def tmp_seed(seed):
    """Centext manager for temporary random seed (random and Numpy modules)."""
    state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield None
    finally:
        random.setstate(state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def get_base_module(module):
    """Returns base torch module from wappers like DP and DDP."""
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        module = module.module
    return module


def freeze(model, freeze=True):
    """Freeze or unfreeze all parameters of the model."""
    for p in model.parameters():
        p.requires_grad = not freeze


def freeze_bn(model, freeze=True):
    """Freeze or unfreeze batchnorm parameters."""
    if isinstance(model, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        for p in model.parameters():
            p.requires_grad = not freeze
    for child in model.children():
        freeze_bn(child, freeze=freeze)


def eval_bn(model, eval=True):
    """Change evaluation mode for model batchnorms."""
    if isinstance(model, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        model.train(not eval)
    for child in model.children():
        eval_bn(child, eval=eval)


def try_cuda(m):
    if torch.cuda.is_available():
        return m.cuda()
    return m
