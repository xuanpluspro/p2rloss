# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

from checkpointing import (load_training_state, save_training_state, find_latest)


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    return load_training_state(config, model, optimizer, lr_scheduler, logger)


def save_checkpoint(config, epoch, model, max_accuracy, logger, metrics=None,
                    epoch_index=None, optimizer=None, lr_scheduler=None, best_epoch=-1):
    return save_training_state(config, epoch, model, max_accuracy, logger,
                               optimizer=optimizer, lr_scheduler=lr_scheduler,
                               epoch_index=epoch_index, metrics=metrics, best_epoch=best_epoch)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    return find_latest(output_dir)


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def plot_curve(label, epo, data, savepath):
    import os
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    
    fig = plt.figure()
    plt.title(label)
    plt.plot(epo, data)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.grid(True)
    plt.savefig(savepath)
    plt.close(fig)

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True