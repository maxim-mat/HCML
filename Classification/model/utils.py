import torch.nn as nn
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
import numpy as np


def get_norm(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == 'BatchNorm':
        return nn.BatchNorm2d(num_features, eps=eps)
    elif norm_type == "GroupNorm":
        return nn.GroupNorm(num_groups, num_features, eps=eps)
    elif norm_type == "InstanceNorm":
        return nn.InstanceNorm2d(num_features, eps=eps,
                                 affine=True, track_running_stats=True)
    else:
        raise Exception('Unknown Norm Function : {}'.format(norm_type))


def get_optimizer(params, cfg):
    if cfg.optimizer == 'SGD':
        return SGD(params, lr=cfg.lr, momentum=cfg.momentum,
                   weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adadelta':
        return Adadelta(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adagrad':
        return Adagrad(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adam':
        return Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'RMSprop':
        return RMSprop(params, lr=cfg.lr, momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay)
    else:
        raise Exception('Unknown optimizer : {}'.format(cfg.optimizer))


def tensor2numpy(input_tensor):
    # device cuda Tensor to host numpy
    return input_tensor.cpu().detach().numpy()


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)
