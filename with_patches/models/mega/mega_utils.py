# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


try:
    from amp_C import multi_tensor_l2norm
    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False


def multi_tensor_total_norm(grads, chunk_size=2048*32) -> torch.Tensor:
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(chunk_size, has_inf, [cur_device_grads], False)
                norms.append(norm[0])
        else:
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None, mode='total'):
    assert mode in ['total', 'each']
    if mode == 'total':
        return clip_total_grad_norm_(params, max_norm, aggregate_norm_fn=aggregate_norm_fn)
    else:
        return clip_param_grad_norm_(params, max_norm, aggregate_norm_fn=aggregate_norm_fn)


def clip_total_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = multi_tensor_total_norm(grads)
        else:
            if torch.cuda.is_available():
                warnings.warn(
                    "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                    "you may get better performance by installing NVIDIA's apex library"
                )
            total_norm = torch.norm(
                torch.stack([torch.norm(g, p=2, dtype=torch.float32) for g in grads])
            )

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads:
            g.mul_(clip_coef)
    return total_norm


def clip_param_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in filter(lambda p: p.grad is not None, params)]
    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.)
        else:
            return torch.tensor(0.)

    if max_norm > 0:
        max_norm = float(max_norm)

    grad_norms = []
    for grad in grads:
        grad_norm = torch.norm(grad, p=2, dtype=torch.float32)
        if aggregate_norm_fn is not None:
            grad_norm = aggregate_norm_fn(grad_norm)
        grad_norms.append(grad_norm)
        if max_norm > 0:
            clip_coef = (max_norm / (grad_norm + 1e-6)).clamp_(max=1)
            grad.mul_(clip_coef)

    return torch.max(torch.stack(grad_norms))


def relu2(x):
    return torch.square(F.relu(x))


def laplace(x, mu=0.707107, sigma=0.282095):
    x = (x - mu).div(sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + torch.erf(x))


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == 'silu':
        return F.silu
    elif activation == "tanh":
        return torch.tanh
    elif activation == 'sin':
        return torch.sin
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "tanh",
        "sin",
        "linear",
        "silu",
    ]
