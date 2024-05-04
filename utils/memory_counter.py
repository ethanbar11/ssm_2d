import logging

import torch.nn as nn
import re

def remove_numbers(s):
    return re.sub(r'\.\d+', '', s)


import torch


def get_total_memory_usage(model):
    # Memory for model parameters
    param_memory = sum(param.nelement() * param.element_size() for param in model.parameters())

    # Memory for gradients
    # grad_memory = sum(param.nelement() * param.element_size() for param in model.parameters() if param.requires_grad)

    # Convert to megabytes (1 MB = 1024 * 1024 bytes)
    total_memory_mb = param_memory / (1024 * 1024)

    return total_memory_mb


def get_recursive_memory_consumption(model, memory, current_path):
    for name, submodule in model.named_modules():
        if isinstance(submodule, nn.Module) and name != '':
            total_params = get_total_memory_usage(submodule)  # sum(p.numel() for p in submodule.parameters())
            full_name = f'{current_path}.{name}' if current_path != '' else f'{name}'
            name_for_memory = remove_numbers(full_name)
            if name_for_memory not in memory:
                memory[name_for_memory] = 0
            memory[name_for_memory] += total_params
            # get_recursive_memory_consumption(submodule, memory, full_name)


def print_model_memory_consumption(model,logger):
    memory = {}
    get_recursive_memory_consumption(model, memory, current_path='')
    sorted_memory_consumption = sorted(memory.items(), key=lambda x: x[1], reverse=True)
    total_memory = get_total_memory_usage(model)

    # Printing the top 10 items
    logger.debug('Parameter memory consumption ordered from large to small:')
    for item, value in sorted_memory_consumption:
        logger.debug(f"{item}: {value : .2f}mb, ({value * 100.0 / total_memory :.2f}%)")
