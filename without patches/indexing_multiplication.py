import torch
import torch.nn as nn
from einops import rearrange, repeat, einsum
import time


def mat_mul(mat, vec):
    return einsum(mat, vec, 'b d, d l -> b l')


def mat_indexing(mat, vec):
    return vec[mat]


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    options_amount = 2 ** 6
    N = 2 ** 10
    mat_length = 2 ** 20
    vec = torch.randn(options_amount, options_amount).to(device)
    mat = torch.randint(options_amount, (mat_length, 1)).to(device).squeeze(1)
    mat_one_hot = torch.nn.functional.one_hot(mat, options_amount).float().to(device)
    times = [time.time()]
    output = mat_indexing(mat, vec)
    times.append(time.time())
    output2 = mat_mul(mat_one_hot, vec)
    times.append(time.time())
    assert torch.allclose(output, output2)
    print('indexing:', times[1] - times[0])
    print('multiplying', times[2] - times[1])
    pass
