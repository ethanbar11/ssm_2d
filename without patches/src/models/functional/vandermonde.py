"""pykeops implementations of the Vandermonde matrix multiplication kernel used in the S4D kernel."""
if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import torch

from einops import rearrange, repeat
from opt_einsum import contract


try:
    import pykeops
    from pykeops.torch import LazyTensor, Genred
except:
    pass

_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [tensor.view((1,)*(max_dim-len(tensor.shape))+tensor.shape) for tensor in tensors]
    return tensors

def _c2r(x): return torch.view_as_real(x)
def _r2c(x): return torch.view_as_complex(x)

def vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    if conj:
        x = _conj(x)
        v = _conj(v)
    vandermonde_matrix = x.unsqueeze(-1) ** torch.arange(L).to(x) # (... N L)
    vandermonde_prod = torch.sum(v.unsqueeze(-1) * vandermonde_matrix, dim=-2) # (... L)
    return vandermonde_prod

def log_vandermonde_naive(v, x, L, conj=True):
    """
    v: (..., N)
    x: (..., N)
    returns: (..., L) \sum v x^l
    """
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... n, ... n l -> ... l', v, vandermonde_matrix) # (... L)
    if conj:
        return 2*vandermonde_prod.real
    else:
        return vandermonde_prod

def log_vandermonde_lazy(v, x, L, conj=True):
    if conj:
        v = _conj(v)
        x = _conj(x)
    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v_l = LazyTensor(rearrange(v, '... N -> ... N 1 1'))
    x_l = LazyTensor(rearrange(x, '... N -> ... N 1 1'))
    l_l = LazyTensor(rearrange(l, '... L -> ... 1 L 1'))
    # exp
    vand = (x_l * l_l).exp()
    s = (v_l*vand).sum(dim=len(v_l.shape)-2)
    return s.squeeze(-1)

def log_vandermonde(v, x, L, conj=True):
    expr = 'ComplexMult(v, ComplexExp(ComplexMult(x, l)))'
    vandermonde_mult = Genred(
        expr,
        [
            'v = Vj(2)',
            'x = Vj(2)',
            'l = Vi(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    l = torch.arange(L).to(x)
    v, x, l = _broadcast_dims(v, x, l)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(v, x, l, backend='GPU')
    if conj:
        return 2*_r2c(r).real
    else:
        return _r2c(r)

def log_vandermonde_transpose_naive(u, v, x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    vandermonde_prod = contract('... l, ... n, ... n l -> ... n', u.to(x), v.to(x), vandermonde_matrix) # (... L)
    return vandermonde_prod

def log_vandermonde_transpose(u, v, x, L):
    """
    u: ... H L
    v: ... H N
    x: ... H N
    Returns: ... H N

    V = Vandermonde(a, L) : (H N L)
    contract_L(V * u * v)
    """
    expr = 'ComplexMult(ComplexMult(v, u), ComplexExp(ComplexMult(x, l)))'
    vandermonde_mult = Genred(
        expr,
        [
            'u = Vj(2)',
            'v = Vi(2)',
            'x = Vi(2)',
            'l = Vj(2)',
        ],
        reduction_op='Sum',
        axis=1,
    )

    l = torch.arange(L).to(x)
    u, v, x, l = _broadcast_dims(u, v, x, l)
    u = _c2r(u)
    v = _c2r(v)
    x = _c2r(x)
    l = _c2r(l)

    r = vandermonde_mult(u, v, x, l, backend='GPU')
    return _r2c(r)

def _log_vandermonde_matmul(x, L):
    vandermonde_matrix = torch.exp(x.unsqueeze(-1) * torch.arange(L).to(x)) # (... N L)
    return vandermonde_matrix

def log_vandermonde_matmul(v, K):
    prod = contract('...n, ...nl -> ...l', v, K)
    return 2*prod.real


def data(B, N, L, conj=True):
    v = torch.randn(B, N//2, dtype=torch.cfloat)
    x = 0.001 * torch.rand(B, N//2) + 1j * N * torch.rand(B, N//2)

    if not conj:
        v = _conj(v)
        x = _conj(x)

    v, x = utils.convert_data(v, x)
    return v, x

def test_vandermonde():
    B, N, L = 2, 4, 8
    # B, N, L = 64, 64, 1024
    v, x = data(B, N, L)

    # Test correctness
    utils.compare_outputs(
        log_vandermonde_naive(v, x, L, conj=True),
        log_vandermonde_lazy(v, x, L),
        log_vandermonde(v, x, L),
        full=False,
        relative=True,
    )

    utils.benchmark(log_vandermonde_naive, v, x, L, repeat=100, memory=True, desc='cauchy conj slow')
    utils.benchmark(log_vandermonde_lazy, v, x, L, repeat=100, memory=True, desc='cauchy conj slow')
    utils.benchmark(log_vandermonde, v, x, L, repeat=100, memory=True, desc='cauchy conj pykeops')

def profile_mults():
    B, H, N, L = 1, 256, 64, 16384
    # B, H, N, L = 1, 256, 64, 1024
    v, x = data(H, N, L, conj=True)
    u, = utils.convert_data(torch.randn(B, H, L, dtype=torch.cfloat))

    # Test correctness
    utils.compare_outputs(
        log_vandermonde_naive(v, x, L, conj=True),
        log_vandermonde_lazy(v, x, L, conj=True),
        log_vandermonde(v, x, L, conj=True),
        full=False,
        relative=True,
    )


    # Measure speed and memory
    repeat = 1000
    utils.benchmark(log_vandermonde_naive, v, x, L, repeat=repeat, memory=True, desc='naive vandermonde')
    utils.benchmark(log_vandermonde_lazy, v, x, L, repeat=repeat, memory=True, desc='fast lazy vandermonde')
    utils.benchmark(log_vandermonde, v, x, L, repeat=repeat, memory=True, desc='fast vandermonde')

    utils.benchmark(log_vandermonde_transpose_naive, u, v, x, L, repeat=repeat, memory=True, desc='naive vandermonde transpose')
    utils.benchmark(log_vandermonde_transpose, u, v, x, L, repeat=repeat, memory=True, desc='fast vandermonde transpose')

def profile_matmul():
    H, N, L = 256, 256, 1024
    B = 1
    # B, N, L = 256, 64, 1024
    v, x = data(H, N, L, conj=True)
    v = repeat(v, 'h n -> b h n', b=B)

    K = _log_vandermonde_matmul(x, L).contiguous()
    print(K.shape)

    # Test correctness
    utils.compare_outputs(
        log_vandermonde_naive(v, x, L, conj=True),
        log_vandermonde_lazy(v, x, L, conj=True),
        log_vandermonde(v, x, L, conj=True),
        log_vandermonde_matmul(v, K),
        full=False,
        relative=True,
    )


    # Measure speed and memory
    T = 10
    utils.benchmark(log_vandermonde_naive, v, x, L, repeat=T, memory=True, desc='naive vandermonde')
    utils.benchmark(log_vandermonde_lazy, v, x, L, repeat=T, memory=True, desc='fast vandermonde LazyTensor')
    utils.benchmark(log_vandermonde, v, x, L, repeat=T, memory=True, desc='fast vandermonde Genred')
    utils.benchmark(_log_vandermonde_matmul, x, L, repeat=T, memory=True, desc='vandermonde matrix')
    utils.benchmark(log_vandermonde_matmul, v, K, repeat=T, memory=True, desc='vandermonde matmul')

if __name__ == '__main__':
    from benchmark import utils
    device = 'cuda'
    # test_vandermonde()
    profile_mults()
    # profile_matmul()
