import torch


def dct(x):
    """
    DCT Type 1: DFT of the function modified to be even
    """
    # TODO: implement the DCT directly
    # Construct an even fn by mirroring x to length 2 * (N - 1)
    x = torch.cat([x, torch.flip(x, dims=(-1,))[..., 1:-1]], dim=-1)
    y = torch.fft.rfft(x)
    return y.real

def idct(y):
    """
    Inverse DCT Type 1: IDFT of the function
    """
    # TODO: implement the inverse DCT directly
    x = torch.fft.irfft(y)[..., :y.shape[-1]]
    return x.real

def chebyshev_nodes(a, b, L):
    """ Compute Chebyshev nodes for given interval. """
    nodes = torch.cos(torch.pi * (2 * torch.arange(L, 0, -1) - 1) / (2 * L))
    nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    return nodes

def chebyshev_quadrature(g, drop_frac=0.):
    """
    Quadrature using Chebyshev interpolation of the integrand g.

    Calculates the integral \int_{0}^{z_i} g(tau) dtau using Chebyshev interpolation
    for all integration limits z_i \in {z_0, z_1, ..., z_L}.

    Implemented by running a DCT on the integrand g, transforming the coefficients
    and then performing an inverse DCT to recover the result.
    """
    g = g.real
    L = g.shape[-1]
    shape = list(g.shape)
    shape[-1] += 1
    coeffs = torch.zeros(*shape, device=g.device)
    coeffs[..., :-1] = dct(g)

    if drop_frac > 0:
        coeffs[..., int(len(coeffs[-1]) * (1 - drop_frac)): ] = 0

    coeffs_out = torch.zeros(*g.shape, device=g.device)
    coeffs_out[..., 1:] = (coeffs[..., :-2] - coeffs[..., 2:]) /  (2 * torch.arange(1, L, device=g.device))

    partials = idct(coeffs_out)
    y = partials - partials[..., :1]
    return - y / 2.
