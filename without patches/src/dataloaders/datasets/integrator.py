import math
import torch

def integrator_data(
    B, # Batch size
    L, # Sequence length
    K, # Number of mixture components
    max_ampl,
    max_freq,
):
    # Generate random amplitudes and frequencies
    t = torch.linspace(0.0, 1.0, L) # Sample L time steps
    ampls = torch.rand(B, K) * 2*max_ampl - max_ampl # Geometrically spaced between 1 and max_ampl
    freqs = torch.exp(torch.rand(B, K) * math.log(max_freq)) # Geometrically spaced between 1 and max_freq
    freqs = 2*math.pi * freqs

    # Generate the function \sum_i \alpha_i * sin(2\pi \beta_i t)
    sins = ampls.unsqueeze(-1) * torch.sin(freqs.unsqueeze(-1) * t) # (B, K, L)
    f = torch.sum(sins, dim=-2) # (B, L)

    # Normalizing factor for unit variance
    Z = torch.mean(f**2)
    f = f / torch.sqrt(Z)

    # Calculate targets: integral of f = \sum -\alpha_i \cos(2\pi\beta_i t) / (2\pi\beta_i)
    coss = - ampls.unsqueeze(-1) / freqs.unsqueeze(-1) * torch.cos(freqs.unsqueeze(-1) * t) # (B, K, L)
    g = torch.sum(coss, dim=-2) # (B, L)
    g -= g[:, :1] # start at 0
    # Normalizing factor for unit variance
    Z = torch.mean(g**2)
    g = g / torch.sqrt(Z)

    return f, g
