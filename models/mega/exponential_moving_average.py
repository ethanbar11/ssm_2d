import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiHeadEMA(nn.Module):
    """Exponential Moving Average Layer.

    See "" for more details.
    """

    def __init__(
        self,
        embed_dim,
        ndim=2,
        bidirectional=False,
        truncation=None,
        shift=True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.ndim = ndim
        self.bidirectional = bidirectional
        self.truncation = truncation
        self.shift = shift
        self.scale = math.sqrt(1.0 / self.ndim)

        kernel_dim = 2 * embed_dim if self.bidirectional else embed_dim
        self.alpha = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.delta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.beta = nn.Parameter(torch.Tensor(kernel_dim, ndim, 1))
        self.gamma = nn.Parameter(torch.Tensor(kernel_dim, ndim))
        self.omega = nn.Parameter(torch.Tensor(embed_dim))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            nn.init.normal_(self.alpha, mean=0.0, std=0.2)
            nn.init.normal_(self.delta, mean=0.0, std=0.2)
            # beta [1, -1, 1, -1, ...] seems more stable.
            val = torch.ones(self.ndim, 1)
            if self.ndim > 1:
                idx = torch.tensor(list(range(1, self.ndim, 2)))
                val.index_fill_(0, idx, -1.0)
            self.beta.normal_(mean=0.0, std=0.02).add_(val)
            # gamma & omega
            nn.init.normal_(self.gamma, mean=0.0, std=1.0)
            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def coeffs(self):
        # D x N x 1
        alpha = torch.sigmoid(self.alpha)
        delta = torch.sigmoid(self.delta)
        q = 1.0 - alpha * delta
        p = alpha * self.beta
        return p, q

    def compute_kernel(self, length: int):
        # D x N x 1
        p, q = self.coeffs()
        # D x N x L
        vander = torch.arange(length).to(q).view(1, 1, length) * torch.log(q)
        kernel = p * torch.exp(vander)
        # D x L
        return torch.einsum('dnl,dn->dl', kernel, self.gamma * self.scale)

    def kernel(self, length: int):
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        return self.compute_kernel(kernel_size)

    def forward(self, x) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # L x B x D
        residual = x * self.omega

        # L x B x D -> B x D x L
        x = x.permute(1, 2, 0)
        # D x L
        k = self.kernel(seq_len)
        kernel_size = k.size(1)
        fft_len = seq_len + kernel_size
        if self.bidirectional:
            k1, k2 = torch.split(k, [self.embed_dim, self.embed_dim], dim=0)
            # D x K+L
            if self.shift:
                k = F.pad(k1, (0, seq_len)) + F.pad(k2.flip(-1), (seq_len, 0))
            else:
                k = F.pad(k1, (0, seq_len)) + F.pad(k2[:, :1], (0, fft_len - 1)) + F.pad(k2[:, 1:].flip(-1), (seq_len + 1, 0))

        k_f = torch.fft.rfft(k.float(), n=fft_len)
        x_f = torch.fft.rfft(x.float(), n=fft_len)
        # B x D x L
        out = torch.fft.irfft(x_f * k_f, n=fft_len)[..., :seq_len]
        out = out.type_as(x)
        # B x D x L -> L x B x D
        out = F.silu(out.permute(2, 0, 1) + residual)
        return out

    def extra_repr(self) -> str:
        return 'edim={}, ndim={}, bidirectional={}, trunction={}, shift={}'.format(self.embed_dim, self.ndim, self.bidirectional,
                                                                                   self.truncation, self.shift)
