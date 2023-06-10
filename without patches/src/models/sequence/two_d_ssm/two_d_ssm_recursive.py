# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from functools import partial

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Dict, Optional, Tuple
from einops import rearrange, einsum, repeat
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.s4.s4 import Activation, DropoutNd
from src.models.sequence.two_d_ssm.ssm_coefficient import CoeffCalculator

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def plot_heatmap(x, title):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(x.cpu().detach().numpy()).set_title(title)
    plt.show()



class OptimModule(nn.Module):
    """ Interface for Module that allows registering buffers/parameters with configurable optimizer hyperparameters """

    def register(self, name, tensor, lr=None, wd=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {}
            if lr is not None: optim["lr"] = lr
            if wd is not None: optim["weight_decay"] = wd
            setattr(getattr(self, name), "_optim", optim)


class TwoDimensionalSSM(OptimModule):
    def __init__(
            self,
            embed_dim=32,
            ndim=2,
            effective_L=32,  #
            force_coeff_calc=False,
            n_ssm=2,
            complex_ssm=False,
            directions_amount=4,
            coeff=None,
            transpose=False,
            dropout=None,
            activation=None,
            linear_layer=True,
            **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_output = self.embed_dim
        self.ndim = ndim
        self.n_ssm = embed_dim
        self.is_complex = complex_ssm
        self.transpose = transpose
        self.directions_amount = directions_amount
        self.repeat = self.embed_dim // self.n_ssm

        self.scale = math.sqrt(1.0 / self.ndim)
        self.kernel_dim = directions_amount * self.n_ssm

        # TODO: The effective_L is the length of the side of the square that we are using
        # to actually create the kernel. After that, padding.
        self.one_side_length = effective_L
        self.coeff_matrix_row_amount = 2 * self.one_side_length ** 3
        self.coeff_calc = CoeffCalculator(self.one_side_length)
        self.coeff_calc.calc_coeffs_lazy(force=force_coeff_calc)
        self.directions = ['horizontal', 'vertical']
        self.symbols = ['A_1', 'A_2', 'A_3', 'A_4', 'B']
        for key, inner_dic in self.coeff_calc.matrices.items():
            for symbol, matrix in inner_dic.items():
                if self.is_complex:
                    matrix = matrix.type(torch.complex64)
                name = f"{key}_{symbol}"
                self.register_buffer(name, matrix)

        # H x N
        if self.is_complex:
            last_dim = 2

            self.A = {
                'A_1': torch.Tensor(self.kernel_dim, self.ndim, last_dim),
                'A_2': torch.Tensor(self.kernel_dim, self.ndim, last_dim),
                'A_3': torch.Tensor(self.kernel_dim, self.ndim, last_dim),
                'A_4': torch.Tensor(self.kernel_dim, self.ndim, last_dim),
            }
            # self.A = nn.ParameterDict(self.A)
            self.B_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, last_dim))
            self.B_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, last_dim))
            # D x N
            self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, last_dim))
            self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, last_dim))
        else:
            self.register("A_1", torch.Tensor(self.kernel_dim, self.ndim), wd=0.0)
            self.register("A_2", torch.Tensor(self.kernel_dim, self.ndim), wd=0.0)
            self.register("A_3", torch.Tensor(self.kernel_dim, self.ndim), wd=0.0)
            self.register("A_4", torch.Tensor(self.kernel_dim, self.ndim), wd=0.0)
            # Add all the A's to one dictionary
            self.A = {
                'A_1': self.A_1,
                'A_2': self.A_2,
                'A_3': self.A_3,
                'A_4': self.A_4,
            }
            self.register('B_1', torch.Tensor(self.kernel_dim, self.ndim), wd=0.0)
            self.register("B_2", torch.Tensor(self.kernel_dim, self.ndim), wd=0.0)
            # D x N
            self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))

        # Registering all params to have 0 weight decay
        self.activation = Activation(activation)
        # self.dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        dropout_fn = partial(DropoutNd, transposed=not self.transpose)
        self.dropout = dropout_fn(dropout) if dropout is not None else nn.Identity()

        if linear_layer:
            self.mixing_linear_layer = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.mixing_linear_layer = nn.Identity()
        self._kernel = None
        self._coeffs = None
        self.first_time = True
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            for symbol, tensor in self.A.items():
                nn.init.normal_(tensor, mean=0, std=0.2)
            nn.init.normal_(self.B_1, mean=0.0, std=0.2)
            nn.init.normal_(self.B_2, mean=0.0, std=0.2)
            # TODO: After expanding to n_dim>1 , checkout what's being done with beta in EMA

            nn.init.normal_(self.C_1, mean=0.0, std=1.0)
            nn.init.normal_(self.C_2, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        A = {}
        for symbol, tensor in self.A.items():
            A[symbol] = torch.sigmoid(tensor)
            if self.is_complex:
                A[symbol] = _r2c(A[symbol])
        B1 = torch.sigmoid(self.B_1)
        B2 = torch.sigmoid(self.B_2)
        if self.is_complex:
            B1 = _r2c(B1)
            B2 = _r2c(B2)
        return A, B1, B2

    def compute_x_matrix(self, kernel_dim):
        # H x N each
        A, B1, B2 = self._calc_coeffs()
        power_dim = self.one_side_length * 2
        # l x l  D x N
        A_powers = {}
        for symbol, tensor in A.items():
            A_powers[symbol] = torch.exp(
                einsum(torch.arange(power_dim).to(tensor.device),
                       torch.log(tensor),
                       'l , h n-> l h n'))
        B = torch.stack([B1, B2], dim=0)
        outputs = {}
        for direction in self.directions:
            # Should be sized R x H x N
            outputs[direction] = None
        for direction in self.directions:
            output = outputs[direction]
            for symbol in self.symbols:
                vec = B if symbol == 'B' else A_powers[symbol]
                name = f"{direction}_{symbol}"
                matrix = getattr(self, name)
                current_calculation = einsum(matrix, vec, 'R V, V h n -> R h n')

                if output is None:
                    output = current_calculation
                else:
                    output = output * current_calculation
            outputs[direction] = output
        for direction, matrix in outputs.items():
            outputs[direction] = rearrange(matrix, '(r1 r2) h n-> r1 r2 h n',
                                           r1=self.one_side_length ** 2,
                                           r2=self.coeff_matrix_row_amount // (self.one_side_length ** 2))
            # Sum over the second dimension
            outputs[direction] = torch.sum(outputs[direction], dim=1)
        return outputs

    def _compute_kernel(self, length: int):
        self._kernel = None
        A, B_1, B_2 = self._calc_coeffs()
        # l x l x D x N
        outputs = self.compute_x_matrix(self.one_side_length)
        # L x L x D x N

        # L x L x H
        if self.is_complex:
            C_1 = _r2c(self.C_1)
            C_2 = _r2c(self.C_2)
        else:
            C_1 = self.C_1
            C_2 = self.C_2
        output_horizontal = einsum(outputs['horizontal'], C_1 * self.scale, "l H N ,H N->l H")
        output_vertical = einsum(outputs['vertical'], C_2 * self.scale, "l H N ,H N->l H")
        # L x L x H
        output = output_horizontal + output_vertical

        output = output.view(self.one_side_length, self.one_side_length, self.kernel_dim)
        output[0, :, :, ] *= 2
        output[:, 0, :, ] *= 2
        output[0, 0] /= 4
        if self.is_complex:
            output = _c2r(output)
            output = output[:, :, :, 0] * output[:, :, :, 1] if self.is_complex else output.squeeze(-1)

        return output

    def compute_sympy_kernel(self):
        A, B1, B2 = self._calc_coeffs()
        return self.coeff_calc.compute_sympy_kernel(A, B1, B2, self.C_1, self.C_2)

    def coeffs(self):
        if self.training:
            return self._calc_coeffs()
        else:
            if self._coeffs is None:
                self._coeffs = self._calc_coeffs()
            return self._coeffs

    def kernel(self, length: int):
        kernel_size = length
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel

    def forward(
            self,
            x,
            state=None,
            rate=None
    ) -> Tensor:
        """Input shape: Time x Batch x Channel
        Args:
            padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
        """

        # Now should be getting B x C x H x W, if not, then we need to transpose
        if self.transpose:
            x = rearrange(x, 'b h w c-> b c h w')
        bsz, embed_dim, H, W = x.size()  # B x C x H x W

        assert embed_dim == self.embed_dim

        # B x C x H x W

        # D x L
        fft_len = H * W
        fft_len = int(math.sqrt(fft_len))
        k = self.kernel(fft_len).permute(2, 0, 1)  # H x effective_L x effective_L
        # for i in range(k.shape[0]):
        #     plot_heatmap(k[i], f'kernel {i}')
        # exit()

        if self.one_side_length < H:
            padding_amount = H - self.one_side_length
            k = torch.nn.functional.pad(k, (0, padding_amount, 0, padding_amount))
        s = 0
        # x = x.view(bsz, embed_dim, int(math.sqrt(seq_len)), int(math.sqrt(seq_len)))
        two_dim_seq_len = H
        y = None
        # Split kernels to four directions
        kernels = list(
            torch.split(k, [self.n_ssm for i in range(self.directions_amount)],
                        dim=0))  # 4 kernels, one for each direction.
        # Transform Kernels from L x L x n_ssm -> L x L x H
        kernels = [repeat(k, ' n l1 l2 ->  (h n) l1 l2', h=self.repeat) for k in kernels]
        if self.directions_amount == 4:
            flip_dims = [[], [-2], [-1], [-2, -1]]
        else:
            flip_dims = [[], [-2, -1]]

        for idx, flip in enumerate(flip_dims):
            k = kernels[idx]
            curr_x = torch.flip(x, dims=flip)

            k_f = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))
            x_f = torch.fft.rfft2(curr_x.float(), s=(2 * fft_len, 2 * fft_len))
            curr = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:two_dim_seq_len + s,
                   s:two_dim_seq_len + s]
            curr_after_flip = torch.flip(curr, dims=flip)
            if y is None:
                y = curr_after_flip
            else:
                y += curr_after_flip
        y = y.type_as(x)
        # B x D x L -> L x B x D
        y = self.dropout(self.activation(y))
        if self.transpose:
            y = rearrange(y, 'b c h w-> b h w c')
        y = self.mixing_linear_layer(y)
        # Using activation and linearity between the channels

        return y,None


class FictionalArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_kernel_to_sympy():
    device = 'cpu'
    ndim = 1
    embed_dim = 1
    L = 20 ** 2
    L_one_sided = int(math.sqrt(L))

    random_x = False
    bidirectional = False
    # truncation = None
    seed = 42
    torch.manual_seed(seed)
    args = FictionalArgs(**{'n_ssm': 1, 'complex_ssm': False, 'directions_amount': 1, 'fp16': False})

    ssm2d = TwoDimensionalSSM(embed_dim, ndim, L=L, force_coeff_calc=True, args=args)
    ssm2d.to(device)

    # X creation
    BATCH_SIZE = 1
    if random_x:
        x = torch.randn(L, BATCH_SIZE, embed_dim).to(device)
    else:
        x = (torch.arange(L, dtype=torch.float) + 1).view(L, 1, 1).repeat(1, BATCH_SIZE, embed_dim).to(device)
    sympy_kernel = ssm2d.compute_sympy_kernel()
    kernel = ssm2d._compute_kernel(L_one_sided).squeeze(-1)
    sympy_kernel = torch.from_numpy(sympy_kernel.values.astype(np.float32)).to(device)
    print(kernel)
    assert torch.allclose(kernel, sympy_kernel, atol=1e-4)


if __name__ == '__main__':
    test_kernel_to_sympy()
