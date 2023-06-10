# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math
from typing import Optional
from einops import rearrange, einsum, repeat
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from models.mega.ssm_coefficient import CoeffCalculator

_c2r = torch.view_as_real
_r2c = torch.view_as_complex


def plot_heatmap(x, title, save_image=False, save_path=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    img = x.cpu().detach().numpy()
    if save_image:
        print('Saving image to: ', save_path)
        dirname = os.path.dirname(save_path)

        # Check if the directory exists
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        sv = sns.heatmap(img)  # cbar=False)
        figure = sv.get_figure()
        figure.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=400)
        # plt.show()
        figure.clf()
    else:
        plt.show()


def plot_histogram(k):
    import seaborn
    import matplotlib.pyplot as plt
    hist = torch.max(torch.max(k, dim=1)[0], dim=1)[0]
    hist = hist.cpu().detach().numpy()
    seaborn.histplot(hist, bins=100)
    plt.show()


class TwoDimensionalSSM(nn.Module):
    def __init__(
            self,
            embed_dim,
            ndim=2,
            truncation=None,
            L=32 ** 2,
            force_coeff_calc=False,
            use_static_kernel=True,
            args=None,
            save_path=None
    ):
        super().__init__()
        print(L)
        self.is_2_dim = True
        self.truncation = truncation
        self.embed_dim = embed_dim
        self.ndim = args.ndim
        self.n_ssm = args.n_ssm
        self.normalization = nn.LayerNorm(embed_dim) if args.normalize else nn.Identity()
        self.is_complex = args.complex_ssm
        print(args.complex_ssm)
        self.directions_amount = args.directions_amount
        self.repeat = self.embed_dim // self.n_ssm

        # TODO: Add support in ndim>1 bidirectionality, and truncation
        self.scale = math.sqrt(1.0 / self.ndim)
        self.kernel_dim = args.directions_amount * self.n_ssm

        # TODO: Change this where we'll work with other benchmarks
        self.one_side_length = math.ceil(math.sqrt(L))
        self.coeff_calc = CoeffCalculator(self.one_side_length)
        self.coeff_calc.calc_coeffs_lazy(force=force_coeff_calc)
        self.matrices = self.coeff_calc.matrices
        for key, inner_dic in self.matrices.items():
            for symbol, matrix in inner_dic.items():
                if self.is_complex:
                    matrix = matrix.type(torch.complex64)
                self.matrices[key][symbol] = matrix.cuda()

        self.use_static_kernel = use_static_kernel
        self.save_kernel = save_path
        self.last_kernel = None
        # H x N
        if self.is_complex:
            self.A_angle = nn.ParameterDict({
                'A_1': torch.Tensor(self.kernel_dim, self.ndim),
                'A_2': torch.Tensor(self.kernel_dim, self.ndim),
                'A_3': torch.Tensor(self.kernel_dim, self.ndim),
                'A_4': torch.Tensor(self.kernel_dim, self.ndim),
            })
            self.A_radius = nn.ParameterDict({
                'A_1': torch.Tensor(self.kernel_dim, self.ndim),
                'A_2': torch.Tensor(self.kernel_dim, self.ndim),
                'A_3': torch.Tensor(self.kernel_dim, self.ndim),
                'A_4': torch.Tensor(self.kernel_dim, self.ndim),
            })
            self.B_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
            self.B_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
            # D x N
            self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
            self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim, 2))
        else:
            self.A = {
                'A_1': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
                'A_2': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
                'A_3': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
                'A_4': nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim)),
            }
            self.A = nn.ParameterDict(self.A)
            self.B_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            self.B_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            # D x N
            self.C_1 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
            self.C_2 = nn.Parameter(torch.Tensor(self.kernel_dim, self.ndim))
        # sized D because this is a residual connection (element-wise)
        self.omega = nn.Parameter(torch.Tensor(embed_dim))

        self.horizontal_flow = None
        self.vertical_flow = None
        self.counter = 0

        self._kernel = None
        self._coeffs = None

        self.reset_parameters()

        self.onnx_trace = False
        self.tpu = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def reset_parameters(self):
        with torch.no_grad():
            # delta & alpha
            if self.is_complex:
                for symbol, tensor in self.A_angle.items():
                    nn.init.uniform_(tensor, a=0.5, b=3)
                for symbol, tensor in self.A_radius.items():
                    nn.init.uniform_(tensor, a=0.5, b=3)
            else:
                for symbol, tensor in self.A.items():
                    nn.init.normal_(tensor, mean=0, std=0.2)
            nn.init.normal_(self.B_1, mean=0.0, std=0.2)
            nn.init.normal_(self.B_2, mean=0.0, std=0.2)
            # TODO: After expanding to n_dim>1 , checkout what's being done with beta in EMA

            nn.init.normal_(self.C_1, mean=0.0, std=1.0)
            nn.init.normal_(self.C_2, mean=0.0, std=1.0)

            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def _calc_coeffs(self):
        self._coeffs = None
        # D x N x 1
        A = {}
        if self.is_complex:
            for symbol, tensor in self.A_angle.items():
                angle = torch.sigmoid(tensor) * 2 * math.pi  # angle between [0,2pi]
                radius = torch.sigmoid(self.A_radius[symbol])  # radius between [0,1]
                A[symbol] = torch.polar(radius, angle)
        else:
            for symbol, tensor in self.A.items():
                A[symbol] = tensor
                A[symbol] = torch.sigmoid(tensor)
        B1 = torch.sigmoid(self.B_1)
        B2 = torch.sigmoid(self.B_2)
        if self.is_complex:
            B1 = _r2c(B1)
            B2 = _r2c(B2)
        return A, B1, B2

    def compute_x_matrix(self, kernel_dim):
        # H x N each
        A, B1, B2 = self._calc_coeffs()
        power_dim = kernel_dim * 2
        # l x l  D x N
        A_powers = {}
        for symbol, tensor in A.items():
            A_powers[symbol] = torch.exp(
                einsum(torch.arange(power_dim).to(tensor.device),
                       torch.log(tensor),
                       'l , h n-> l h n'))
        B = torch.stack([B1, B2], dim=0)
        outputs = {}
        for direction in self.matrices.keys():
            # Should be sized R x H x N
            outputs[direction] = None
        for direction, matrices in self.matrices.items():
            output = outputs[direction]
            for symbol, matrix in matrices.items():
                vec = B if symbol == 'B' else A_powers[symbol]
                current_calculation = einsum(matrix, vec, 'R V, V h n -> R h n')

                if output is None:
                    output = current_calculation
                else:
                    output = output * current_calculation
            outputs[direction] = output
        for direction, matrix in outputs.items():
            outputs[direction] = rearrange(matrix, '(r1 r2) h n-> r1 r2 h n',
                                           r1=self.one_side_length ** 2,
                                           r2=self.coeff_calc.coeff_rows_amount // (self.one_side_length ** 2))
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
        # C_1 = torch.softmax(C_1, dim=1)
        # C_2 = torch.softmax(C_2, dim=1)
        output_horizontal = einsum(outputs['horizontal'], C_1 * self.scale, "l H N ,H N->l H")
        output_vertical = einsum(outputs['vertical'], C_2 * self.scale, "l H N ,H N->l H")
        # L x L x H
        output = output_horizontal + output_vertical

        output = output.view(self.one_side_length, self.one_side_length, self.kernel_dim)
        output[0, :, :, ] *= 2
        output[:, 0, :, ] *= 2
        output[0, 0] /= 4

        if self.is_complex:
            output = output.real
        self.last_kernel = output
        # output = rearrange(torch.softmax(rearrange(output,'h w c -> (h w) c'),dim=0),'(h w) c -> h w c',h=8)
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
        kernel_size = length if self.truncation is None else min(self.truncation, length)
        if self.training:
            return self._compute_kernel(kernel_size)
        else:
            if self._kernel is None or self._kernel.size(-1) < kernel_size:
                self._kernel = self._compute_kernel(kernel_size)
            return self._kernel

    def forward(
            self,
            x,
            padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
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
        if padding_mask is not None:
            x = x * (1.0 - padding_mask.unsqueeze(1).type_as(x))

        # D x L
        fft_len = seq_len
        fft_len = int(math.sqrt(fft_len))
        k = self.kernel(fft_len).permute(2, 0, 1)  # H x L x L
        s = 0
        if self.save_kernel:
            for i in range(k.shape[0]):
                # Create image path and save it
                img_path = os.path.join(self.save_kernel, f'kernel_{i}.png')
                plot_heatmap(k[i], f'kernel {i}', save_image=True, save_path=img_path)
        kernel_size = k.size(1)
        # Pad x with zeros to power 2 of self.one_side_length
        # x = torch.nn.functional.pad(x, (self.one_side_length ** 2 - x.shape[-1],0, 0, 0, 0, 0))
        x = x.view(bsz, embed_dim, self.one_side_length, -1)
        out = None
        if self.directions_amount > 1:
            # Split kernels to four directions
            kernels = list(
                torch.split(k, [self.n_ssm for i in range(self.directions_amount)],
                            dim=0))  # 4 kernels, one for each direction.
            # for i in range(k.shape[0]):
            #     plot_heatmap(k[i], f'kernel {i}')
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
                curr = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:self.one_side_length + s,
                       s:self.one_side_length + s]
                curr_after_flip = torch.flip(curr, dims=flip)
                if out is None:
                    out = curr_after_flip
                else:
                    out += curr_after_flip
        else:
            k_f = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))
            x_f = torch.fft.rfft2(x.float(), s=(2 * fft_len, 2 * fft_len))
            out = torch.fft.irfft2(x_f * k_f, s=(2 * fft_len, 2 * fft_len))[..., s:two_dim_seq_len + s,
                  s:two_dim_seq_len + s]
        out = out.type_as(x)
        out = rearrange(out, 'b d l1 l2 -> b d (l1 l2)')
        # B x D x L -> L x B x D
        out = out.permute(2, 0, 1) + residual
        # out = F.silu(out.permute(2, 0, 1) + residual)
        return self.normalization(out)
