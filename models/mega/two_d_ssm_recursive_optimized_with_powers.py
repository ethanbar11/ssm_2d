# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import os
import timeit

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


def plot_heatmap(x, title=None, save_image=False, save_path=None):
    import os
    import matplotlib.pyplot as plt
    import seaborn as sns

    img = x.cpu().detach().numpy()

    if save_image:
        print('Saving image to:', save_path)
        dirname = os.path.dirname(save_path)

        # Check if the directory exists
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        heatmap = sns.heatmap(img)
        if title is not None:
            plt.title(title)

        figure = heatmap.get_figure()
        figure.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=400)
        figure.clf()
    else:
        sns.heatmap(img)
        if title is not None:
            plt.title(title)
        plt.show()


def plot_histogram(k):
    import seaborn
    import matplotlib.pyplot as plt
    hist = torch.max(torch.max(k, dim=1)[0], dim=1)[0]
    hist = hist.cpu().detach().numpy()
    seaborn.histplot(hist, bins=100)
    plt.show()


class TwoDimensionalSSMOptimized(nn.Module):
    def __init__(
            self,
            embed_dim,
            L=32 ** 2,
            force_coeff_calc=False,
            args=None,
    ):
        super().__init__()
        self.use_old_compute_x = True
        self.is_2_dim = True
        self.embed_dim = embed_dim
        self.ndim = args.ndim
        self.n_ssm = args.n_ssm
        self.normalization = nn.LayerNorm(embed_dim) if args.normalize else nn.Identity()
        self.is_complex = args.complex_ssm
        self.directions_amount = args.directions_amount
        self.use_residual = args.use_residual_inside_ssm
        self.repeat = self.embed_dim // self.n_ssm

        self.scale = math.sqrt(1.0 / self.ndim)
        self.kernel_dim = args.directions_amount * self.n_ssm

        # TODO: Change this where we'll work with other benchmarks
        self.one_side_length = math.ceil(math.sqrt(L))
        self.coeff_calc = CoeffCalculator(self.one_side_length)
        self.coeff_calc.calc_coeffs_lazy(force=force_coeff_calc)
        self.matrices = self.coeff_calc.matrices
        self.one_matrix = self.coeff_calc.whole_as_one
        self.powers = self.coeff_calc.whole_as_one[:, :-1]
        self.one_matrix_only_B_with_coeffs = self.coeff_calc.whole_as_one[:, -1, :, :2]
        self.powers = rearrange(torch.argmax(self.powers, dim=-1).unsqueeze(-1).unsqueeze(-1), 'a b c d e -> a c b d e')

        if self.is_complex:
            self.one_matrix = self.one_matrix.unsqueeze(-1)

            # Create a tensor of zeros with the same shape as x_unsqueezed
            zeros = torch.zeros_like(self.one_matrix)

            # Concatenate along the last dimension to get a tensor of shape (a, b, 2)
            self.one_matrix = torch.cat((self.one_matrix, zeros), -1)

            self.one_matrix = _r2c(self.one_matrix)
        for key, inner_dic in self.matrices.items():
            for symbol, matrix in inner_dic.items():
                if self.is_complex:
                    matrix = matrix.type(torch.complex64)
                self.matrices[key][symbol] = matrix.cuda()

        # self.save_kernel = save_path
        self.last_kernel = None
        self.C_dimensions = self.embed_dim * self.directions_amount
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
            self.C_1 = nn.Parameter(torch.Tensor(self.C_dimensions, self.ndim, 2))
            self.C_2 = nn.Parameter(torch.Tensor(self.C_dimensions, self.ndim, 2))
            self.parameters_without_weight_decay = [self.A_angle, self.A_radius, self.B_1, self.B_2, self.C_1, self.C_2]
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
            self.C_1 = nn.Parameter(torch.Tensor(self.C_dimensions, self.ndim))
            self.C_2 = nn.Parameter(torch.Tensor(self.C_dimensions, self.ndim))
            self.parameters_without_weight_decay = [self.A, self.B_1, self.B_2, self.C_1, self.C_2]

        for param in self.parameters_without_weight_decay:
            param.should_be_without_weight_decay = True
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
        self.whole_output_last_time = None

        # TODO: DELETE, only for speed improvement
        self.kernel_raw = nn.Parameter(torch.rand((2, 1024, 5, 4, 16)))

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

            nn.init.normal_(self.C_1, mean=0.0, std=1.0)
            nn.init.normal_(self.C_2, mean=0.0, std=1.0)

            nn.init.normal_(self.omega, mean=0.0, std=1.0)

    def compute_sympy_kernel(self):
        A, B1, B2 = self._calc_coeffs()
        outcome = self.coeff_calc.compute_sympy_kernel(A, B1, B2, self.C_1, self.C_2)
        outcome = torch.tensor(outcome.astype(float).values).to('cuda')
        return outcome

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

        # l x l  D x N
        A_values = torch.stack(list(A.values()), dim=0)

        # copy A_values power dim times
        A_powers = torch.pow(A_values, self.powers)
        B = torch.stack([B1, B2])
        B_with_coeffs = einsum(self.one_matrix_only_B_with_coeffs, B, 'd R B_1_B_2, B_1_B_2 n_ssm N-> d R n_ssm N')
        whole_output_before_prod = torch.cat((A_powers, B_with_coeffs.unsqueeze(2)), dim=2)
        whole_output = torch.prod(whole_output_before_prod, dim=2)
        whole_output = rearrange(whole_output, 'd (r1 r2) n_ssm N-> d r1 r2 n_ssm N', r1=self.one_side_length ** 2)
        whole_output = einsum(whole_output, 'd r1 r2 n_ssm N-> d r1 n_ssm N')
        return whole_output

    def compute_x_matrix_old(self, kernel_dim):
        # H x N each
        A, B1, B2 = self._calc_coeffs()
        power_dim = kernel_dim * 2
        # l x l  D x N
        A_values = torch.stack(list(A.values()), dim=0)
        # copy A_values power dim times
        # A_values = repeat(A_values, 'a n_ssm N-> a P n_ssm N', P=power_dim)
        A_values = rearrange(torch.linalg.vander(A_values, N=power_dim),
                             'a n_ssm N L -> a L n_ssm N')
        B = torch.nn.functional.pad(torch.stack([B1, B2], dim=0),
                                    (0, 0, 0, 0, 0, A_values.shape[1] - 2)).unsqueeze(0)
        values = torch.cat([A_values, B], dim=0)
        whole_output = einsum(self.one_matrix, values, 'd a R V, a V n_ssm N-> d a R n_ssm N')
        whole_output = einsum(whole_output[:, 0], whole_output[:, 1], whole_output[:, 2], whole_output[:, 3],
                              whole_output[:, 4],
                              'd R n_ssm N, d R n_ssm N, d R n_ssm N, d R n_ssm N, d R n_ssm N-> d R n_ssm N')
        self.whole_output_last_time = whole_output

        whole_output = rearrange(whole_output, 'd (r1 r2) n_ssm N-> d r1 r2 n_ssm N', r1=self.one_side_length ** 2)
        whole_output = einsum(whole_output, 'd r1 r2 n_ssm N-> d r1 n_ssm N')
        return whole_output

    def _compute_kernel(self):
        self._kernel = None
        # l x l x D x N
        if self.use_old_compute_x:
            x_matrix = self.compute_x_matrix_old(self.one_side_length)
        else:
            x_matrix = self.compute_x_matrix(self.one_side_length)
        # L x L x D x N

        # L x L x H
        if self.is_complex:
            C_1 = _r2c(self.C_1)
            C_2 = _r2c(self.C_2)
        else:
            C_1 = self.C_1
            C_2 = self.C_2
        C = torch.stack([C_1, C_2], dim=0) * self.scale  # direction X (embed_dim * corners) X N
        C = rearrange(C,
                      'axis (H n_ssm_multiplied_by_kernel_corners) N ->'  # Embed dim 96 N_ssm =2 directions =4 (96 *4)
                      'axis H n_ssm_multiplied_by_kernel_corners N',
                      n_ssm_multiplied_by_kernel_corners=self.n_ssm * self.directions_amount)
        # C shape: axis X (embed_dim // (n_ssm * directions)) X (n_ssm * directions) X N
        # output = einsum(outputs, C, 'direction patches n_ssm N, directions  n_ssm N -> patches n_ssm')
        output = einsum(x_matrix, C, 'axis patches n_ssm_directions N, axis H  n_ssm_directions N '
                                     '-> patches H n_ssm_directions')
        # output2 = einsum(x_matrix, C, 'axis patches n_ssm_directions N, axis H  n_ssm_directions N '
        #                              '-> axis patches H n_ssm_directions')
        # output3 = output2[0] + output2[1]
        output = rearrange(output, 'patches H n_ssm_directions -> patches (H n_ssm_directions)')

        output = output.view(self.one_side_length, self.one_side_length, self.C_dimensions)
        output[0, :, :, ] *= 2
        output[:, 0, :, ] *= 2
        output[0, 0] /= 4

        if self.is_complex:
            output = output.real
        self.last_kernel = output
        return output

    def kernel(self):
        return self._compute_kernel()

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
        assert self.directions_amount > 1
        if len(x.shape) == 3:
            # Expecting L x B x D
            seq_len, bsz, embed_dim = x.size()
            residual = einsum(x, self.omega, 'L B D, D -> L B D')
            # L x B x D -> B x D x L
            x = x.permute(1, 2, 0)
            fft_len = int(math.sqrt(seq_len))
            x = x.view(bsz, embed_dim, fft_len, fft_len)
        elif len(x.shape) == 4:
            # Expecting B x D x L
            bsz, embed_dim, seq_len, seq_len2 = x.size()
            residual = einsum(x, self.omega, 'B D L1 L2, D -> B D L1 L2')
            assert seq_len == seq_len2, "2-D SSM Currently implemented only for square images."
            assert embed_dim == self.embed_dim
            fft_len = seq_len
        else:
            raise TypeError(
                'Tensor inserted into 2-D SSM should be 3 dimensional (Length Batch Channels) or 4 dimensional (Batch '
                'Dimension Length Length)')
        # D x L
        k = self.kernel().permute(2, 0, 1)  # (Directions * N_SSM) x kernel_size x kernel_size
        if k.shape[-1] < fft_len:
            # print("! Padding")
            padding_amount = fft_len - k.shape[-1]
            k = torch.nn.functional.pad(k, (0, padding_amount, 0, padding_amount))
        s = 0
        out = None
        # Split kernels to four directions
        k_f_s = torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len))

        # kernels = list(
        #     torch.split(k, [self.embed_dim for i in range(self.directions_amount)],
        #                 dim=0))  # 4 kernels, one for each direction.
        # Transform Kernels from L x L x n_ssm -> L x L x H
        # kernels = [torch.fft.rfft2(k.float(), s=(2 * fft_len, 2 * fft_len)) for k in kernels]
        if self.directions_amount == 4:
            flip_dims = [[], [-2], [-1], [-2, -1]]
        else:
            flip_dims = [[], [-2, -1]]
        all_x_lst = torch.cat([torch.flip(x, dims=flip) for flip in flip_dims], dim=1)

        x_f_s = torch.fft.rfft2(all_x_lst, s=(2 * fft_len, 2 * fft_len))
        curr_all = torch.fft.irfft2(x_f_s * k_f_s, s=(2 * fft_len, 2 * fft_len))[..., s:fft_len + s,
                   s:fft_len + s]
        for idx, flip in enumerate(flip_dims):
            curr = curr_all[:, idx * embed_dim:(idx + 1) * embed_dim, :, :]
            curr_after_flip = torch.flip(curr, dims=flip)
            if out is None:
                out = curr_after_flip
            else:
                out += curr_after_flip
        out = out.type_as(x)
        if len(residual.shape) == 3:
            out = rearrange(out, 'b d l1 l2 -> (l1 l2) b d ')
        if self.use_residual:
            out += residual
        return self.normalization(out)  # notice normalization might be the identity function.
