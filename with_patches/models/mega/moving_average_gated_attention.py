# Author: Xuezhe Ma (Max)
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Parameter

from .drop import DropPath
from .fairseq_dropout import FairseqDropout, FairseqFeatureDropout
from .relative_positional_bias import RelativePositionalBias
from .sequence_norm import SequenceNorm
from .exponential_moving_average import MultiHeadEMA
from .two_d_ssm_recursive import TwoDimensionalSSM
from .mega_utils import relu2, laplace, get_activation_fn
from src.models.sequence.modules.s4nd import S4ND

class MovingAverageGatedAttention(nn.Module):
    """Exponential Moving Average Gated Attention.

    See "" for more details.
    """

    def __init__(
            self,
            embed_dim,
            zdim,
            hdim,
            ndim,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            drop_path=0.0,
            activation='silu',
            attention_activation='softmax',
            bidirectional=False,
            chunk_size=-1,
            truncation=None,
            norm_type='layernorm',
            feature_dropout=False,
            no_rel_pos_bias=False,
            max_positions=1024,
            patch_amount=None,
            heads_num=1,
            args=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads_num = heads_num
        self.embed_dim_per_head = zdim // heads_num
        self.hdim = hdim
        self.zdim = zdim
        self.ndim = ndim
        self.activation = get_activation_fn(activation=activation)
        self.attention_activation = attention_activation
        self.scaling = self.zdim ** -0.5 if attention_activation == 'softmax' else None

        dropout_module = FairseqFeatureDropout if feature_dropout else FairseqDropout
        self.dropout = dropout_module(dropout, module_name=self.__class__.__name__)
        self.hidden_dropout = dropout_module(hidden_dropout, module_name=self.__class__.__name__)
        # Attention dropout is standard dropout
        self.attention_dropout = FairseqDropout(attention_dropout, module_name=self.__class__.__name__)
        self.drop_path = DropPath(drop_path, dim=1) if drop_path > 0. else nn.Identity()

        self.chunk_size = chunk_size
        self.norm = SequenceNorm(norm_type, embed_dim)

        if args.ema == 'ssm_2d':
            self.move = TwoDimensionalSSM(embed_dim, ndim=ndim, truncation=truncation, L=patch_amount, args=args)
        elif args.ema == 's4nd':
            config_path = args.s4nd_config
            # Read from config path with ymal
            import yaml
            config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
            self.move = S4ND(**config, d_model=embed_dim, l_max=int(math.sqrt(patch_amount)),return_state=False)
        elif args.ema == 'ema':
            self.move = MultiHeadEMA(embed_dim, ndim=ndim, bidirectional=bidirectional, truncation=truncation)
        else:
            self.move = nn.Identity()

        self.v_proj = nn.Linear(embed_dim, hdim)
        self.mx_proj = nn.Linear(embed_dim, zdim + hdim + 2 * embed_dim)
        self.h_proj = nn.Linear(hdim, embed_dim)

        self.gamma = Parameter(torch.Tensor(2, zdim))
        self.beta = Parameter(torch.Tensor(2, zdim))

        self.max_positions = max_positions
        if no_rel_pos_bias:
            self.rel_pos_bias = None
        else:
            self.rel_pos_bias = RelativePositionalBias(max_positions if chunk_size < 0 else chunk_size)

        self.reset_parameters()

    def reset_parameters(self):
        std = 0.02
        nn.init.normal_(self.v_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.v_proj.bias, 0.0)

        nn.init.normal_(self.mx_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.mx_proj.bias, 0.0)

        nn.init.normal_(self.h_proj.weight, mean=0.0, std=std)
        nn.init.constant_(self.h_proj.bias, 0.0)

        nn.init.normal_(self.gamma, mean=0.0, std=std)
        nn.init.constant_(self.beta, 0.0)

    def element_attention(self, q, k, mask=None):
        slen = k.size(2)
        lengths = slen
        # B x K x C x C
        qk = torch.matmul(q, k.transpose(2, 3)) / lengths
        # C x C
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias(torch.Tensor([slen]))
            qk = qk + bias
        if mask is not None:
            qk = self.add_mask(qk, mask)

        if self.attention_activation == 'relu2':
            attn_weights = relu2(qk)
        elif self.attention_activation == 'laplace':
            attn_weights = laplace(qk)
        else:
            raise ValueError('Unknown attention activation function: {}'.format(self.attention_activation))

        return attn_weights

    def softmax_attention(self, q, k, mask=None):
        slen = k.size(2)
        # scaled attention
        q = q * self.scaling
        # B x K x C x C
        # q = rearrange(q,'b k l (h d) -> b k l h d', h=self.heads_num)
        # k = rearrange(k,'b k l (h d) -> b k l h d', h=self.heads_num)
        qk = torch.matmul(q, k.transpose(-2, -1))
        if self.rel_pos_bias is not None:
            bias = self.rel_pos_bias(slen)
            qk = qk + bias

        # C x C
        if mask is not None:
            qk = self.add_mask(qk, mask)

        attn_weights = F.softmax(qk, dim=-1)
        return attn_weights

    def add_mask(self, qk, mask):
        B_, _, seq_len, _ = qk.size()
        initial_size = qk.size()
        nW = mask.shape[0]
        qk = qk.view(B_ // nW, nW, 1, seq_len, seq_len) + mask.unsqueeze(1).unsqueeze(0)
        qk = qk.view(-1, 1, seq_len, seq_len)
        assert qk.size() == initial_size
        return qk

    def forward(self, x, mask=None) -> Tensor:
        """Input shape: Time x Batch x Channel
        """

        seq_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        residual = x
        x = self.norm(x)

        # L x B x E
        v = self.activation(self.v_proj(x))

        # L x B x D
        mx = self.move(x)
        mx = self.dropout(mx)

        # L x B x D -> L x B x (2*D+S+E)
        base = self.mx_proj(mx)
        u, zr, hx = torch.split(base, [self.embed_dim, self.zdim + self.hdim, self.embed_dim], dim=-1)
        # L x B x D
        u = torch.sigmoid(u)
        # L x B x (E+S)
        z, r = torch.split(F.silu(zr), [self.zdim, self.hdim], dim=-1)
        # L x B x S -> L x B x 1 x S -> L x B x 2 x S
        z = z.unsqueeze(2) * self.gamma + self.beta
        # L x B x 2 x S -> L x B x S
        q, k = torch.unbind(z, dim=2)

        # L x B x D -> B x L x D
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        ctx_len = k.size(1)
        if self.chunk_size < 0:
            # B x L x S -> B x 1 x L x S
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)
        else:
            if seq_len < self.chunk_size:
                q = q.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = seq_len // self.chunk_size
                q = q.reshape(bsz, nc, self.chunk_size, self.zdim)

            if ctx_len < self.chunk_size:
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)
            else:
                # B x L x S -> B x K x C x S
                nc = ctx_len // self.chunk_size
                k = k.reshape(bsz, nc, self.chunk_size, self.zdim)
                v = v.reshape(bsz, nc, self.chunk_size, self.hdim)

        if self.attention_activation == 'softmax':
            attn_weights = self.softmax_attention(q, k, mask)
        else:
            attn_weights = self.element_attention(q, k, mask)

        v = self.hidden_dropout(v, batch_first=True)
        kernel = self.attention_dropout(attn_weights)
        # B x K x C x E -> B x L x E -> L x B x E
        h = torch.matmul(kernel, v).view(bsz, seq_len, self.hdim).transpose(0, 1)
        # L x B x E -> L x B x D
        h = self.activation(hx + self.h_proj(h * r))
        h = self.dropout(h)
        h = self.drop_path(h)
        # L x B x D
        out = torch.addcmul(residual, u, h - residual)

        return out

    def extra_repr(self) -> str:
        return 'edim={}, zdim={}, hdim={}, ndim={}, chunk={}, attn_act={}'.format(
            self.embed_dim, self.zdim, self.hdim, self.ndim, self.chunk_size, self.attention_activation)
