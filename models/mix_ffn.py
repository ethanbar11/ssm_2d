from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch

from models.mega.two_d_ssm_recursive import TwoDimensionalSSM


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        H = int(math.sqrt(N))
        W = H
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MixFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., num_patches=-1,
                 with_cls=False,
                 args=None):
        super().__init__()
        self.with_cls = with_cls
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if args.ema == 'ssm_2d':
            self.move = TwoDimensionalSSM(hidden_features,  L=num_patches, args=args)
            self.ssm = True
        else:
            self.move = DWConv(hidden_features)
            self.ssm = False
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        if self.with_cls:
            cls_token = x[:, 0, :].unsqueeze(1)
            x_to_be_moved = x[:, 1:, :]
        else:
            x_to_be_moved = x
        if not self.ssm:
            x_to_be_moved = self.move(x_to_be_moved)
        else:
            x_to_be_moved = rearrange(self.move(rearrange(x_to_be_moved, 'b l h -> l b h')),
                                      'l b h -> b l h')
        if self.with_cls:
            x = torch.cat((cls_token, x_to_be_moved), dim=1)
        else:
            x = x_to_be_moved
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
