import timeit
import math
import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .SPT import ShiftedPatchTokenization
from models.mega.exponential_moving_average import MultiHeadEMA
from models.mega.two_d_ssm_recursive import TwoDimensionalSSM
from .mega.relative_positional_bias import RelativePositionalBias

# from src.models.sequence.modules.s4nd import S4ND
from .mix_ffn import MixFFN


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout=0.):
        super().__init__()
        self.dim = dim

        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., is_LSA=False, args=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.init_qkv()
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def init_qkv(self):
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        init_weights(self.to_qkv)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class EmaAttention(Attention):
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., is_LSA=False,
                 bidirectional=True, args=None):
        super().__init__(dim, num_patches, heads, dim_head, dropout, is_LSA, args)
        self.inner_dim = dim_head * heads
        self.dim = dim
        # self.use_cls_token = args.use_cls_token
        self.smooth_v_as_well = True  # args.smooth_v_as_well
        self.use_relative_pos_embedding = args.use_relative_pos_embedding
        if self.use_relative_pos_embedding:
            self.rel_pos_bias = RelativePositionalBias(num_patches + 1)
        if args.ema == 'ssm_2d':
            self.move = TwoDimensionalSSM(embed_dim=self.dim, L=num_patches, args=args)
        elif args.ema == 's4nd':
            config_path = args.s4nd_config
            # Read from config path with ymal
            import yaml
            config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
            config['n_ssm'] = args.n_ssm
            config['d_state'] = args.ndim
            self.move = S4ND(**config, d_model=self.dim, l_max=int(math.sqrt(num_patches)), return_state=False)
        elif args.ema == 'ema':
            self.move = MultiHeadEMA(self.dim, ndim=args.ndim, bidirectional=bidirectional, truncation=None)
        else:
            self.move = nn.Identity()
        # TODO: DELETE THIS
        # Go over self.move named parameters and print name, shape, total size
        # tot = 0
        # for name, param in self.move.named_parameters():
        #     print(name, param.shape, param.numel())
        #     tot += param.numel()
        # print('Total:',tot)
        # exit()

    def init_qkv(self):
        self.to_qk = nn.Linear(self.dim, self.inner_dim * 2, bias=False)
        self.to_v = nn.Linear(self.dim, self.inner_dim, bias=False)
        init_weights(self.to_qk)
        init_weights(self.to_v)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x_to_be_moved = x[:, 1:, :]
        cls_token = x[:, 0, :].unsqueeze(1)
        x_without_cls_token_moved = rearrange(self.move(rearrange(x_to_be_moved, 'b l h -> l b h')),
                                              'l b h -> b l h')
        x_moved = x_without_cls_token_moved
        x_moved = torch.cat([cls_token, x_without_cls_token_moved], dim=1)

        qk = self.to_qk(x_moved).chunk(2, dim=-1)

        if self.smooth_v_as_well:
            v = self.to_v(x_moved)
        else:
            v = self.to_v(x)
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qk)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        if self.use_relative_pos_embedding:
            bias = self.rel_pos_bias(k.size(2))
            attn = attn + bias
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout=0., stochastic_depth=0.,
                 is_LSA=False, args=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}
        attention_cls = Attention if (not args.ema or args.use_mix_ffn) else EmaAttention
        for i in range(depth):
            layers = []
            layers.append(PreNorm(num_patches, dim,
                                  attention_cls(dim, num_patches, heads=heads, dim_head=dim_head, dropout=dropout,
                                                is_LSA=is_LSA,
                                                args=args)))
            if args.use_mix_ffn:
                ffn = PreNorm(num_patches, dim,
                              MixFFN(dim, hidden_features=dim * mlp_dim_ratio, drop=dropout, num_patches=num_patches,
                                     with_cls=True, args=args))
            else:
                ffn = PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout=dropout))
            layers.append(ffn)
            self.layers.append(nn.ModuleList(layers))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
        return x


class ViT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, channels=3,
                 dim_head=16, dropout=0., emb_dropout=0., stochastic_depth=0., is_LSA=False, is_SPT=False, args):
        super().__init__()
        image_height, image_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_classes = num_classes
        self.use_relative_pos_embedding = args.use_relative_pos_embedding
        self.no_pos_embedding = args.no_pos_embedding

        if not is_SPT:
            self.to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
                nn.Linear(self.patch_dim, self.dim)
            )

        else:
            self.to_patch_embedding = ShiftedPatchTokenization(3, self.dim, patch_size, is_pe=True)

        real_patch_amount = self.num_patches + 1
        self.pos_embedding = nn.Parameter(
            torch.randn(1, real_patch_amount, self.dim)) if (
                not self.use_relative_pos_embedding and not self.no_pos_embedding) else \
            torch.zeros(1, real_patch_amount, self.dim).requires_grad_(False).cuda()

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout,
                                       stochastic_depth, is_LSA=is_LSA, args=args)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.num_classes)
        )

        self.apply(init_weights)
        self.tot_times = []
        self.idx = 0

    def forward(self, img):
        # patch embedding
        # B x C x H x W -> B x N x D
        start = timeit.default_timer()
        x = self.to_patch_embedding(img)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        out = self.mlp_head(x[:, 0])
        end = timeit.default_timer()
        self.tot_times.append(end - start)
        self.idx += 1
        return out
