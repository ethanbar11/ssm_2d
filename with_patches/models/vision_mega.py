""" Vision MEGA in PyTorch
"""
import math
import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mega.sequence_norm import SequenceNorm
from timm.models.layers import trunc_normal_, lecun_normal_, PatchEmbed
from .mega.mega_layer import MegaLayer

_logger = logging.getLogger(__name__)


class VisionMEGA(nn.Module):
    """ Vision MEGA
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, depth=12,
                 embed_dim=768, hidden_dim=1536, ffn_hidden_dim=1536, zdim=256, ndim=16,
                 representation_size=None, embed_layer=PatchEmbed, patch_impl='conv',
                 norm_type='layernorm', no_pos_emb=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., distilled=False, args=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 0

        assert img_size % patch_size == 0
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dim)#, bias=True)  # , impl=patch_impl)

        num_patches = self.patch_embed.num_patches
        if no_pos_emb:
            self.register_parameter("pos_embed", None)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
            trunc_normal_(self.pos_embed, std=(embed_dim ** -0.5))

        self.patch_norm = nn.Identity()
        patch_amount = (img_size // patch_size) ** 2

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            MegaLayer(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                z_dim=zdim,
                n_dim=ndim,
                ffn_embed_dim=ffn_hidden_dim,
                dropout=drop_rate,
                attention_dropout=attn_drop_rate,
                hidden_dropout=0.,
                drop_path=dpr[i],
                chunk_size=-1,
                truncation=None,
                max_positions=(img_size // patch_size) ** 2,
                activation='silu',  # hard coded in the code. We will always use silu in VIM.
                attention_activation='laplace',
                norm_type=norm_type,
                patch_amount=patch_amount,
                args=args,
            )
            for i in range(depth)])

        self.final_norm = SequenceNorm(norm_type, embed_dim)
        self.out_proj = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(embed_dim, embed_dim)),
            ('act', nn.SiLU())
        ]))
        nn.init.normal_(self.out_proj[0].weight, mean=0.0, std=0.02)
        nn.init.constant_(self.out_proj[0].bias, 0.0)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.SiLU())
            ]))
            nn.init.xavier_uniform_(self.pre_logits[0].weight)
            nn.init.zeros_(self.pre_logits[0].bias)
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        raise NotImplementedError

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', }

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        raise NotImplementedError

    def forward_features(self, x):
        # B x L x D
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # B x L x D -> L x B x D
        x = x.transpose(0, 1)
        x = self.patch_norm(x)
        # Mega layers
        x = self.blocks(x)

        # layers before pooling
        x = self.out_proj(self.final_norm(x))

        # L x B x D -> 1 x B x D
        x = torch.mean(x, dim=0)

        return self.pre_logits(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
