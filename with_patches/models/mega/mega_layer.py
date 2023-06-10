import torch.nn as nn
from .moving_average_gated_attention import MovingAverageGatedAttention
from .normalized_feedforward_network import NormalizedFeedForwardNetwork
from torch import Tensor


class MegaLayer(nn.Module):
    """Encoder layer block.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(
            self,
            embed_dim: int = 512,
            hidden_dim: int = 1024,
            z_dim: int = 128,
            n_dim: int = 2,
            ffn_embed_dim: int = 256,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            hidden_dropout: float = 0.0,
            activation_dropout: float = 0.,
            drop_path=0.0,
            chunk_size: int = -1,
            truncation=None,
            max_positions: int = 1024,
            activation='silu',
            attention_activation='softmax',
            norm_type: str = 'layernorm',
            no_rel_pos_bias=False,
            patch_amount=None,
            args=None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.mega_layer = MovingAverageGatedAttention(
            embed_dim=embed_dim,
            zdim=z_dim,
            hdim=hidden_dim,
            ndim=n_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            drop_path=drop_path,
            chunk_size=chunk_size,
            truncation=truncation,
            max_positions=max_positions,
            activation=activation,
            attention_activation=attention_activation,
            bidirectional=True,
            norm_type=norm_type,
            no_rel_pos_bias=no_rel_pos_bias,
            patch_amount=patch_amount,
            args=args
        )
        self.nffn = NormalizedFeedForwardNetwork(
            embed_dim=embed_dim,
            ffn_hidden_dim=ffn_embed_dim,
            dropout=dropout,
            hidden_dropout=activation_dropout,
            drop_path=drop_path,
            activation=activation,
            norm_type=norm_type,
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        x = self.mega_layer(x)
        x = self.nffn(x)
        return x
