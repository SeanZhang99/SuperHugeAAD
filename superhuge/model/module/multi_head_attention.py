from typing import Sequence
from einops import rearrange
from torch import nn, Tensor
from einops.layers.torch import EinMix


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        embed_dim_per_head: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.embed_dim_per_head = embed_dim_per_head
        self.dropout = dropout
        self.qkv_project = EinMix(
            "b L d_in -> b L d_embed",
            weight_shape="d_in d_embed",
            d_in=input_dim,
            d_embed=num_heads * embed_dim_per_head * 3,
        )
        self.qkv_dropout = nn.Dropout(dropout)
        self.output_project = EinMix(
            "b L d_embed -> b L d_in",
            weight_shape="d_embed d_in",
            d_embed=num_heads * embed_dim_per_head,
            d_in=input_dim,
        )
        self.output_dropout = nn.Dropout(dropout)

        self.scale = self.embed_dim_per_head**0.5

        self.dropout = dropout

    def forward(self, x):
        qkv: Sequence[Tensor] = self.qkv_project(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(
                self.qkv_dropout(t), "b L (h d) -> b h L d", h=self.num_heads
            ),
            qkv,
        )
        out = nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout
        )
        out = rearrange(out, "b h L d -> b L (h d)")
        out = self.output_project(out)
        out = self.output_dropout(out)
        return out
