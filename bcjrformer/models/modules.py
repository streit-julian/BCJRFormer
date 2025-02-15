from einops import einsum, rearrange
import torch.nn as nn
from torch.nn import LayerNorm
import torch


class PreNorm(nn.Module):
    def __init__(self, size):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x):
        return self.norm(x)


class MultiHeadedAttention(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        dropout: float = 0.0,
        mask: torch.Tensor | None = None,
        cache_attn: bool = False,
    ):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h

        self.scale = self.d_k**-0.5

        self.h = h

        self.to_k = nn.Linear(d_model, d_model)
        self.to_q = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)

        self.to_out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        self.attn = None
        self.cache_attn = cache_attn

        self.mask = mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        q = self.to_q(q)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.h)

        k = self.to_k(k)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.h)

        v = self.to_v(v)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.h)

        dots = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        if self.mask is not None:
            dots = dots.masked_fill(self.mask, -1e9)

        if padding_mask is not None:
            dots = dots.masked_fill(padding_mask, -1e9)

        attn = dots.softmax(dim=-1)

        if self.cache_attn:
            self.attn = attn

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SelfAttentionLayer(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        dropout: float = 0.0,
        mask: torch.Tensor | None = None,
        cache_attn: bool = False,
    ):
        super(SelfAttentionLayer, self).__init__()

        self.pre_norm_attn = PreNorm(d_model)
        self.pre_norm_ff = PreNorm(d_model)

        self.attn = MultiHeadedAttention(
            h,
            d_model,
            dropout,
            mask,
            cache_attn,
        )

        self.ff = FeedForward(d_model, d_model * 4, dropout=dropout)

    def forward(self, x, padding_mask: torch.Tensor | None = None):

        xn = self.pre_norm_attn(x)

        x = x + self.attn(xn, xn, xn, padding_mask)

        x = x + self.ff(self.pre_norm_ff(x))

        return x


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        h: int,
        d_model: int,
        dropout: float = 0.0,
        mask: torch.Tensor | None = None,
        cache_attn: bool = False,
    ):
        super(CrossAttentionLayer, self).__init__()

        self.pre_norm_attn = PreNorm(d_model)
        self.pre_norm_ff = PreNorm(d_model)

        self.attn = MultiHeadedAttention(
            h,
            d_model,
            dropout,
            mask,
            cache_attn,
        )

        self.ff = FeedForward(d_model, d_model * 4, dropout=dropout)

    def forward(self, x, y):

        x = x + self.attn(self.pre_norm_attn(x), y, y)

        x = x + self.ff(self.pre_norm_ff(x))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        heads: int,
        dropout: float = 0.0,
        mask: torch.Tensor | None = None,
        cache_attn: bool = False,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([])

        for _ in range(n_layers):
            self.layers.append(
                SelfAttentionLayer(
                    heads,
                    d_model,
                    dropout=dropout,
                    mask=mask,
                    cache_attn=cache_attn,
                )
            )

    def forward(self, x, padding_mask: torch.Tensor | None = None):
        for attn in self.layers:
            x = attn(x, padding_mask)
        return x
