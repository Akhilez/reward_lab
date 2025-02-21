"""
Let's try:
- Vanilla decoder transformer with RoPE
- MLA: Multi-head Latent Attention
- KV caching
- Abstract thinking tokens
- Recurrent latent reasoning

"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from evo_devo_nano.model.rope import apply_rotary_emb, precompute_freqs_cis


class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=False)
        self.out = nn.Linear(self.dim, self.dim)

    def forward(self, x, freqs_cis, is_causal: bool = True):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # 3 x b x nh x l x hd
        q, k, v = qkv[0], qkv[1], qkv[2]  # b x nh x l x hd

        q = apply_rotary_emb(q, freqs_cis)  # b x nh x l x hd
        k = apply_rotary_emb(k, freqs_cis)  # b x nh x l x hd

        attention = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=is_causal)  # b x nh x l x hd
        out = attention.transpose(1, 2).reshape(B, T, C)  # b x l x d
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.attention = Attention(dim, n_heads, dropout=0.1)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(0.1),
        )
        self.layer_id = layer_id
        norm_eps = 1e-6
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, is_causal: bool = True):
        h = x + self.attention(self.attention_norm(x), freqs_cis, is_causal)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class VanillaTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, max_len):
        super(VanillaTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.head_dim = embed_dim // num_heads

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.num_layers):
            self.layers.append(TransformerBlock(layer_id, embed_dim, num_heads))

        self.output_norm = nn.RMSNorm(embed_dim, eps=1e-6)

        self.freqs_cis = precompute_freqs_cis(self.head_dim, max_len * 2)

    def forward(self, in_embeddings, is_causal=True):
        # in_embeddings: (b, l, d)
        b, l, d = in_embeddings.shape
        freq_cis = self.freqs_cis[:l]
        x = in_embeddings
        for layer in self.layers:
            x = layer(x, freq_cis, is_causal)
        return self.output_norm(x)


def try_vanilla_transformer():
    embed_dim = 256
    n_heads = 4
    n_layers = 6
    max_len = 1024

    model = VanillaTransformer(embed_dim, n_heads, n_layers, max_len)

    x = torch.randn(2, 16, embed_dim)
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    try_vanilla_transformer()
