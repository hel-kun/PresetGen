# DualStreamAttention CNN Decoder
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from utils.param import CATEGORICAL_PARAM_NAMES, CONTINUOUS_PARAM_NAMES

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.silu(x)
        return self.dropout(x)
    
class DualStreamInteraction(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_atten_A_to_B = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.cross_atten_B_to_A = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_A = nn.LayerNorm(embed_dim)
        self.norm_B = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, A: Tensor, B: Tensor) -> Tuple[Tensor, Tensor]:
        # (B, C, L )-> (B, L, C)
        a_t = A.permute(0, 2, 1)
        b_t = B.permute(0, 2, 1)

        attn_output_A, _ = self.cross_atten_A_to_B(a_t, b_t, b_t)
        attn_output_B, _ = self.cross_atten_B_to_A(b_t, a_t, a_t)

        a_t = self.norm_A(a_t + attn_output_A)
        b_t = self.norm_B(b_t + attn_output_B)

        return a_t.permute(0, 2, 1), b_t.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)

class DsaCnnDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        cnn_kernel_size: int = 3,
        dropout: float = 0.1,
        categorical_param_size: Optional[dict] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.categ_convs = nn.ModuleList([
            ConvBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=cnn_kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.cont_convs = nn.ModuleList([
            ConvBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=cnn_kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.atten_layers = nn.ModuleList([
            DualStreamInteraction(
                embed_dim=embed_dim,
                num_heads=4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.categ_param_heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, size) for name, size in (categorical_param_size if categorical_param_size is not None else {name: 5 for name in CATEGORICAL_PARAM_NAMES}).items()
        })
        self.cont_param_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid() # 0〜1に正規化
            ) for name in CONTINUOUS_PARAM_NAMES
        })

    def forward(self, tgt_cont: Tensor, tgt_categ: Tensor) -> Tuple[dict, dict]:
        categ_x = tgt_categ.permute(0, 2, 1)  # (B, 1, C) -> (B, C, 1)
        cont_x = tgt_cont.permute(0, 2, 1)    # (B, 1, C) -> (B, C, 1)

        for conv_categ, conv_cont, atten in zip(self.categ_convs, self.cont_convs, self.atten_layers):
            categ_x = conv_categ(categ_x)
            cont_x = conv_cont(cont_x)
            categ_x, cont_x = atten(categ_x, cont_x)

        categ_x = categ_x.permute(0, 2, 1)  # (B, C, 1) -> (B, 1, C)
        cont_x = cont_x.permute(0, 2, 1)    # (B, C, 1) -> (B, 1, C)

        categ_outputs = {
            name: head(categ_x) for name, head in self.categ_param_heads.items()
        }
        cont_outputs = {
            name: head(cont_x) for name, head in self.cont_param_heads.items()
        }
        return categ_outputs, cont_outputs

