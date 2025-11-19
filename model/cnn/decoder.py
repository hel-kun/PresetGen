import torch
from torch import nn
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

class CnnDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        kernel_size: int = 3,
        categorical_param_size: Optional[dict] = None
    ):
        super(CnnDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.categ_convs = nn.ModuleList([
            ConvBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.cont_convs = nn.ModuleList([
            ConvBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        self.categ_param_heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, size) for name, size in (categorical_param_size if categorical_param_size is not None else {name: 5 for name in CATEGORICAL_PARAM_NAMES}).items()
        })
        self.cont_param_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Tanh()
            ) for name in CONTINUOUS_PARAM_NAMES
        })

    def forward(self, tgt_cont: Tensor, tgt_categ: Tensor, memory: Tensor) -> Tuple[dict, dict]:
        x = memory.permute(0, 2, 1)  # (B, L, C) -> (B, C, L)
        feat_categ = x
        feat_cont = x

        for conv_categ, conv_cont in zip(self.categ_convs, self.cont_convs):
            feat_categ = conv_categ(feat_categ)
            feat_cont = conv_cont(feat_cont)

        pooled_categ = nn.functional.adaptive_avg_pool1d(feat_categ, 1).squeeze(-1)  # (B, C)
        pooled_cont = nn.functional.adaptive_avg_pool1d(feat_cont, 1).squeeze(-1)  # (B, C, 1) -> (B, C)

        outputs = {'categorical': {}, 'continuous': {}}
        for name, head in self.categ_param_heads.items():
            outputs['categorical'][name] = head(pooled_categ)
        for name, head in self.cont_param_heads.items():
            outputs['continuous'][name] = head(pooled_cont)
        
        return outputs