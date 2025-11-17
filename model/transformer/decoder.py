import torch
from torch import nn
from torch import Tensor
from typing import Optional, Tuple
from utils.param import CATEGORICAL_PARAM_NAMES, CONTINUOUS_PARAM_NAMES

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        categorical_param_size: Optional[dict] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.categ_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.cont_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

        self.cont_norm = nn.LayerNorm(embed_dim)
        self.categ_norm = nn.LayerNorm(embed_dim)

        self.categ_param_heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, size) for name, size in (categorical_param_size if categorical_param_size is not None else {name: 5 for name in CATEGORICAL_PARAM_NAMES}).items()
        })
        self.cont_param_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Sigmoid() # 0〜1に正規化
            ) for name in CONTINUOUS_PARAM_NAMES
        })

    def forward(
        self,
        tgt_cont: Tensor,
        tgt_categ: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tuple[dict, dict]:
        cont_output = tgt_cont
        categ_output = tgt_categ

        for _ in range(self.num_layers):
            cont_output = self.cont_layer(
                cont_output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            categ_output = self.categ_layer(
                categ_output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        cont_output = self.cont_norm(cont_output)
        categ_output = self.categ_norm(categ_output)

        outputs = {'categorical': {}, 'continuous': {}}
        for name, head in self.categ_param_heads.items():
            outputs['categorical'][name] = head(categ_output.squeeze(1))  # (batch, size)
        for name, head in self.cont_param_heads.items():
            outputs['continuous'][name] = head(cont_output.squeeze(1))  # (batch, 1)
        
        return outputs