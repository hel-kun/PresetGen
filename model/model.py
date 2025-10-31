import torch
from torch import nn
from config import DEVICE
from model.encoder import CLAPTextEncorder, RoBERTaTextEncorder
from model.decoder import PresetGenDecoder

class PresetGenModel(nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, num_layers=6, dropout=0.1):
        super(PresetGenModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.text_encoder = CLAPTextEncorder(output_dim=embedding_dim) # もしくは RoBERTaTextEncorder
        self.decoder = PresetGenDecoder(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, src, tgt=None):
        text_embeddings = self.text_encoder(src)
        if tgt is None:
            # 推論モード(のはず)
            batch_size = text_embeddings.size(0)
            tgt = {
                'continuous': torch.zeros(batch_size, 1, self.embedding_dim).to(DEVICE),
                'categorical': torch.zeros(batch_size, 1, self.embedding_dim).to(DEVICE)
            }

        if isinstance(self.text_encoder, RoBERTaTextEncorder):
            memory_key_padding_mask = self.text_encoder.tokenizer.get_padding_mask(src).to(DEVICE)
        else:
            memory_key_padding_mask = None
        outputs = self.decoder(
            tgt,
            text_embeddings,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return outputs