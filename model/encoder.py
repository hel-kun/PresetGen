import torch
import torch.nn as nn
from transformers import ClapTextModelWithProjection, AutoProcessor, AutoTokenizer, AutoModel

# RoBERTaを用いたテキストエンコーダ
class RoBERTaTextEncorder(nn.Module):
    def __init__(
        self,
        model_name="FacebookAI/roberta-large",
        output_dim=512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if self.model.config.hidden_size != output_dim:
            self.project = nn.Linear(self.model.config.hidden_size, output_dim)
        else:
            self.project = nn.Identity()
    
    def forward(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        embeddings = self.project(last_hidden)  # 次元変換
        return embeddings

    def get_padding_mask(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        padding_mask = ~inputs['attention_mask'].bool()
        return padding_mask

# CLAPを用いたテキストエンコーダ
class CLAPTextEncorder(nn.Module):
    def __init__(
        self,
        model_name="laion/clap-htsat-unfused",
        output_dim=512,
    ):
        super().__init__()
        self.model = ClapTextModelWithProjection.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        if self.model.config.projection_dim != output_dim:
            self.project = nn.Linear(self.model.config.projection_dim, output_dim)
        else:
            self.project = nn.Identity()
    
    def forward(self, text):
        # テキストの前処理
        inputs = self.processor(
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embeddings = outputs.text_embeds  # (batch_size, projection_dim)
        embeddings = self.project(embeddings) # 次元変換
        embeddings = embeddings.unsqueeze(1)  # seq_len次元を追加(batch_size, 1, dim)
        
        return embeddings