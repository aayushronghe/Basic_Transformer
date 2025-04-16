import torch.nn as nn
from TransformerBlock import TransformerBlock
from LayerNorm import LayerNorm
import torch

class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.token_emb=nn.Embedding(int(cfg["vocab_size"]),int(cfg["emb_dim"]))
        self.pos_emb=nn.Embedding(int(cfg['context_length']),int(cfg["emb_dim"]))
        self.drop_emb=nn.Dropout(float(cfg["drop_rate"]))

        self.trf_blocks=nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(int(cfg["n_layers"]))]
        )
        self.final_norm=LayerNorm(int(cfg["emb_dim"]))
        self.out_head=nn.Linear(int(cfg["emb_dim"]),int(cfg["vocab_size"]),bias=False)

    def forward(self,in_idx):
        batch_size,seq_len=in_idx.shape
        tok_embeds=self.token_emb(in_idx)
        pos_embeds=self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x=tok_embeds+pos_embeds
        x=self.drop_emb(x)
        x=self.trf_blocks(x)
        x=self.final_norm(x)
        logits=self.out_head(x)

        return logits
