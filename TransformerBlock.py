import torch.nn as nn
from MultiHeadAttention import MultiHeadAttention
import torch
from FeedForward import FeedForward
from LayerNorm import LayerNorm
import configparser

class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn=MultiHeadAttention(d_in=cfg["emb_dim"],
                                     d_out=cfg["emb_dim"],
                                     context_length=cfg["context_length"],
                                     num_heads=cfg["n_heads"],
                                     dropout=cfg["drop_rate"],
                                     qkv_bias=cfg["qkv_bias"])
        self.ff=FeedForward(cfg)
        self.norm1=LayerNorm(cfg["emb_dim"])
        self.norm2=LayerNorm(cfg["emb_dim"])
        self.drop_shortcut=nn.Dropout(cfg["drop_rate"])

    def forward(self,x):
        shortcut=x
        x=self.norm1(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        shortcut=x
        x=self.norm2(x)
        x=self.attn(x)
        x=self.drop_shortcut(x)
        x=x+shortcut

        return x