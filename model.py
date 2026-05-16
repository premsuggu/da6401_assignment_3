"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
import spacy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Vocabulary

# ══════════════════════════════════════════════════════════════════════
#   CORE ATTENTION 
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard scaled dot-product attention implementation.
    """
    d_k = Q.size(-1)
    # Scaled dot-product
    attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # Masked values are set to a very large negative number
        attn_logits = attn_logits.masked_fill(mask == True, -1e9)
    
    weights = F.softmax(attn_logits, dim=-1)
    # In case of full masking (NaNs), zero them out
    weights = torch.nan_to_num(weights, nan=0.0)
    
    output = torch.matmul(weights, V)
    return output, weights


# ══════════════════════════════════════════════════════════════════════
#  MASKING HELPERS 
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Creates a padding mask for the source sequence."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """Creates a combined causal and padding mask for the target sequence."""
    seq_len = tgt.size(1)
    # Look-ahead (causal) mask
    causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1).bool()
    # Padding mask
    padding_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    return causal_mask.unsqueeze(0).unsqueeze(0) | padding_mask


# ══════════════════════════════════════════════════════════════════════
#  SUB-MODULES
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using simple naming conventions.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # No bias in projection layers to match reference model's weight structure
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 1. Linear Projections & Split Heads
        # (B, S, D) -> (B, S, H, d_k) -> (B, H, S, d_k)
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        attn_out, _ = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 3. Concatenation & Output Projection
        # (B, H, S, d_k) -> (B, S, H, d_k) -> (B, S, D)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(self.dropout(attn_out))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding registered as a buffer."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """Two-layer feed-forward network."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ARCHITECTURE BLOCKS
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """Single encoder block using simple naming."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Self-attention + Add & Norm
        x = self.attn_norm(x + self.dropout(self.self_attention(x, x, x, src_mask)))
        # Feed-forward + Add & Norm
        x = self.ffn_norm(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayer(nn.Module):
    """Single decoder block using simple naming."""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.attn_norm = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # 1. Self-Attention
        x = self.attn_norm(x + self.dropout(self.self_attention(x, x, x, tgt_mask)))
        # 2. Cross-Attention
        x = self.cross_norm(x + self.dropout(self.encoder_attention(x, memory, memory, src_mask)))
        # 3. Feed-Forward
        x = self.ffn_norm(x + self.dropout(self.feed_forward(x)))
        return x


class Encoder(nn.Module):
    """Full encoder stack."""
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.attn_norm.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Full decoder stack."""
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.attn_norm.normalized_shape)

    def forward(self, x: torch.Tensor, memory: torch.Tensor, 
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FINAL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Unified Transformer model with autonomous setup.
    """
    def __init__(
        self,
        src_vocab_size: int   = 18669,
        tgt_vocab_size: int   = 9797,
        d_model:        int   = 256,
        N:              int   = 4,
        num_heads:      int   = 8,
        d_ff:           int   = 1024,
        dropout:        float = 0.1,
        weights_id:     str   = "1WslWWWpo_IUEornHD7qBhHKaQIBFXHIU",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        # Architecture components
        self.source_embedding = nn.Embedding(src_vocab_size, d_model)
        self.target_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout)
        
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder_stack = Encoder(enc_layer, N)
        
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder_stack = Decoder(dec_layer, N)
        
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # Autonomous Setup
        self.vocab_src = None
        self.vocab_tgt = None
        self.spacy_de = None
        
        if weights_id:
            # We use 'best_model.pt' as the local filename for the autograder
            self._load_from_drive(weights_id, local_path="best_model.pt")

    def _load_from_drive(self, drive_id: Optional[str], local_path: str = "best_model.pt"):
        """Autonomous download and loading of weights/vocab."""
        if not os.path.exists(local_path) and drive_id:
            print(f"Downloading weights from Drive ID: {drive_id}...")
            gdown.download(id=drive_id, output=local_path, quiet=False)
        
        if os.path.exists(local_path):
            ckpt = torch.load(local_path, map_location='cpu', weights_only=False)
            # Extract weights
            if 'model_state_dict' in ckpt:
                self.load_state_dict(ckpt['model_state_dict'])
            else:
                self.load_state_dict(ckpt)
            
            # Extract vocabs
            self.vocab_src = ckpt.get('vocab_src')
            self.vocab_tgt = ckpt.get('vocab_tgt')
            print(f"Successfully loaded weights and vocab from {local_path}")
            
        # Autonomous Spacy Setup
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except OSError:
            print("Downloading spacy model 'de_core_news_sm'...")
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "de_core_news_sm"], check=True)
            self.spacy_de = spacy.load("de_core_news_sm")

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.source_embedding(src) * math.sqrt(self.d_model))
        return self.encoder_stack(x, src_mask)

    def decode(self, memory: torch.Tensor, src_mask: torch.Tensor, 
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        x = self.pos_enc(self.target_embedding(tgt) * math.sqrt(self.d_model))
        x = self.decoder_stack(x, memory, src_mask, tgt_mask)
        return self.output_layer(x)

    def forward(self, src, tgt, src_mask, tgt_mask) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, german_sentence: str) -> str:
        """String-to-string translation using greedy decoding."""
        self.eval()
        device = next(self.parameters()).device
        
        if not self.spacy_de or not self.vocab_src or not self.vocab_tgt:
            return "Error: Model not fully initialized with weights and vocab."

        # Tokenize & Numericalize
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(german_sentence)]
        src_ids = [2] + [self.vocab_src[t] for t in tokens] + [3] # SOS=2, EOS=3
        src_tensor = torch.tensor(src_ids).unsqueeze(0).to(device)
        src_mask = make_src_mask(src_tensor)
        
        # Decoding
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            ys = torch.ones(1, 1).fill_(2).type_as(src_tensor.data)
            
            for _ in range(100):
                tgt_mask = make_tgt_mask(ys)
                logits = self.decode(memory, src_mask, ys, tgt_mask)
                next_word = logits[0, -1].argmax().item()
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
                if next_word == 3: break
        
        # Detokenize
        res = [self.vocab_tgt.itos[idx] for idx in ys[0].tolist() if idx not in [2, 3, 1]]
        return " ".join(res)
