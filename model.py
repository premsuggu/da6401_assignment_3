"""
model.py — Transformer Architecture Skeleton
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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    """
    d_k = Q.size(-1)
    # scores: (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # mask should be broadcastable to (..., seq_q, seq_k)
        # Use a large negative value instead of -inf to avoid NaNs
        scores = scores.masked_fill(mask == True, -1e9)
    
    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).
    """
    # src shape: [batch, src_len]
    # mask shape: [batch, 1, 1, src_len]
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.
    """
    # tgt shape: [batch, tgt_len]
    batch_size, tgt_len = tgt.size()
    
    # 1. Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    
    # 2. Causal mask: [1, 1, tgt_len, tgt_len]
    # True means masked out (future tokens)
    causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) # [1, 1, tgt_len, tgt_len]
    
    # Combine masks: Masked if either is True
    tgt_mask = pad_mask | causal_mask
    
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
        # Optional: Initialize biases to zero
        nn.init.constant_(self.w_q.bias, 0)
        nn.init.constant_(self.w_k.bias, 0)
        nn.init.constant_(self.w_v.bias, 0)
        nn.init.constant_(self.w_o.bias, 0)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        batch_size = query.size(0)
        
        # 1. Linear projections
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        # x: (batch, heads, seq_q, d_k), attn: (batch, heads, seq_q, seq_k)
        x, attn = scaled_dot_product_attention(q, k, v, mask=mask)
        
        # 3. Concatenation and Output Projection
        # (batch, heads, seq_q, d_k) -> (batch, seq_q, heads, d_k) -> (batch, seq_q, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.w_o(x)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        # Register as buffer (won't be updated by optimizer)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]
        """
        # x shape: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
        """
        # 1. Self-Attention + Add & Norm
        _x = self.self_attn(x, x, x, mask=src_mask)
        x = self.norm1(x + self.dropout1(_x))
        
        # 2. Feed-Forward + Add & Norm
        _x = self.ffn(x)
        x = self.norm2(x + self.dropout2(_x))
        
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        """
        # 1. Masked Self-Attention + Add & Norm
        _x = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(_x))
        
        # 2. Encoder-Decoder Cross-Attention + Add & Norm
        _x = self.cross_attn(x, memory, memory, mask=src_mask)
        x = self.norm2(x + self.dropout2(_x))
        
        # 3. Feed-Forward + Add & Norm
        _x = self.ffn(x)
        x = self.norm3(x + self.dropout3(_x))
        
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

import gdown
import os
import spacy
from dataset import Vocabulary

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = 7853,
        tgt_vocab_size: Optional[int] = 5893,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        load_weights: bool = True
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        # Placeholders
        self.vocab_src = None
        self.vocab_tgt = None
        self.spacy_de = None

        # 1. Download and Extract (Autonomous Setup)
        if load_weights:
            self._load_autonomous()
            # If vocab was in the checkpoint, update sizes
            if self.vocab_src:
                src_vocab_size = len(self.vocab_src)
            if self.vocab_tgt:
                tgt_vocab_size = len(self.vocab_tgt)

        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc   = PositionalEncoding(d_model, dropout)
        
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder  = Encoder(encoder_layer, N)
        
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder  = Decoder(decoder_layer, N)
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 2. Finalize weights loading if it was downloaded
        if load_weights and os.path.exists("best_model.pt"):
            checkpoint = torch.load("best_model.pt", map_location='cpu')
            self.load_state_dict(checkpoint['model_state_dict'])
            print("Model weights and vocabularies loaded successfully.")
        else:
            self._init_parameters()

    def _load_autonomous(self):
        """Single-file download and setup."""
        WEIGHTS_ID = "YOUR_SINGLE_WEIGHTS_GDRIVE_ID" # Copy your ID here
        
        if not os.path.exists("best_model.pt") and WEIGHTS_ID != "YOUR_SINGLE_WEIGHTS_GDRIVE_ID":
            gdown.download(id=WEIGHTS_ID, output="best_model.pt", quiet=False)
            
        if os.path.exists("best_model.pt"):
            checkpoint = torch.load("best_model.pt", map_location='cpu')
            self.vocab_src = checkpoint.get('vocab_src')
            self.vocab_tgt = checkpoint.get('vocab_tgt')
            
        try:
            self.spacy_de = spacy.load("de_core_news_sm")
        except:
            pass

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def infer(self, german_sentence: str) -> str:
        """
        End-to-end inference: String -> String.
        """
        self.eval()
        device = next(self.parameters()).device
        
        if not self.spacy_de or not self.vocab_src or not self.vocab_tgt:
            return "Error: Autonomous setup failed (vocab or tokenizer missing)."

        # 1. Tokenize
        tokens = [tok.text.lower() for tok in self.spacy_de.tokenizer(german_sentence)]
        # 2. Vectorize
        # SOS=2, EOS=3, PAD=1, UNK=0
        src_indices = [2] + [self.vocab_src[tok] for tok in tokens] + [3]
        src_tensor = torch.tensor(src_indices).unsqueeze(0).to(device)
        
        # 3. Mask
        src_mask = make_src_mask(src_tensor, pad_idx=1)
        
        # 4. Greedy Decode
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)
            ys = torch.ones(1, 1).fill_(2).type_as(src_tensor.data) # Start with <sos>
            
            for i in range(100): # max_len=100
                tgt_mask = make_tgt_mask(ys, pad_idx=1)
                out = self.decode(memory, src_mask, ys, tgt_mask)
                prob = out[:, -1]
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.item()
                
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word)], dim=1)
                if next_word == 3: # <eos>
                    break
        
        # 5. Detokenize
        output_indices = ys[0].tolist()
        output_tokens = []
        for idx in output_indices:
            if idx in [2, 3, 1]: # Skip <sos>, <eos>, <pad>
                continue
            output_tokens.append(self.vocab_tgt.itos[idx])
            
        return " ".join(output_tokens)
