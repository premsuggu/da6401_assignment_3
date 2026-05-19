"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion  = nn.KLDivLoss(reduction='sum')

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]
            target : shape [batch * tgt_len]
        """
        assert logits.size(1) == self.vocab_size
        
        true_dist = logits.data.clone()
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0 and mask.size(0) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            
        return self.criterion(F.log_softmax(logits, dim=-1), true_dist)


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.
    """
    if is_train:
        model.train()
    else:
        model.eval()
        
    total_loss = 0
    total_tokens = 0

    for src, tgt in data_iter:
        src = src.to(device)
        tgt = tgt.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_y = tgt[:, 1:]
        
        src_mask = make_src_mask(src, pad_idx=1)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=1)
        
        if is_train:
            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = loss_fn(
                logits.contiguous().view(-1, logits.size(-1)),
                tgt_y.contiguous().view(-1),
            )
            
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        else:
            with torch.no_grad():
                logits = model(src, tgt_input, src_mask, tgt_mask)
                loss = loss_fn(
                    logits.contiguous().view(-1, logits.size(-1)),
                    tgt_y.contiguous().view(-1),
                )
        
        total_loss += loss.item()
        token_count = (tgt_y != 1).sum().item()
        total_tokens += token_count

    return total_loss / total_tokens if total_tokens > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)
    
    memory = model.encode(src, src_mask)
    
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    
    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, pad_idx=1)
        
        out = model.decode(memory, src_mask, ys, tgt_mask)
        
        prob = out[:, -1]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        
        if next_word == end_symbol:
            break
            
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

from torchtext.data.metrics import bleu_score

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.
    """
    model.eval()
    targets = []
    outputs = []
    
    SOS_IDX = 2
    EOS_IDX = 3
    PAD_IDX = 1
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(test_dataloader):
            for sample_idx in range(src.size(0)):
                src_item = src[sample_idx].unsqueeze(0).to(device)
                tgt_item = tgt[sample_idx]
                
                src_mask = make_src_mask(src_item, pad_idx=PAD_IDX)
                
                ys = greedy_decode(model, src_item, src_mask, max_len, SOS_IDX, EOS_IDX, device)
                
                candidate_tokens = []
                for idx in ys[0].tolist():
                    if idx in [SOS_IDX, EOS_IDX, PAD_IDX]:
                        continue
                    candidate_tokens.append(tgt_vocab.itos[idx])
                outputs.append(candidate_tokens)
                
                reference_tokens = []
                for idx in tgt_item.tolist():
                    if idx in [SOS_IDX, EOS_IDX, PAD_IDX]:
                        continue
                    reference_tokens.append(tgt_vocab.itos[idx])
                targets.append([reference_tokens])
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  - Processed {batch_idx + 1} batches...")

    score = bleu_score(outputs, targets)
    return score * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
    vocab_src=None,
    vocab_tgt=None
) -> None:
    """
    Save model + optimiser + scheduler state to disk.
    """
    model_config = {
        'src_vocab_size': model.src_embed.num_embeddings,
        'tgt_vocab_size': model.tgt_embed.num_embeddings,
        'd_model':   model.d_model,
        'N':         len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].self_attn.num_heads,
        'd_ff':      model.encoder.layers[0].ffn.linear1.out_features,
        'dropout':   model.encoder.layers[0].self_attn.dropout.p
    }
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'model_config': model_config,
        'vocab_src': vocab_src,
        'vocab_tgt': vocab_tgt
    }
    
    torch.save(state, path)
    print(f"Checkpoint saved to {path} (bundled with vocab)")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.
    """
    checkpoint = torch.load(path, map_location=next(model.parameters()).device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and checkpoint['optimizer_state_dict']:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    return checkpoint['epoch']


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

import wandb
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler
from functools import partial

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.
    """
    config = {
        "d_model": 256,
        "N": 3,
        "num_heads": 8,
        "d_ff": 512,
        "dropout": 0.1,
        "batch_size": 32,
        "num_epochs": 10,
        "warmup_steps": 4000,
        "smoothing": 0.1,
        "learning_rate": 1.0,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # wandb.init(project="da6401-a3", config=config)
    
    print("Loading datasets...")
    train_ds = Multi30kDataset(split='train')
    train_ds.build_vocab()
    
    val_ds = Multi30kDataset(split='validation')
    val_ds.vocab_src = train_ds.vocab_src
    val_ds.vocab_tgt = train_ds.vocab_tgt
    
    test_ds = Multi30kDataset(split='test')
    test_ds.vocab_src = train_ds.vocab_src
    test_ds.vocab_tgt = train_ds.vocab_tgt
    
    PAD_IDX = train_ds.PAD_IDX
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], 
                              shuffle=True, collate_fn=partial(collate_fn, pad_idx=PAD_IDX))
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"], 
                              shuffle=False, collate_fn=partial(collate_fn, pad_idx=PAD_IDX))
    test_loader  = DataLoader(test_ds, batch_size=1,
                              shuffle=False, collate_fn=partial(collate_fn, pad_idx=PAD_IDX))
    
    model = Transformer(
        src_vocab_size=len(train_ds.vocab_src),
        tgt_vocab_size=len(train_ds.vocab_tgt),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=config["d_model"], warmup_steps=config["warmup_steps"])
    
    loss_fn = LabelSmoothingLoss(vocab_size=len(train_ds.vocab_tgt), pad_idx=PAD_IDX, smoothing=config["smoothing"])
    
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler, epoch, is_train=True, device=device)
        print(f"  Train Loss: {train_loss:.4f}")
        
        val_loss = run_epoch(val_loader, model, loss_fn, None, None, epoch, is_train=False, device=device)
        print(f"  Val Loss:   {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path="best_model.pt", 
                            vocab_src=train_ds.vocab_src, vocab_tgt=train_ds.vocab_tgt)
            
    print("\nEvaluating BLEU on test set...")
    load_checkpoint("best_model.pt", model)
    bleu = evaluate_bleu(model, test_loader, train_ds.vocab_tgt, device=device)
    print(f"Final BLEU: {bleu:.2f}")


if __name__ == "__main__":
    run_training_experiment()
