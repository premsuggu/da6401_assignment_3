import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from collections import Counter
import os

import pickle

class Vocabulary:
    def __init__(self, counter=None, min_freq=2, specials=['<unk>', '<pad>', '<sos>', '<eos>']):
        if counter is None:
            self.itos = specials[:]
            self.stoi = {token: i for i, token in enumerate(self.itos)}
            self.unk_idx = 0
            return

        self.itos = specials[:]
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        
        # Sort by frequency, then alphabetically for consistency
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        for token, freq in sorted_tokens:
            if freq >= min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
        
        self.unk_idx = self.stoi['<unk>']

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'itos': self.itos, 'stoi': self.stoi, 'unk_idx': self.unk_idx}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(counter=None)
        vocab.itos = data['itos']
        vocab.stoi = data['stoi']
        vocab.unk_idx = data['unk_idx']
        return vocab

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_idx)

class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_lang='de', tgt_lang='en'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Load dataset from Hugging Face
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        
        # Load spacy tokenizers
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        self.tokenizer_src = lambda text: [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        self.tokenizer_tgt = lambda text: [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]
        
        # Vocab placeholders
        self.vocab_src = None
        self.vocab_tgt = None
        
        # Special tokens
        self.special_symbols = ['<unk>', '<pad>', '<sos>', '<eos>']
        self.UNK_IDX, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX = 0, 1, 2, 3

    def build_vocab(self, min_freq=2):
        """
        Builds the vocabulary mapping for src and tgt.
        """
        # Build vocab from training split only
        train_dataset = load_dataset("bentrevett/multi30k", split='train')
        
        counter_src = Counter()
        counter_tgt = Counter()
        
        for sample in train_dataset:
            counter_src.update(self.tokenizer_src(sample[self.src_lang]))
            counter_tgt.update(self.tokenizer_tgt(sample[self.tgt_lang]))

        self.vocab_src = Vocabulary(counter_src, min_freq=min_freq, specials=self.special_symbols)
        self.vocab_tgt = Vocabulary(counter_tgt, min_freq=min_freq, specials=self.special_symbols)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        src_text = item[self.src_lang]
        tgt_text = item[self.tgt_lang]
        
        src_tokens = [self.SOS_IDX] + [self.vocab_src[token] for token in self.tokenizer_src(src_text)] + [self.EOS_IDX]
        tgt_tokens = [self.SOS_IDX] + [self.vocab_tgt[token] for token in self.tokenizer_tgt(tgt_text)] + [self.EOS_IDX]
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def build_vocabs(de_nlp, en_nlp):
    print("Building vocabs for reference model compatibility...")
    ds = Multi30kDataset(split='train')
    ds.build_vocab(min_freq=1)
    # Patch the vocab objects to have .encode and .lookup_token
    def encode(self, tokens):
        return [self.stoi.get(t, self.unk_idx) for t in tokens]
    def lookup_token(self, idx):
        return self.itos[idx]
    
    import types
    ds.vocab_src.encode = types.MethodType(encode, ds.vocab_src)
    ds.vocab_src.lookup_token = types.MethodType(lookup_token, ds.vocab_src)
    ds.vocab_tgt.encode = types.MethodType(encode, ds.vocab_tgt)
    ds.vocab_tgt.lookup_token = types.MethodType(lookup_token, ds.vocab_tgt)
    
    return ds.vocab_src, ds.vocab_tgt
