import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
import pickle

class Vocabulary:
    def __init__(self):
        self.itos = []
        self.stoi = {}
        self.unk_idx = 0

    def __len__(self):
        return len(self.itos)

    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_idx)

    def save(self, path):
        payload = {'itos': self.itos, 'stoi': self.stoi, 'unk_idx': self.unk_idx}
        with open(path, 'wb') as file_obj:
            pickle.dump(payload, file_obj)

    @classmethod
    def load_from_original(cls, path):
        orig = torch.load(path, map_location='cpu', weights_only=False)
        obj = cls()
        obj.stoi = orig.token2idx
        max_idx = max(orig.idx2token.keys())
        obj.itos = [orig.idx2token.get(i, "<unk>") for i in range(max_idx + 1)]
        obj.unk_idx = 0 
        return obj

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as file_obj:
            saved_data = pickle.load(file_obj)
        vocab = cls()
        vocab.itos = saved_data['itos']
        vocab.stoi = saved_data['stoi']
        vocab.unk_idx = saved_data['unk_idx']
        return vocab

class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_lang='de', tgt_lang='en'):
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dataset = load_dataset("bentrevett/multi30k", split=split)
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")
        self.tokenizer_src = lambda text: [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]
        self.tokenizer_tgt = lambda text: [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]
        self.vocab_src = None
        self.vocab_tgt = None
        self.UNK_IDX, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX = 0, 1, 2, 3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        src_text = sample[self.src_lang]
        tgt_text = sample[self.tgt_lang]
        src_tokens = [self.SOS_IDX] + [self.vocab_src[token] for token in self.tokenizer_src(src_text)] + [self.EOS_IDX]
        tgt_tokens = [self.SOS_IDX] + [self.vocab_tgt[token] for token in self.tokenizer_tgt(tgt_text)] + [self.EOS_IDX]
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def build_vocabs(de_nlp, en_nlp):
    return None, None

def collate_fn(batch, pad_idx):
    src_items, tgt_items = [], []
    for src, tgt in batch:
        src_items.append(src)
        tgt_items.append(tgt)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_items, batch_first=True, padding_value=pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_items, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded
