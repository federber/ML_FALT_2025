import os
import dill
import pandas as pd
import numpy as np
import torch
from torchtext.data import Field, Example, Dataset, BucketIterator
from torchtext.vocab import Vocab
from collections import Counter
import gzip
import spacy
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def spacy_tokenize(text):
    nlp = spacy.load("ru_core_news_sm")
    return [token.text.lower() for token in nlp(text) if not token.is_space and not token.is_punct]


def prepare_data(config, device):
    BOS, EOS, UNK, PAD = "<s>", "</s>", "<unk>", "<pad>"
    ru_stopwords = set(stopwords.words("russian"))

    word_field = Field(
        tokenize=spacy_tokenize,
        init_token=BOS,
        eos_token=EOS,
        unk_token=UNK,
        pad_token=PAD,
        include_lengths=True
    )
    fields = [("source", word_field), ("target", word_field)]

    if os.path.exists(config.word_field_path):
        word_field = torch.load(config.word_field_path, pickle_module=dill)
        vocab = torch.load(config.vocab_path, pickle_module=dill)
        word_field.vocab = vocab
    else:
        data = pd.read_csv(config.dataset_path)
        examples = [
            Example.fromlist([
                word_field.preprocess(row.text),
                word_field.preprocess(row.title)
            ], fields)
            for _, row in data.iterrows()
        ]
        token_counter = Counter(word for ex in examples for word in ex.source)
        for stopword in ru_stopwords:
            if stopword in token_counter:
                token_counter[stopword] = max(1, token_counter[stopword] // 10)
        word_field.vocab = Vocab(token_counter, specials=[UNK, PAD, BOS, EOS], min_freq=10)
        torch.save(word_field, config.word_field_path, pickle_module=dill)
        torch.save(word_field.vocab, config.vocab_path, pickle_module=dill)

    data = pd.read_csv(config.dataset_path)
    examples = [
        Example.fromlist([
            word_field.preprocess(row.text),
            word_field.preprocess(row.title)
        ], fields)
        for _, row in data.iterrows()
    ]
    dataset = Dataset(examples, fields)
    train_data, test_data = dataset.split(split_ratio=0.85)

    train_iter, test_iter = BucketIterator.splits(
        (train_data, test_data),
        batch_sizes=(config.batch_size, config.batch_size * 2),
        sort_key=lambda x: len(x.source),
        sort=False,
        device=device
    )

    pretrained_embeddings = load_embeddings(word_field, config.embedding_path)

    return word_field, train_iter, test_iter, pretrained_embeddings


def load_embeddings(field, path):
    vectors = {}
    with open(path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)
            vector /= np.linalg.norm(vector) if np.linalg.norm(vector) > 0 else 1
            vectors[word] = vector

    dim = len(next(iter(vectors.values())))
    matrix = np.random.normal(scale=0.01, size=(len(field.vocab), dim))
    for i, token in enumerate(field.vocab.itos):
        if token in vectors:
            matrix[i] = vectors[token]
    return torch.tensor(matrix, dtype=torch.float)


def make_mask(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(-2)
    tgt_mask = (tgt != pad_idx).unsqueeze(-2)
    size = tgt.size(-1)
    nopeak = torch.triu(torch.ones((1, size, size), device=tgt.device), diagonal=1).bool()
    tgt_mask = tgt_mask & ~nopeak
    return src_mask, tgt_mask


def convert_batch(batch, pad_idx=1):
    src, _ = batch.source
    tgt, _ = batch.target
    src = src.transpose(0, 1)
    tgt = tgt.transpose(0, 1)
    src_mask, tgt_mask = make_mask(src, tgt, pad_idx)
    return src, tgt, src_mask, tgt_mask
