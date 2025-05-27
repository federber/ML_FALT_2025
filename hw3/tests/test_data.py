
import os
import dill
import pytest
import torch
import numpy as np
import pandas as pd
from torchtext.data import Example, Dataset, Field

FASTTEXT_VEC_PATH = "data/cc.ru.300.vec"
WORDFIELD_PATH = "data/word_field_nlp_pupupu.pt"
VOCAB_PATH = "data/vocab_nlp_pupupu.pt"

@pytest.fixture(scope="module")
def word_field():
    assert os.path.exists(WORDFIELD_PATH), "word_field не найден"
    field = torch.load(WORDFIELD_PATH, pickle_module=dill)
    assert hasattr(field, 'vocab'), "word_field не содержит vocab"
    return field

@pytest.fixture(scope="module")
def vocab():
    assert os.path.exists(VOCAB_PATH), "vocab не найден"
    return torch.load(VOCAB_PATH, pickle_module=dill)

@pytest.fixture(scope="module")
def pretrained_vectors():
    assert os.path.exists(FASTTEXT_VEC_PATH), "FastText вектора не найдены"
    vectors = {}
    with open(FASTTEXT_VEC_PATH, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            tokens = line.rstrip().split(" ")
            word = tokens[0]
            vec = np.array(tokens[1:], dtype=np.float32)
            vectors[word] = vec / np.linalg.norm(vec) if np.linalg.norm(vec) > 0 else vec
    return vectors

def test_vocab_special_tokens(word_field):
    vocab = word_field.vocab
    for token in ["<unk>", "<pad>", "<s>", "</s>"]:
        assert token in vocab.stoi, f"Токен {token} отсутствует в словаре"

def test_vocab_size(word_field):
    vocab_size = len(word_field.vocab)
    assert 1000 < vocab_size < 60000, f"Размер словаря подозрителен: {vocab_size}"

def test_vocab_unk_ratio(word_field):
    unk_idx = word_field.vocab.stoi["<unk>"]
    unk_count = sum(1 for word in word_field.vocab.itos if word_field.vocab.stoi[word] == unk_idx)
    unk_ratio = unk_count / len(word_field.vocab.itos)
    assert unk_ratio < 0.15, f"Слишком много <unk> токенов: {unk_ratio:.2%}"

def test_fasttext_coverage(word_field, pretrained_vectors):
    vocab = word_field.vocab
    total = len(vocab.itos)
    covered = sum(1 for token in vocab.itos if token in pretrained_vectors)
    coverage = covered / total
    print(f"FastText покрытие: {coverage:.2%}")
    assert coverage > 0.85, f"Покрытие FastText слишком низкое: {coverage:.2%}"

def test_sample_token_embedding(word_field, pretrained_vectors):
    for word in ['москва', 'экономика', 'президент', 'интернет']:
        if word in word_field.vocab.stoi:
            assert word in pretrained_vectors, f"Слово {word} есть в словаре, но отсутствует в fastText"