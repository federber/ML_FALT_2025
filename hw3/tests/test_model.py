

import torch
import torch.nn as nn
import pytest
import numpy as np

from model import (
    EncoderDecoder, make_mask, PositionalEncoding,
    LayerNorm, MultiHeadedAttention, LabelSmoothingLoss
)

@pytest.fixture
def dummy_batch():
    batch_size, seq_len = 4, 10
    src = torch.randint(2, 100, (batch_size, seq_len))
    tgt = torch.randint(2, 100, (batch_size, seq_len))
    return src, tgt

def test_encoder_decoder_forward(dummy_batch):
    src, tgt = dummy_batch
    model = EncoderDecoder(source_vocab_size=200, target_vocab_size=200, d_model=128, heads_count=4)
    src_mask, tgt_mask = make_mask(src, tgt, pad_idx=0)
    out = model(src, tgt, src_mask, tgt_mask)
    assert out.shape == (src.shape[0], tgt.shape[1], 200)

def test_mask_shapes(dummy_batch):
    src, tgt = dummy_batch
    src_mask, tgt_mask = make_mask(src, tgt, pad_idx=0)
    assert src_mask.shape == (src.shape[0], 1, src.shape[1])
    assert tgt_mask.shape == (src.shape[0], tgt.shape[1], tgt.shape[1])
    assert torch.all(tgt_mask[:, 0].bool()), "Первые токены должны быть видимы"

def test_encoder_output_shape(dummy_batch):
    src, _ = dummy_batch
    src_mask = (src != 0).unsqueeze(-2)
    model = EncoderDecoder(200, 200, d_model=128, heads_count=4)
    enc_out = model.encoder(src, src_mask)
    assert enc_out.shape == (src.shape[0], src.shape[1], model.d_model)

def test_decoder_output_shape(dummy_batch):
    src, tgt = dummy_batch
    src_mask, tgt_mask = make_mask(src, tgt, pad_idx=0)
    model = EncoderDecoder(200, 200, d_model=128, heads_count=4)
    enc_out = model.encoder(src, src_mask)
    out = model.decoder(tgt, enc_out, src_mask, tgt_mask)
    assert out.shape == (tgt.shape[0], tgt.shape[1], 200)

def test_shared_embeddings():
    model = EncoderDecoder(1000, 1000, use_shared_emb=True)
    emb_enc = model.encoder._emb[0].weight
    emb_dec = model.decoder._emb[0].weight
    assert torch.allclose(emb_enc, emb_dec), "Shared embeddings должны совпадать"

def test_positional_encoding_stability():
    pos_enc = PositionalEncoding(d_model=32, dropout=0.1)
    x = torch.zeros(1, 100, 32)
    out = pos_enc(x)
    assert not torch.isnan(out).any(), "PositionalEncoding даёт NaN"

def test_layernorm_behavior():
    norm = LayerNorm(32)
    x = torch.randn(4, 10, 32)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()

def test_multihead_attention_output():
    mha = MultiHeadedAttention(heads_count=4, d_model=64)
    q = k = v = torch.randn(2, 5, 64)
    mask = torch.ones(2, 1, 5).bool()
    out = mha(q, k, v, mask)
    assert out.shape == (2, 5, 64)

def test_label_smoothing_loss_behavior():
    criterion = LabelSmoothingLoss(smoothing=0.1, vocab_size=10, ignore_index=0)
    pred = torch.randn(8, 10)
    target = torch.randint(1, 10, (8,))
    loss = criterion(pred, target)
    assert loss.item() > 0