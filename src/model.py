import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self._gamma = nn.Parameter(torch.ones(features))
        self._beta = nn.Parameter(torch.zeros(features))
        self._eps = eps

    def forward(self, inputs):
        mean = inputs.mean(-1, keepdim=True)
        std = inputs.std(-1, keepdim=True)
        return self._gamma * (inputs - mean) / (std + self._eps) + self._beta

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout_rate):
        super().__init__()
        self._norm = LayerNorm(size)
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, sublayer):
        return inputs + self._dropout(sublayer(self._norm(inputs)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_rate):
        super().__init__()
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self._dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, heads_count, d_model, dropout_rate=0.1):
        super().__init__()
        assert d_model % heads_count == 0
        self._d_k = d_model // heads_count
        self._heads_count = heads_count
        self._attention = ScaledDotProductAttention(dropout_rate)
        self._w_q = nn.Linear(d_model, d_model)
        self._w_k = nn.Linear(d_model, d_model)
        self._w_v = nn.Linear(d_model, d_model)
        self._w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query = self._w_q(query).view(nbatches, -1, self._heads_count, self._d_k).transpose(1, 2)
        key = self._w_k(key).view(nbatches, -1, self._heads_count, self._d_k).transpose(1, 2)
        value = self._w_v(value).view(nbatches, -1, self._heads_count, self._d_k).transpose(1, 2)
        x, _ = self._attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self._heads_count * self._d_k)
        return self._w_o(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.attn_block = ResidualBlock(size, dropout_rate)
        self.ff_block = ResidualBlock(size, dropout_rate)

    def forward(self, x, mask):
        x = self.attn_block(x, lambda x: self.self_attn(x, x, x, mask))
        return self.ff_block(x, self.feed_forward)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout_rate):
        super().__init__()
        self.self_attn = self_attn
        self.encoder_attn = encoder_attn
        self.feed_forward = feed_forward
        self.self_block = ResidualBlock(size, dropout_rate)
        self.enc_block = ResidualBlock(size, dropout_rate)
        self.ff_block = ResidualBlock(size, dropout_rate)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.self_block(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.enc_block(x, lambda x: self.encoder_attn(x, enc_out, enc_out, src_mask))
        return self.ff_block(x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, blocks_count, heads_count, dropout, embeddings, save_probs=False):
        super().__init__()
        self.emb = embeddings
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, MultiHeadedAttention(heads_count, d_model, dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout)
            for _ in range(blocks_count)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.emb(x)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, blocks_count, heads_count, dropout, embeddings, out_layer, save_probs=False):
        super().__init__()
        self.emb = embeddings
        self.out_layer = out_layer
        self.blocks = nn.ModuleList([
            DecoderLayer(d_model, MultiHeadedAttention(heads_count, d_model, dropout),
                         MultiHeadedAttention(heads_count, d_model, dropout),
                         PositionwiseFeedForward(d_model, d_ff, dropout), dropout)
            for _ in range(blocks_count)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.emb(x)
        for block in self.blocks:
            x = block(x, enc_out, src_mask, tgt_mask)
        return self.out_layer(self.norm(x))

class EncoderDecoder(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, d_ff=1024,
                 blocks_count=4, heads_count=8, dropout=0.1, save_probs=False, use_shared_emb=False,
                 pretrained_embeddings=None):
        super().__init__()
        self.out_layer = nn.Linear(d_model, target_vocab_size)
        self.d_model = d_model

        if pretrained_embeddings is not None:
            emb_layer = nn.Embedding(source_vocab_size, 300)
            emb_layer.weight.data.copy_(pretrained_embeddings)
            emb_layer.weight.requires_grad = True
            projection = nn.Linear(300, d_model)
            pos_enc = PositionalEncoding(d_model, dropout)
            emb = nn.Sequential(emb_layer, projection, pos_enc)
        else:
            emb = nn.Sequential(nn.Embedding(source_vocab_size, d_model), PositionalEncoding(d_model, dropout))

        if use_shared_emb:
            self.enc_emb = self.dec_emb = emb
            self.out_layer.weight = self.dec_emb[0].weight
        else:
            self.enc_emb = emb
            self.dec_emb = emb

        self.encoder = Encoder(source_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout, self.enc_emb)
        self.decoder = Decoder(target_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout, self.dec_emb, self.out_layer)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)
