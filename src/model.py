import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import heapq


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
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
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
        self._attn_probs = None

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
        x, self._attn_probs = self._attention(query, key, value, mask)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self._heads_count * self._d_k)
        x = self._w_o(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        return self.w_2(self.dropout(F.relu(self.w_1(inputs))))


class EncoderBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout_rate):
        super().__init__()

        self._self_attn = self_attn
        self._feed_forward = feed_forward
        self._self_attention_block = ResidualBlock(size, dropout_rate)
        self._feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, mask):
        outputs = self._self_attention_block(inputs, lambda inputs: self._self_attn(inputs, inputs, inputs, mask))
        return self._feed_forward_block(outputs, self._feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, encoder_attn, feed_forward, dropout_rate):
        super().__init__()

        self._self_attn = self_attn
        self._encoder_attn = encoder_attn
        self._feed_forward = feed_forward
        self._self_attention_block = ResidualBlock(size, dropout_rate)
        self._attention_block = ResidualBlock(size, dropout_rate)
        self._feed_forward_block = ResidualBlock(size, dropout_rate)

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        outputs = self._self_attention_block(
            inputs, lambda inputs: self._self_attn(inputs, inputs, inputs, target_mask)
        )
        outputs = self._attention_block(
            outputs, lambda inputs: self._encoder_attn(inputs, encoder_output, encoder_output, source_mask)
        )
        return self._feed_forward_block(outputs, self._feed_forward)


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate, embeddings, save_probs = False):
        super().__init__()

        self._emb = embeddings

        block = lambda: EncoderBlock(
            size=d_model,
            self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
            feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
            dropout_rate=dropout_rate
        )
        self._blocks = nn.ModuleList([block() for _ in range(blocks_count)])
        self._norm = LayerNorm(d_model)

        self.save_probs = save_probs
        self.attn_probs = []

    def forward(self, inputs, mask):
        inputs = self._emb(inputs)
        self.attn_probs = []
        for block in self._blocks:
            inputs = block(inputs, mask)
            if self.save_probs:
                self.attn_probs.append(block._self_attn._attn_probs)

        return self._norm(inputs)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate, embeddings, out_layer, save_probs = False):
        super().__init__()
        self._out_layer = out_layer
        self._emb = embeddings

        block = lambda: DecoderLayer(
            size=d_model,
            self_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
            encoder_attn=MultiHeadedAttention(heads_count, d_model, dropout_rate),
            feed_forward=PositionwiseFeedForward(d_model, d_ff, dropout_rate),
            dropout_rate=dropout_rate
        )
        self._blocks = nn.ModuleList([block() for _ in range(blocks_count)])
        self._norm = LayerNorm(d_model)


        self.save_probs = save_probs
        self.self_attn_probs = []
        self.enc_attn_probs = []

    def forward(self, inputs, encoder_output, source_mask, target_mask):
        inputs = self._emb(inputs)
        self.self_attn_probs = []
        self.enc_attn_probs = []
        for block in self._blocks:
            inputs = block(inputs, encoder_output, source_mask, target_mask)
            if self.save_probs:
                self.self_attn_probs.append(block._self_attn._attn_probs)
                self.enc_attn_probs.append(block._encoder_attn._attn_probs)
        return self._out_layer(self._norm(inputs))


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture.
    """
    def __init__(self, source_vocab_size, target_vocab_size, d_model=256, d_ff=1024,
                 blocks_count=4, heads_count=8, dropout_rate=0.1, save_probs=False, use_shared_emb=False,
                 pretrained_embeddings=None):
        super(EncoderDecoder, self).__init__()

        self._out_layer = nn.Linear(d_model, target_vocab_size)
        self.d_model = d_model

        if pretrained_embeddings is not None:
            # Общий эмбеддинг-слой (для энкодера)
            self.embedding_layer = nn.Embedding(source_vocab_size, 300)
            self.embedding_layer.weight.data.copy_(pretrained_embeddings)
            self.embedding_layer.weight.requires_grad = True

            self.embedding_proj = nn.Linear(300, d_model)
            self._pos_enc = PositionalEncoding(d_model, dropout_rate)
            self.enc_emb = nn.Sequential(self.embedding_layer, self.embedding_proj, self._pos_enc)

            if use_shared_emb:
                self.dec_emb = self.enc_emb
                self._out_layer.weight = self.dec_emb[0].weight  # весовая привязка
            else:
                self.embedding_layer2 = nn.Embedding(target_vocab_size, 300)
                self.embedding_layer2.weight.data.copy_(pretrained_embeddings)
                self.embedding_layer2.weight.requires_grad = True

                self.embedding_proj2 = nn.Linear(300, d_model)
                self._pos_enc2 = PositionalEncoding(d_model, dropout_rate)
                self.dec_emb = nn.Sequential(self.embedding_layer2, self.embedding_proj2, self._pos_enc2)

        else:
            if use_shared_emb:
                self.enc_emb = nn.Sequential(
                    nn.Embedding(source_vocab_size, d_model),
                    PositionalEncoding(d_model, dropout_rate)
                )
                self.dec_emb = self.enc_emb
                self._out_layer.weight = self.dec_emb[0].weight  # весовая привязка
            else:
                self.enc_emb = nn.Sequential(
                    nn.Embedding(source_vocab_size, d_model),
                    PositionalEncoding(d_model, dropout_rate)
                )
                self.dec_emb = nn.Sequential(
                    nn.Embedding(target_vocab_size, d_model),
                    PositionalEncoding(d_model, dropout_rate)
                )

        self.encoder = Encoder(source_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate, self.enc_emb, save_probs)
        self.decoder = Decoder(target_vocab_size, d_model, d_ff, blocks_count, heads_count, dropout_rate, self.dec_emb, self._out_layer, save_probs)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_inputs, target_inputs, source_mask, target_mask):
        encoder_output = self.encoder(source_inputs, source_mask)
        return self.decoder(target_inputs, encoder_output, source_mask, target_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

