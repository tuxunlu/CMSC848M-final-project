import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff, dropout):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )

    def forward(self, x):
        return self.ff(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dim_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Masked self-attention
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))
        # Cross-attention with encoder memory
        cross_attn_out, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))
        # Feed forward
        ff_out = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))
        return tgt


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_ff=2048, dropout=0.1, max_len=512):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dim_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # Embedding + Positional Encoding
        src = self.pos_enc(self.src_tok_emb(src))  # [B, S, D]
        tgt = self.pos_enc(self.tgt_tok_emb(tgt))  # [B, T, D]

        # Encoder
        memory = src
        for layer in self.encoder:
            memory = layer(memory, src_mask=src_mask)

        # Decoder
        output = tgt
        for layer in self.decoder:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        return self.fc_out(output)

if __name__ == '__main__':
    def generate_square_subsequent_mask(sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        return mask.masked_fill(mask == 1, float('-inf'))
    model = Transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_ff=1024,
        dropout=0.1
    )

    src = torch.randint(0, 10000, (32, 40))  # [batch, src_seq_len]
    tgt = torch.randint(0, 10000, (32, 30))  # [batch, tgt_seq_len]

    tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
    out = model(src, tgt, tgt_mask=tgt_mask)
    print(out.shape)  # [32, tgt_len, tgt_vocab_size]