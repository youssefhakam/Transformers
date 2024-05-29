import torch
import torch.nn as nn
import torch.nn.functional as f
import math

### Positional Encoding : 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100000.0)/d_model))
        pe[:, 0:2] = torch.sin(position * div_term)
        pe[:, 1:2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
### 
class Attention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model / num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        bs = q.size(0)
        # (batch_seq, len_seq, d_model) ----> (batch_seq, len_seq, num_heads, d_k)
        q = self.query(q).view(bs, -1, self.num_heads, self.d_k)
        k = self.key(k).view(bs, -1, self.num_heads, self.d_k)
        v = self.value(v).view(bs, -1, self.num_heads, self.d_k)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.nn.functional.softmax(scores, dim=-1)

        output = torch.matmul(attention, v)

        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        return self.fc(output), attention

### Create a MultiHead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.attention = Attention(d_model, num_heads)
        self.nomLayer = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        attn_out, attention = self.attention(q, k, v, mask)
        attn_out = self.nomLayer(attn_out + q)
        return attn_out, attention


### Create layer Feed Forward 
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff)
        self.layer2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.AddNorm = nn.LayerNorm(d_model)

    def forward(selx, x):
        x = self.layer1(x)
        x = f.relu(x)
        x = self.layer2(x)
        x = x + self.dropout(x)
        return self.AddNorm(x)

### Create a Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff) -> None:
        super(EncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)

    def forward(self, x):
        self.self_attention(x, mask= None)
        return self.feedforward(x)

### Create a Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff) -> None:
        super(DecoderBlock, self).__init()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, d_ff)
    def forward(self, x, enc_out, src_mask = None, tgt_mask = None):
        x, _ = self.self_attention(x, x, x, tgt_mak)
        x, _ = self.cross_attention(x, enc_out, enc_out, src_mask)
        return self.feedforward(x)


### Encoder

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
    def forward(self, src, mask = None):
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

### Decoder

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len) -> None:
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
    def forward(self, src, enc_out, src_mask = None, tgt_mask = None):
        x = self.embed(src) * math.sqrt(self.d_model)
        x = self.pos_decoder(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)


### Create a Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model , num_heads, d_ff, max_len, num_layers = 6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model,num_heads, d_ff, num_layers, max_len)
        self.deocder = Decoder(tgt_vocab_size, d_model,num_heads, d_ff, num_layers, max_len)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask = None, tgt_mask = None):
        enc_out = self.encoder(src, src_mask)
        dec_out = self.deocder(tgt, enc_out, src_mask, tgt_mask)
        return self.fc(dec_out)
