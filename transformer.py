import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class Attention(nn.Module):
    def __init__(self, d_model, num_heads) -> None:
        super().__init__(Attention, self)
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
        super().__init__(MultiHeadAttention, self)
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




  
