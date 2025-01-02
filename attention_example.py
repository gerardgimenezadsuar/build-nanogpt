import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import time
import inspect

# Let's work with a tiny example:
# Batch size (B) = 3
# Sequence length (T) = 4 
# Embedding dimension (n_embd) = 6
# Number of heads (n_head) = 2

# Sample input: imagine these are token embeddings for 3 different sequences
x = torch.tensor([
    [[1.0, 0.5, -0.2, 0.3, 0.1, -0.1],  # Sequence 1, token 1
     [0.2, 0.8, 0.3, -0.5, 0.4, 0.2],   # Sequence 1, token 2
     [-0.4, 0.1, 0.7, 0.2, -0.3, 0.5],  # Sequence 1, token 3
     [0.3, -0.2, 0.4, 0.1, 0.6, -0.3]], # Sequence 1, token 4
    
    [[0.5, 0.2, 0.1, -0.3, 0.4, 0.2],   # Sequence 2
     [-0.1, 0.6, 0.3, 0.2, -0.2, 0.4],
     [0.3, -0.4, 0.5, 0.1, 0.3, -0.1],
     [0.2, 0.3, -0.1, 0.4, 0.5, 0.2]],
    
    [[0.4, 0.1, -0.3, 0.5, 0.2, 0.3],   # Sequence 3
     [-0.2, 0.4, 0.2, 0.1, 0.6, -0.1],
     [0.3, 0.5, 0.1, -0.2, 0.4, 0.2],
     [0.1, -0.3, 0.4, 0.3, 0.2, 0.5]]
], dtype=torch.float32)

class SimpleAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        
        # Single projection matrix for Q, K, V
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        
    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim
        
        # 1. Compute Q, K, V projections
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each is (B, T, C)
        
        # 2. Split into heads and rearrange
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # 3. Compute attention scores
        scores = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        
        # 4. Apply causal mask (optional for this example)
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        # 5. Apply softmax
        attn = F.softmax(scores, dim=-1)  # (B, nh, T, T)
        
        # 6. Apply attention to values
        out = attn @ v  # (B, nh, T, hs)
        
        # 7. Reshape and project output
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        out = self.c_proj(out)
        
        return out

# Initialize and test the attention module
attention = SimpleAttention(n_embd=6, n_head=2)

# Forward pass
output = attention(x)
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# Let's look at q, k, v for both heads
with torch.no_grad():
    qkv = attention.c_attn(x)
    q, k, v = qkv.split(6, dim=2)
    
    # Split into heads
    q = q[0:1].view(1, -1, 2, 3)  # [batch=1, seq_len, n_head=2, head_size=3]
    k = k[0:1].view(1, -1, 2, 3)
    v = v[0:1].view(1, -1, 2, 3)
    
    print("\nQuery values:")
    print("Head 1:", q[0, :, 0].round(decimals=2))  # First head
    print("Head 2:", q[0, :, 1].round(decimals=2))  # Second head
    
    print("\nKey values:")
    print("Head 1:", k[0, :, 0].round(decimals=2))
    print("Head 2:", k[0, :, 1].round(decimals=2))
    
    print("\nValue values:") 
    print("Head 1:", v[0, :, 0].round(decimals=2))
    print("Head 2:", v[0, :, 1].round(decimals=2))