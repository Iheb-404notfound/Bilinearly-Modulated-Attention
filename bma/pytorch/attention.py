"""
PyTorch implementation of Bilinearly Modulated Attention.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class BilinearlyModulatedAttention(nn.Module):
    """
    Bilinearly Modulated Attention (BMA) mechanism.
    
    Applies query-conditioned value gating through bilinear transformations
    before attention aggregation, offering improved expressiveness while
    maintaining stable optimization.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
        causal: Whether to apply causal masking (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        
        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model, bias=bias)
        
        # Per-head gating matrices
        self.W_g = nn.Parameter(
            torch.randn(n_heads, self.d_head, self.d_head) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of bilinearly modulated attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional attention mask of shape (batch, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        B, T, D = x.shape
        
        # QKV projection
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # (B, heads, T, d_head)
        
        # Standard attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # (B, heads, T, T)
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(T, T, device=x.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        
        # Apply custom mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        # Attention weights
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Query-conditioned value gating
        # g = sigmoid(Q @ W_g)
        g = torch.sigmoid(
            torch.einsum("bhtd,hde->bhte", q, self.W_g)
        )
        # (B, heads, T, d_head)
        
        # Modulate values
        v = g * v
        
        # Aggregation
        out = attn @ v  # (B, heads, T, d_head)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        
        return out


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention for comparison.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
        causal: Whether to apply causal masking (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(T, T, device=x.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out(out)
        
        return out


class GatedAttention(nn.Module):
    """
    Post-SDPA Gated Attention (baseline from NeurIPS 2025).
    
    Applies gating after attention aggregation as in the Gated Attention paper.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in projections (default: True)
        causal: Whether to apply causal masking (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.causal = causal
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out = nn.Linear(d_model, d_model, bias=bias)
        
        # Post-attention gating
        self.gate = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if self.causal:
            causal_mask = torch.tril(
                torch.ones(T, T, device=x.device, dtype=torch.bool)
            )
            scores = scores.masked_fill(~causal_mask, float("-inf"))
        
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        
        # Post-aggregation gating
        g = torch.sigmoid(self.gate(x))
        out = g * out
        
        out = self.out(out)
        
        return out
