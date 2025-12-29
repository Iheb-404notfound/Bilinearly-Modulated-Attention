"""
Transformer building blocks using BMA.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

from .attention import (
    BilinearlyModulatedAttention,
    MultiHeadAttention,
    GatedAttention
)


class TransformerBlock(nn.Module):
    """
    Transformer block with configurable attention mechanism.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feedforward dimension
        dropout: Dropout probability
        attention_type: Type of attention to use ('bma', 'standard', 'gated')
        activation: Activation function ('gelu', 'relu')
        causal: Whether to use causal masking
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_type: Literal['bma', 'standard', 'gated'] = 'bma',
        activation: str = 'gelu',
        causal: bool = True
    ):
        super().__init__()
        
        # Select attention mechanism
        if attention_type == 'bma':
            self.attn = BilinearlyModulatedAttention(
                d_model, n_heads, dropout, causal=causal
            )
        elif attention_type == 'standard':
            self.attn = MultiHeadAttention(
                d_model, n_heads, dropout, causal=causal
            )
        elif attention_type == 'gated':
            self.attn = GatedAttention(
                d_model, n_heads, dropout, causal=causal
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Feedforward network
        act_fn = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with pre-norm architecture.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Attention block with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        
        # Feedforward block with residual
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x


class TransformerLM(nn.Module):
    """
    Transformer language model with configurable attention.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        attention_type: Type of attention mechanism
        tie_embeddings: Whether to tie input/output embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        attention_type: Literal['bma', 'standard', 'gated'] = 'bma',
        tie_embeddings: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                attention_type=attention_type
            )
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings if requested
        if tie_embeddings:
            self.lm_head.weight = self.token_emb.weight
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for language modeling.
        
        Args:
            x: Input token IDs (batch, seq_len)
            mask: Optional attention mask
            
        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        B, T = x.shape
        
        # Create position IDs
        pos = torch.arange(T, device=x.device)
        
        # Embed tokens and positions
        h = self.token_emb(x) + self.pos_emb(pos)
        
        # Apply transformer layers
        for layer in self.layers:
            h = layer(h, mask)
        
        # Final normalization and projection
        h = self.norm(h)
        logits = self.lm_head(h)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VisionTransformer(nn.Module):
    """
    Vision Transformer with configurable attention mechanism.
    
    Args:
        image_size: Input image size
        patch_size: Size of image patches
        n_classes: Number of output classes
        in_channels: Number of input channels
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feedforward dimension
        dropout: Dropout probability
        attention_type: Type of attention mechanism
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        n_classes: int = 1000,
        in_channels: int = 3,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        attention_type: Literal['bma', 'standard', 'gated'] = 'bma'
    ):
        super().__init__()
        
        assert image_size % patch_size == 0
        n_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_emb = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embeddings
        self.pos_emb = nn.Parameter(
            torch.randn(1, n_patches + 1, d_model) * 0.02
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer layers (no causal masking for vision)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                attention_type=attention_type,
                causal=False
            )
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for image classification.
        
        Args:
            x: Input images (batch, channels, height, width)
            
        Returns:
            Class logits (batch, n_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_emb(x)  # (B, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_emb
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Extract CLS token and classify
        x = self.norm(x[:, 0])
        x = self.head(x)
        
        return x
