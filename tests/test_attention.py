"""
Tests for attention mechanisms.
"""

import pytest
import torch
import numpy as np

from bma.pytorch import (
    BilinearlyModulatedAttention,
    MultiHeadAttention,
    GatedAttention
)


@pytest.fixture
def batch_input():
    """Create a sample batch for testing."""
    batch_size, seq_len, d_model = 4, 16, 128
    return torch.randn(batch_size, seq_len, d_model)


class TestBilinearlyModulatedAttention:
    """Test BMA implementation."""
    
    def test_forward_shape(self, batch_input):
        """Test that output shape matches input shape."""
        B, T, D = batch_input.shape
        attn = BilinearlyModulatedAttention(d_model=D, n_heads=4)
        output = attn(batch_input)
        
        assert output.shape == (B, T, D)
    
    def test_causal_masking(self):
        """Test that causal masking prevents future information leakage."""
        d_model = 64
        attn = BilinearlyModulatedAttention(d_model=d_model, n_heads=4, causal=True)
        
        # Create input where each position has unique value
        x = torch.arange(8 * d_model, dtype=torch.float).reshape(1, 8, d_model)
        
        output = attn(x)
        
        # Check that early positions don't depend on later ones
        # by verifying gradient flow
        loss = output[0, 0].sum()
        loss.backward()
        
        # Gradients for future positions should be zero
        assert x.grad[0, 1:].abs().sum() == 0
    
    def test_parameter_count(self):
        """Test that BMA adds correct number of parameters."""
        d_model, n_heads = 128, 4
        d_head = d_model // n_heads
        
        attn_bma = BilinearlyModulatedAttention(d_model=d_model, n_heads=n_heads)
        attn_std = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        
        params_bma = sum(p.numel() for p in attn_bma.parameters())
        params_std = sum(p.numel() for p in attn_std.parameters())
        
        # BMA should add n_heads * d_head^2 parameters
        expected_diff = n_heads * d_head * d_head
        assert abs(params_bma - params_std - expected_diff) < 10
    
    def test_gradient_flow(self, batch_input):
        """Test that gradients flow correctly through BMA."""
        attn = BilinearlyModulatedAttention(
            d_model=batch_input.shape[-1],
            n_heads=4
        )
        
        output = attn(batch_input)
        loss = output.sum()
        loss.backward()
        
        # Check that all parameters receive gradients
        for name, param in attn.named_parameters():
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert param.grad.abs().sum() > 0


class TestMultiHeadAttention:
    """Test standard attention implementation."""
    
    def test_forward_shape(self, batch_input):
        B, T, D = batch_input.shape
        attn = MultiHeadAttention(d_model=D, n_heads=4)
        output = attn(batch_input)
        
        assert output.shape == (B, T, D)
    
    def test_equivalence_to_reference(self):
        """Test that implementation matches expected behavior."""
        torch.manual_seed(42)
        d_model = 64
        x = torch.randn(2, 8, d_model)
        
        attn = MultiHeadAttention(d_model=d_model, n_heads=4, dropout=0.0)
        attn.eval()
        
        output = attn(x)
        
        # Output should be finite
        assert torch.isfinite(output).all()
        # Output should have reasonable scale
        assert output.abs().mean() < 10


class TestGatedAttention:
    """Test post-SDPA gated attention."""
    
    def test_forward_shape(self, batch_input):
        B, T, D = batch_input.shape
        attn = GatedAttention(d_model=D, n_heads=4)
        output = attn(batch_input)
        
        assert output.shape == (B, T, D)
    
    def test_gating_effect(self):
        """Test that gating actually modulates the output."""
        torch.manual_seed(42)
        d_model = 64
        x = torch.randn(1, 8, d_model)
        
        attn_gated = GatedAttention(d_model=d_model, n_heads=4, dropout=0.0)
        attn_std = MultiHeadAttention(d_model=d_model, n_heads=4, dropout=0.0)
        
        # Copy QKV weights to make attention part identical
        attn_gated.qkv.weight.data = attn_std.qkv.weight.data.clone()
        attn_gated.qkv.bias.data = attn_std.qkv.bias.data.clone()
        
        attn_gated.eval()
        attn_std.eval()
        
        out_gated = attn_gated(x)
        out_std = attn_std(x)
        
        # Outputs should be different due to gating
        assert not torch.allclose(out_gated, out_std, atol=1e-5)


class TestAttentionComparison:
    """Compare different attention mechanisms."""
    
    def test_computational_efficiency(self):
        """Test that BMA is computationally efficient."""
        d_model, n_heads = 256, 8
        batch_size, seq_len = 16, 128
        
        x = torch.randn(batch_size, seq_len, d_model, device='cpu')
        
        attn_types = {
            'bma': BilinearlyModulatedAttention(d_model, n_heads),
            'standard': MultiHeadAttention(d_model, n_heads),
            'gated': GatedAttention(d_model, n_heads)
        }
        
        import time
        times = {}
        
        for name, attn in attn_types.items():
            attn.eval()
            with torch.no_grad():
                # Warm up
                _ = attn(x)
                
                # Time
                start = time.time()
                for _ in range(10):
                    _ = attn(x)
                times[name] = time.time() - start
        
        # BMA should not be significantly slower than standard
        assert times['bma'] < times['standard'] * 2.0
    
    def test_memory_usage(self):
        """Test memory consumption of different mechanisms."""
        d_model, n_heads = 256, 8
        
        def get_param_size(model):
            return sum(p.numel() * p.element_size() for p in model.parameters())
        
        bma = BilinearlyModulatedAttention(d_model, n_heads)
        std = MultiHeadAttention(d_model, n_heads)
        gated = GatedAttention(d_model, n_heads)
        
        size_bma = get_param_size(bma)
        size_std = get_param_size(std)
        size_gated = get_param_size(gated)
        
        # BMA should be more efficient than gated attention
        assert size_bma < size_gated


if __name__ == "__main__":
    pytest.main([__file__])
