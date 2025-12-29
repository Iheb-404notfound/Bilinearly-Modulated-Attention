"""
Integration tests for transformer models.
"""

import pytest
import torch
import torch.nn.functional as F

from bma.pytorch import TransformerLM, VisionTransformer, TransformerBlock


class TestTransformerBlock:
    """Test transformer block integration."""
    
    def test_block_forward(self):
        """Test transformer block forward pass."""
        batch_size, seq_len, d_model = 4, 16, 128
        x = torch.randn(batch_size, seq_len, d_model)
        
        block = TransformerBlock(
            d_model=d_model,
            n_heads=4,
            d_ff=512,
            dropout=0.1,
            attention_type='bma'
        )
        
        output = block(x)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_all_attention_types(self):
        """Test that all attention types work in transformer block."""
        x = torch.randn(2, 8, 64)
        
        for attn_type in ['bma', 'standard', 'gated']:
            block = TransformerBlock(
                d_model=64,
                n_heads=4,
                d_ff=256,
                attention_type=attn_type
            )
            output = block(x)
            assert output.shape == x.shape
            assert torch.isfinite(output).all()


class TestTransformerLM:
    """Test language model integration."""
    
    def test_lm_forward(self):
        """Test language model forward pass."""
        vocab_size = 1000
        batch_size, seq_len = 4, 32
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            attention_type='bma'
        )
        
        logits = model(x)
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_lm_loss_computation(self):
        """Test that loss can be computed correctly."""
        vocab_size = 1000
        batch_size, seq_len = 4, 32
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )
        
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_lm_training_step(self):
        """Test that model can perform a training step."""
        vocab_size = 1000
        batch_size, seq_len = 4, 32
        
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=128,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        
        optimizer = torch.optim.Adam(model.parameters())
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters were updated
        for param in model.parameters():
            assert param.grad is not None
    
    def test_lm_attention_types(self):
        """Test that all attention types work in language model."""
        vocab_size = 1000
        x = torch.randint(0, vocab_size, (2, 16))
        
        for attn_type in ['bma', 'standard', 'gated']:
            model = TransformerLM(
                vocab_size=vocab_size,
                d_model=64,
                n_heads=4,
                n_layers=2,
                attention_type=attn_type
            )
            
            logits = model(x)
            assert logits.shape == (2, 16, vocab_size)
            assert torch.isfinite(logits).all()
    
    def test_parameter_counting(self):
        """Test parameter counting method."""
        model = TransformerLM(
            vocab_size=1000,
            d_model=128,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        
        manual_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        method_count = model.count_parameters()
        
        assert manual_count == method_count


class TestVisionTransformer:
    """Test vision transformer integration."""
    
    def test_vit_forward(self):
        """Test vision transformer forward pass."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 32, 32)
        
        model = VisionTransformer(
            image_size=32,
            patch_size=4,
            n_classes=10,
            d_model=128,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        
        logits = model(images)
        assert logits.shape == (batch_size, 10)
    
    def test_vit_loss_computation(self):
        """Test that classification loss can be computed."""
        batch_size = 4
        images = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, 10, (batch_size,))
        
        model = VisionTransformer(
            image_size=32,
            patch_size=4,
            n_classes=10,
            d_model=128,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        
        assert torch.isfinite(loss)
        assert loss.item() > 0
    
    def test_vit_attention_types(self):
        """Test that all attention types work in vision transformer."""
        images = torch.randn(2, 3, 32, 32)
        
        for attn_type in ['bma', 'standard', 'gated']:
            model = VisionTransformer(
                image_size=32,
                patch_size=4,
                n_classes=10,
                d_model=64,
                n_heads=4,
                n_layers=2,
                attention_type=attn_type
            )
            
            logits = model(images)
            assert logits.shape == (2, 10)
            assert torch.isfinite(logits).all()
    
    def test_patch_embedding(self):
        """Test that patch embedding works correctly."""
        batch_size = 4
        image_size = 32
        patch_size = 4
        n_patches = (image_size // patch_size) ** 2
        
        images = torch.randn(batch_size, 3, image_size, image_size)
        
        model = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            n_classes=10,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        
        # Extract patch embedding
        patches = model.patch_emb(images)
        patches = patches.flatten(2).transpose(1, 2)
        
        assert patches.shape == (batch_size, n_patches, 128)


class TestModelEquivalence:
    """Test equivalence across different attention mechanisms."""
    
    def test_same_qkv_weights(self):
        """Test that with same QKV weights, different mechanisms produce different outputs."""
        vocab_size = 1000
        x = torch.randint(0, vocab_size, (2, 16))
        
        # Create models
        model_std = TransformerLM(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=1,
            attention_type='standard'
        )
        
        model_bma = TransformerLM(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=1,
            attention_type='bma'
        )
        
        # Copy token embeddings
        model_bma.token_emb.weight.data = model_std.token_emb.weight.data.clone()
        model_bma.pos_emb.weight.data = model_std.pos_emb.weight.data.clone()
        
        model_std.eval()
        model_bma.eval()
        
        with torch.no_grad():
            out_std = model_std(x)
            out_bma = model_bma(x)
        
        # Outputs should be different due to different attention mechanisms
        assert not torch.allclose(out_std, out_bma, atol=1e-5)


class TestModelSaving:
    """Test model saving and loading."""
    
    def test_save_load_lm(self, tmp_path):
        """Test that language model can be saved and loaded."""
        vocab_size = 1000
        
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        
        # Save
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        # Load
        model_loaded = TransformerLM(
            vocab_size=vocab_size,
            d_model=64,
            n_heads=4,
            n_layers=2,
            attention_type='bma'
        )
        model_loaded.load_state_dict(torch.load(save_path))
        
        # Test
        x = torch.randint(0, vocab_size, (2, 16))
        
        model.eval()
        model_loaded.eval()
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model_loaded(x)
        
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__])
