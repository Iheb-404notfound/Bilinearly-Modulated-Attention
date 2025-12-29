# Bilinearly Modulated Attention (BMA)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)

A novel attention mechanism that introduces query-conditioned value gating through bilinear transformations, offering improved expressiveness with minimal parameter overhead compared to standard attention and post-SDPA gating approaches.

## Overview

Bilinearly Modulated Attention introduces a theoretically motivated alternative to recent gating mechanisms in attention layers. Rather than gating the attention output after aggregation, BMA applies query-conditioned value gating before aggregation, preserving softmax geometry while adding query-aware feature selection.

### Added Value

The mechanism provides several advantages over standard attention mechanisms. It enables query-aware filtering where each token filters values based on contextual needs, creating richer interactions through bilinear maps that generate d²ₕ feature combinations compared to dₕ scalar gates in simpler approaches. The design maintains stable optimization by leaving softmax logits untouched, while improved gradient flow allows gates to receive signals from both query representations and final loss. Information flows more cleanly as values are gated before mixing rather than after, and each attention head learns specialized gating patterns for different query types.

### Architecture

The mathematical formulation follows standard transformer projections with learned weight matrices for queries, keys, and values. Attention weights are computed using the standard scaled dot-product mechanism. The key innovation lies in query-conditioned value gating, where a per-head gating matrix Wg transforms queries to produce gates G = σ(Q·Wg), and modulated values are computed as Ṽ = G ⊙ V. Finally, output aggregation proceeds with O = A·Ṽ followed by concatenation and projection across heads.

The parameter efficiency is notable, adding only H·d²ₕ parameters where H is the number of heads and dₕ is the head dimension. For a typical configuration with eight heads and head dimension of 64, this adds approximately 32,768 parameters per layer, which is negligible for large models.

## Installation

You can install the package directly from the repository or set up a development environment.

### From Source

```bash
git clone https://github.com/yourusername/bilinearly-modulated-attention.git
cd bilinearly-modulated-attention
pip install -e .
```

### Dependencies

Core dependencies include PyTorch 2.0 or higher, JAX 0.4 or higher with jaxlib, NumPy 1.20 or higher, and einops 0.6 or higher for tensor operations.

For development and examples, you will also need datasets from Hugging Face, tokenizers, matplotlib for visualization, and pytest for testing.

## Quick Start

### PyTorch Implementation

```python
import torch
from bma.pytorch import BilinearlyModulatedAttention, TransformerBlock

# Create attention layer
attention = BilinearlyModulatedAttention(
    d_model=512,
    n_heads=8,
    dropout=0.1
)

# Forward pass
batch_size, seq_len, d_model = 16, 128, 512
x = torch.randn(batch_size, seq_len, d_model)
output = attention(x)

# Use in transformer block
block = TransformerBlock(
    d_model=512,
    n_heads=8,
    d_ff=2048,
    dropout=0.1
)
output = block(x)
```

### JAX Implementation

```python
import jax
import jax.numpy as jnp
from bma.jax import BilinearlyModulatedAttention, init_attention

# Initialize parameters
rng = jax.random.PRNGKey(0)
params = init_attention(
    rng,
    d_model=512,
    n_heads=8
)

# Create attention layer
attention = BilinearlyModulatedAttention(n_heads=8)

# Forward pass
batch_size, seq_len, d_model = 16, 128, 512
x = jnp.ones((batch_size, seq_len, d_model))
output = attention.apply(params, x)
```

## Examples

The repository includes several complete examples demonstrating the mechanism in different contexts.

### Language Modeling

The language modeling example trains a small transformer language model on WikiText-2 using BMA, comparing performance against standard attention and post-SDPA gating baselines. You can run this example using the provided script in the examples directory.

### Vision Transformer

The vision transformer example implements a ViT-style model with BMA for image classification on CIFAR-10, demonstrating the mechanism's effectiveness in computer vision tasks. The implementation includes patch embedding, positional encoding, and classification head components.

### Benchmark Comparison

A comprehensive benchmark script compares BMA against standard attention and gated attention across multiple metrics including perplexity, downstream task performance on MMLU and GSM8k, training stability measured by loss variance, parameter count efficiency, and memory usage during training and inference.

## Preliminary Results

Initial experiments on a 35 million parameter model trained on 400 billion tokens show promising results. BMA achieves a perplexity of 216.2, compared to 221.1 for standard attention and 217.49 for post-SDPA gating. The mechanism accomplishes these improvements with only 1 million additional parameters compared to 4.1 million for post-SDPA gating, demonstrating four times better parameter efficiency. Training stability remains comparable to gated attention with no loss spikes observed during training.

## Project Structure

The repository is organized into several key directories. The bma directory contains core implementations with separate subdirectories for PyTorch and JAX versions, along with shared utilities. The examples directory provides complete training scripts for language modeling, vision transformers, and benchmarking. Tests ensure correctness across implementations, and notebooks contain experimental code and preliminary analyses. Documentation provides detailed API references and theoretical background.

## Development

To set up the development environment, clone the repository and install dependencies including development requirements. Run tests using pytest to ensure all implementations work correctly. Code formatting follows Black style for Python code and is enforced through pre-commit hooks.

## Citation

If you use this code in your research, please cite the work as follows:

```bibtex
@misc{gafsi2025bma,
  title={Bilinearly Modulated Attention: Query-Conditioned Value Gating},
  author={Gafsi, Iheb},
  year={2025},
  howpublished={\\url{https://github.com/yourusername/bilinearly-modulated-attention}}
}
```

## Related Work

This work builds upon recent advances in gating mechanisms for attention layers, particularly Gated Attention presented at NeurIPS 2025, which demonstrated the effectiveness of gating mechanisms applied after scaled dot-product attention output.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Special thanks to Professor Rabaa Youssef for guidance on this research. The project was inspired by the Gated Attention mechanism and aims to provide a theoretically motivated alternative with improved parameter efficiency.

## Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact the author directly through the repository.
