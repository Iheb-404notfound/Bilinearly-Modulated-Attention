from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bilinearly-modulated-attention",
    version="0.1.0",
    author="Iheb Gafsi",
    author_email="your.email@example.com",
    description="Query-conditioned value gating for transformer attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bilinearly-modulated-attention",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "einops>=0.6.0",
    ],
    extras_require={
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "flax>=0.6.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ],
        "examples": [
            "datasets>=2.0.0",
            "tokenizers>=0.13.0",
            "matplotlib>=3.5.0",
            "tqdm>=4.64.0",
            "wandb>=0.13.0",
        ],
    },
)
