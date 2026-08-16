# synthfin-aml V9.1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valiyevoktay-cmd/synthfin-aml-/blob/main/examples/benchmark_tutorial.ipynb)
[![PyPI version](https://badge.fury.io/py/synthfin-aml.svg)](https://badge.fury.io/py/synthfin-aml)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-synthfin--aml-yellow)](https://huggingface.co/datasets/synthfin-aml)

A graph-native Anti-Money Laundering (AML) benchmark dataset.

## The Synthetic Leakage Problem

If you train a standard Gradient Boosting model (like LightGBM) on existing AML datasets (e.g. Elliptic, IBM), you often see a PR-AUC of 0.99+. 
This isn't because the model learned complex money laundering typologies. It's because of **synthetic leakage**: the simulated transaction amounts for fraudulent nodes have completely different statistical distributions than legitimate nodes. The model just splits on `amount` and ignores the graph structure entirely.

## What's New in V9.1

In V9.1, we calibrated the base distributions. A naive tabular model can no longer cheat by looking at raw transaction volumes. 

However, we embedded realistic AML typologies: **Structuring**. Fraudulent actors use high-frequency fan-out/fan-in patterns to dynamically split their transfers. To a tabular model evaluating a single transaction, these look identical to normal P2P or escrow economic activity. But topologically, they form distinct sub-graphs.

### Benchmark Results
Because the signal is now purely structural and temporal, the baseline metrics shift dramatically. We established a strict 3-tier benchmark:

*   **Level 1: LightGBM (Raw Tabular)**
    *   PR-AUC: **0.127** 
*   **Level 2: LightGBM + Explicit Graph Features (in/out degree, entropy)**
    *   PR-AUC: **0.703** 
    *   Inference Latency: **~1500 ms / 1k tx** (Severe feature-store lookup latency)
*   **Level 3: Temporal EdgeSAGE (End-to-End GNN)**
    *   PR-AUC: **0.865**
    *   Inference Latency: **1.3 ms / 1k tx**

**The Takeaway:** By engineering explicit graph features, tabular models can become competitive again (Level 2), but the heavy feature-store lookups cause massive production latency (~1500 ms). Our Temporal EdgeSAGE model not only beats it in accuracy (0.865 PR-AUC) but operates over **1000x faster**.

We also evaluate what actually matters in production AML systems:
*   **Precision@Top-500:** Because human investigation teams have limited daily bandwidth. Temporal EdgeSAGE dominates here.
*   **Latency (ms / 1000 tx):** Because real-time transaction blocking requires strict inference limits (1.3ms vs 1500ms).

## Quickstart

### 1. Run the Benchmark Tutorial
You can reproduce the GNN vs Tabular benchmark locally or on Colab.

```bash
pip install -r requirements.txt
jupyter notebook examples/benchmark_tutorial.ipynb
```

> **Note on Scale:** 
> The Polars feature extractor is currently optimized for graphs up to 50M nodes. For larger datasets, chunking is required to avoid OOM. PyTorch Geometric execution is tested and supported on Linux/WSL.

### 2. Loading the Static Datasets
Pre-generated temporal splits are available on Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("synthfin-aml", "small")
```
