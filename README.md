# synthfin-aml V9.1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valiyevoktay-cmd/synthfin-aml-/blob/main/examples/benchmark_tutorial.ipynb)
[![PyPI version](https://badge.fury.io/py/synthfin-aml.svg)](https://badge.fury.io/py/synthfin-aml)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-synthfin--aml-yellow)](https://huggingface.co/datasets/ovvaliyev/synthfin-aml)

A graph-native Anti-Money Laundering (AML) benchmark dataset.

## The Synthetic Leakage Problem

If you train a standard Gradient Boosting model (like LightGBM) on existing AML datasets (e.g. Elliptic, IBM), you often see a PR-AUC of 0.99+. 
This isn't because the model learned complex money laundering typologies. It's because of **synthetic leakage**: the simulated transaction amounts for fraudulent nodes have completely different statistical distributions than legitimate nodes. The model just splits on `amount` and ignores the graph structure entirely.

## What's New in V9.1

In V9.1, we calibrated the base distributions. A naive tabular model can no longer cheat by looking at raw transaction volumes. 

However, we embedded realistic AML typologies: **Structuring**. Fraudulent actors use high-frequency fan-out/fan-in patterns to dynamically split their transfers. To a tabular model evaluating a single transaction, these look identical to normal P2P or escrow economic activity. But topologically, they form distinct sub-graphs.

### Benchmark Results
To ensure maximum rigor and eliminate temporal look-ahead bias, we formulate this as a strict **Transductive Node Classification** task on a static 10-day snapshot. Both baseline and GNN models see the exact same 80/20 node split, ensuring zero leakage advantages.

We established a strict benchmark evaluated over **5 random seeds**, validated by a paired t-test (p = 0.000046):

*   **LightGBM (Tuned + 11 Features + Weighted PageRank)**
    *   PR-AUC: **0.8483 ± 0.0169** 
    *   Precision@100: **0.96 ± 0.01**
*   **Pure PyTorch GraphSAGE (40-line implementation)**
    *   PR-AUC: **0.8817 ± 0.0147**
    *   Precision@100: **0.98 ± 0.01**

**The Takeaway:** Even when providing LightGBM with explicit topological features (like Weighted PageRank and neighbor aggregates) and extensive hyperparameter tuning, the end-to-end GraphSAGE architecture maintains a statistically significant performance advantage in detecting structured AML patterns.

We also evaluate what actually matters in production AML systems:
*   **Precision@Top-100:** Because human investigation teams have limited daily bandwidth. GraphSAGE dominates here.

## Quickstart

### 1. Run the Benchmark Tutorial
You can reproduce the GNN vs Tabular benchmark locally or on Colab.

```bash
pip install -r requirements.txt
jupyter notebook examples/benchmark_tutorial.ipynb
# Or run the benchmark script directly
python examples/reddit_benchmark.py
```

> **Note on Scale:** 
> The Polars feature extractor is currently optimized for graphs up to 50M nodes. For larger datasets, chunking is required to avoid OOM. PyTorch Geometric execution is tested and supported on Linux/WSL.

### 2. PyTorch Geometric Wrapper (New in V9.1)
You can directly load the pre-generated benchmark dataset (100k nodes) into PyTorch Geometric in a single line. The dataset is hosted on Hugging Face (`ovvaliyev/synthfin-aml`) and automatically downloads and processes the explicit structural features (PageRank, neighbor aggregates).

```python
# pip install synthfin-aml
from synthfin_aml_pkg import SynthFinDataset

# Automatically downloads from Hugging Face and builds the PyG graph
dataset = SynthFinDataset(root='./data')
data = dataset[0]

print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
```
