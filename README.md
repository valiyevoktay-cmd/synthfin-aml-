# synthfin-aml V9.1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valiyevoktay-cmd/synthfin-aml-/blob/main/examples/benchmark_tutorial.ipynb)
[![PyPI version](https://badge.fury.io/py/synthfin-aml.svg)](https://badge.fury.io/py/synthfin-aml)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-synthfin--aml-yellow)](https://huggingface.co/datasets/ovvaliyev/synthfin-aml)

## What is SynthFin-AML?

SynthFin-AML is a graph-native Anti-Money Laundering (AML) benchmark dataset. It represents a 10-day synthetic snapshot of transactional data between bank accounts. 

**Dataset Statistics:**
* Nodes: 100,000
* Edges: 1,273,403
* Task: Transductive node classification

The objective is to classify nodes into clean or fraudulent entities based on the transaction graph.

## Data Schema

### Nodes
Nodes represent individual bank accounts. Each node contains 10 features (topological and aggregational), all of which undergo a `log1p` transformation:
1. `initial_balance`
2. `out_degree`
3. `in_degree`
4. `out_volume`
5. `in_volume`
6. `out_max_amt`
7. `in_max_amt`
8. `nbr_in_volume`
9. `nbr_out_volume`
10. `pagerank`

### Edges
Edges represent directed financial transactions between accounts.
* Edge Features: `amount`

### Classes
* `0`: Clean
* `1`: Fraud
*(Note: The fraud ratio exhibits a high class imbalance, reflecting realistic financial crime distributions.)*

## The Synthetic Leakage Problem & Our Solution

Existing synthetic AML datasets often contain distribution leakage. Tabular models frequently exploit this by splitting on basic distributional artifacts (e.g., raw amount variances) rather than identifying underlying financial crime typologies. 

SynthFin-AML corrects this by fixing the base distribution for both normal and illicit activities. We programmatically embed specific topological AML patterns, such as Structuring and Smurfing. Consequently, models must learn the graph structure and transaction sequences to correctly identify fraudulent nodes, preventing tabular baseline exploitation.

## Benchmark Results

Evaluations were conducted across 5 random seeds using a strict 80/20 transductive split. 

| Model | Modality | PR-AUC | P@100 |
| :--- | :--- | :--- | :--- |
| LightGBM | Tabular (11 features incl. PageRank) | 0.8483 ± 0.0169 | 0.96 |
| PyTorch GraphSAGE | Graph | **0.8817 ± 0.0147** | **0.98** |

GraphSAGE demonstrates a statistically significant improvement over the tabular baseline (p=0.000046).

## Quickstart / PyG Wrapper

Install the dataset package via PyPI:
```bash
pip install synthfin-aml
```

The package provides a PyTorch Geometric (PyG) dataset wrapper that automatically downloads the graph from Hugging Face:

```python
from synthfin_aml_pkg import SynthFinDataset

# Initializes the dataset and downloads from Hugging Face if not cached
dataset = SynthFinDataset(root='./data')
data = dataset[0]

print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
```

To reproduce the benchmark results, refer to the provided script in the repository: `examples/reddit_benchmark.py`.

## License

This dataset and associated code are released under the MIT License. SynthFin-AML is 100% synthetic, contains no Personally Identifiable Information (PII), and is safe for both commercial and academic usage.
