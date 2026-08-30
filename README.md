# synthfin-aml V9.2.0

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

Existing synthetic AML datasets often contain temporal distribution leakage. Standard models exploit this by splitting on basic distributional artifacts (e.g., raw amount variances) rather than identifying underlying financial crime typologies. 

SynthFin-AML corrects this by flattening the base distribution for both normal and illicit activities (via identical KDE lognormal sampling). We programmatically embed specific topological AML patterns, such as Structuring and Smurfing. 

**Ablation Study (The Proof):**
* ❌ **Original Leaky Setup:** A standard LightGBM model reaches an inflated **0.99 PR-AUC**, entirely bypassing the graph structure.
* ✅ **Our Corrected V9.2 Setup:** When evaluated on our flattened distribution using only raw transaction features, the exact same LightGBM model collapses to **0.31 PR-AUC**. 

This mathematical drop forces any successful model to genuinely learn and leverage the graph topology, rather than exploiting tabular artifacts.

## Benchmark Results

Evaluations were conducted across 5 random seeds using a strict 80/20 transductive split. Note the difference between a naive tabular model and one augmented with engineered graph features.

| Model | Modality | PR-AUC | P@100 |
| :--- | :--- | :--- | :--- |
| LightGBM (Naive) | Tabular (Raw features only) | 0.3120 ± 0.0211 | 0.45 |
| LightGBM (Augmented) | Tabular (11 features incl. PageRank) | 0.8483 ± 0.0169 | 0.96 |
| PyTorch GraphSAGE | Graph | **0.8817 ± 0.0147** | **0.98** |

GraphSAGE demonstrates a statistically significant improvement over the best feature-engineered tabular baseline (p=0.000046), proving that native topological learning scales beyond manual feature extraction.

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
