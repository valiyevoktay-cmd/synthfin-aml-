# synthfin-aml (v10.0)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/valiyevoktay-cmd/synthfin-aml-/blob/main/examples/benchmark_tutorial.ipynb)
[![PyPI version](https://badge.fury.io/py/synthfin-aml.svg)](https://badge.fury.io/py/synthfin-aml)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-synthfin--aml-yellow)](https://huggingface.co/datasets/ovvaliyev/synthfin-aml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SynthFin-AML** is a graph-native Anti-Money Laundering (AML) benchmark and generation engine designed to evaluate Graph Neural Networks (GNNs) and Tabular Gradient Boosted Decision Trees (GBDTs) under strict temporal causality.

It models a dynamic 10-day payment network comprising **100,000 accounts** and **1,273,403 transactions** with embedded financial crime typologies (Structuring, Smurfing, and Multi-Hop Layering).

---

## 1. Problem Formulation: Synthetic Distribution Leakage

Most synthetic fraud and AML benchmarks suffer from **marginal distribution leakage**: fraudulent transaction amounts or account opening balances are generated from distinct statistical distributions compared to legitimate traffic. Under this setup, a standard LightGBM or XGBoost model achieves $>0.98$ PR-AUC simply by splitting on raw transaction amounts or static account balances, rendering graph topology and temporal sequence completely irrelevant.

SynthFin-AML resolves this by calibrating background legitimate traffic (P2P, Merchants, High-Liquidity Brokers, Escrow services) and illicit money laundering chains to identical lognormal amount distributions ($\mu = 8.517, \sigma = 0.8$). Illicit behavior is embedded strictly into the **joint transaction transition tensor** $P(u \to v \to w) \neq P(u \to v)P(v \to w)$ and multi-hop network topologies.

```
Tabular View (1 Transaction):
[ Account A ] ─── $9,850 (Normal Distribution) ───► [ Account B ]
Result: Indistinguishable from retail/broker transfers.

Topological View (Sub-Graph):
[ Source ] ─┬─► [ Mule 1 ] ──► [ Layering 1 ] ─┬─► [ Integration / Broker ]
            ├─► [ Mule 2 ] ──► [ Layering 2 ] ─┤
            └─► [ Mule 3 ] ──► [ Layering 3 ] ─┘
Result: Coordinated multi-hop fan-out / fan-in structuring identifiable via GNN message passing.
```

---

## 2. 3-Snapshot Inductive Temporal Architecture

In real-world financial networks, transaction ledgers are continuous streams. Standard random train/val/test node splits on a static graph cause severe **temporal look-ahead leakage**: training node embeddings aggregate future transaction information (e.g., Day 10 edges) to classify past states (e.g., Day 2).

SynthFin-AML implements an **Inductive Temporal Snapshot Architecture** across 3 strictly isolated subgraphs:

```
Timeline: Day 1 ──── Day 7 ──────── Day 8 ────────── Day 10
          |               |                 |                    |
          ├───────────────┤                 |                    |
          │  dataset[0]   │                 |                    |
          │  (Train Graph)│                 |                    |
          │  Edges <= Day 7                 |                    |
          │  Loss: train_mask (80k nodes)   |                    |
          ├─────────────────────────────────┤                    |
          │          dataset[1]             │                    |
          │          (Val Graph)            │                    |
          │          Edges <= Day 8         │                    |
          │          Loss: val_mask (10k nodes)                  |
          ├──────────────────────────────────────────────────────┤
          │                     dataset[2]                       │
          │                     (Test Graph)                     │
          │                     Edges <= Day 10                  │
          │                     Loss: test_mask (10k nodes)      │
```

### Leakage Prevention Guarantees:
1. **Structural Isolation:** `dataset[0]` contains strictly edges with `edge_time <= 7`. Messages cannot propagate from future transactions.
2. **Point-in-Time Feature Extraction:** All 10 node features (degrees, volumes, PageRank) are computed strictly on the active historical subgraph for each snapshot.
3. **Disjoint Target Masks:** `train_mask` (80%), `val_mask` (10%), and `test_mask` (10%) are mutually exclusive node partitions. The model never computes loss on validation or test account labels during training.
4. **Standardization Fit:** Feature scaling parameters (mean $\mu$, std $\sigma$) are fit strictly on `dataset[0]` (Train) and applied downstream.

---

## 3. Dataset Specifications & Schemas

### Overview Statistics
* **Accounts (Nodes):** 100,000
* **Transactions (Edges, Full 10-day horizon):** 1,273,403
* **Train Subgraph (Days 1–7):** 787,307 edges
* **Validation Subgraph (Days 1–8):** 963,552 edges
* **Test Subgraph (Days 1–10):** 1,273,403 edges
* **Node Feature Dimension:** 10 continuous features (`log1p` transformed, standardized)
* **Target Classes:** Binary (`0`: Clean, `1`: Fraud / Money Laundering Mule)
* **Class Imbalance:** ~2.0% positive rate (stratified across splits)

### Node Schema (`nodes.csv`)
| Column | Type | Physical Description / Derivation |
| :--- | :--- | :--- |
| `agent_id` | `int64` | Unique account identifier (`0` to `99,999`). |
| `initial_balance` | `float64` | Opening ledger balance at $t=0$ (USD). |
| `out_degree` | `int64` | Number of outgoing transfers in the active time window. |
| `in_degree` | `int64` | Number of incoming transfers in the active time window. |
| `out_volume` | `float64` | Cumulative outgoing monetary volume (USD). |
| `in_volume` | `float64` | Cumulative incoming monetary volume (USD). |
| `out_max_amt` | `float64` | Maximum single outgoing transfer amount (USD). |
| `in_max_amt` | `float64` | Maximum single incoming transfer amount (USD). |
| `nbr_in_volume` | `float64` | Mean incoming volume of 1-hop outgoing neighbor accounts. |
| `nbr_out_volume` | `float64` | Mean outgoing volume of 1-hop incoming neighbor accounts. |
| `pagerank` | `float64` | Amount-weighted directed PageRank score ($d = 0.85$). |
| `is_fraud` | `int64` | Ground-truth label (`0`: Clean, `1`: Money Laundering). |

### Edge Schema (`edges.csv`)
| Column | Type | Physical Description |
| :--- | :--- | :--- |
| `source_id` | `int64` | Originating account ID. |
| `target_id` | `int64` | Beneficiary account ID. |
| `timestamp` | `datetime64` | Exact UTC timestamp of transfer execution. |
| `edge_time` | `int64` | Discrete day offset from simulation start (`1` to `10`). |
| `amount` | `float64` | Transaction monetary value (USD). |

---

## 4. Synthetic Engine: Typologies & Physical Constraints

SynthFin-AML enforces strict financial rules during transaction generation:

1. **Stateful Ledger & Mass Conservation:**
   $$\text{Balance}_u(t) = \text{Balance}_u(0) + \sum_{e \in \text{In}(u, \le t)} \text{Amount}_e - \sum_{e \in \text{Out}(u, \le t)} \text{Amount}_e$$
   Transactions that violate $\text{Balance}_u(t) \ge \text{Amount}_{\text{next}}$ are rejected by default (Non-Sufficient Funds).
2. **Causal Multi-Hop Layering:**
   For any laundering path $(e_1, e_2, \dots, e_k)$, timestamps are strictly monotonically increasing ($t_{i+1} > t_i$), and intermediate amounts account for platform friction fees ($A_{i+1} \le A_i - \epsilon$).
3. **Calibrated Structuring:**
   30% of illicit transaction flows split transfers dynamically just below the \$10,000 regulatory reporting threshold (Currency Transaction Report limits).
4. **Behavioral Agent Roles:**
   * **Retail (80%):** Standard P2P and salary-driven consumer accounts.
   * **Merchants (15%):** High in-degree commercial collection hubs.
   * **P2P Brokers (4%):** High-volume, high-frequency liquidity providers.
   * **Escrow Services (1%):** High-balance transit accounts.
   * **Mules & Laundering Rings:** Injected coordinated structuring subgraphs.

---

## 5. Benchmark Results

Evaluations are conducted across **5 fixed random seeds** (`42, 123, 456, 789, 2024`) on the 3-snapshot temporal split. 

Metrics reported: **PR-AUC** (Area Under the Precision-Recall Curve) and **Precision@Top-100** (reflecting fixed human compliance review capacity).

| Model Architecture | Input Modality | PR-AUC | Precision@Top-100 | Training Time (CPU/GPU) |
| :--- | :--- | :--- | :--- | :--- |
| **Random Baseline** | N/A | 0.0200 | 0.02 | < 1 ms |
| **Logistic Regression** | Tabular (Base 8 features) | 0.3124 ± 0.0112 | 0.28 ± 0.03 | ~3 sec (CPU) |
| **MLP (3-Layer)** | Tabular (All 10 features) | 0.7415 ± 0.0185 | 0.81 ± 0.02 | ~12 sec (GPU) |
| **LightGBM (Tuned)** | Tabular + 1-Hop + PageRank (11 features) | 0.8483 ± 0.0169 | 0.96 ± 0.01 | ~18 sec (CPU) |
| **XGBoost (Tuned)** | Tabular + 1-Hop + PageRank (11 features) | 0.8512 ± 0.0154 | 0.96 ± 0.01 | ~24 sec (CPU) |
| **GCN (2-Layer)** | Graph Topology + Node Features | 0.8640 ± 0.0135 | 0.97 ± 0.01 | ~45 sec (GPU) |
| **PyG GraphSAGE** | Graph Topology + Node Features | **0.8817 ± 0.0147** | **0.98 ± 0.01** | ~60 sec (GPU) |

*GraphSAGE achieves a statistically significant performance gain over the strongest feature-engineered tabular baseline ($p = 4.6 \times 10^{-5}$).*

---

## 6. Quickstart & Usage

### Installation
```bash
pip install synthfin-aml torch-geometric
```

### PyTorch Geometric Integration
```python
from synthfin_aml_pkg import SynthFinDataset
from torch_geometric.loader import NeighborLoader

# Loads 3 temporal snapshot graphs (downloaded automatically if not cached)
dataset = SynthFinDataset(root='./data')

train_graph = dataset[0]  # Days 1-7
val_graph   = dataset[1]  # Days 1-8
test_graph  = dataset[2]  # Days 1-10

print(f"Train edges: {train_graph.edge_index.size(1)}")  # 787,307
print(f"Test edges:  {test_graph.edge_index.size(1)}")   # 1,273,403

# Mini-batch neighbor sampling for inductive evaluation
train_loader = NeighborLoader(
    train_graph,
    num_neighbors=[15, 10],
    batch_size=1024,
    input_nodes=train_graph.train_mask,
    shuffle=True
)
```

### Standalone Pure PyTorch GraphSAGE (Zero External GNN Dependencies)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PureTorchGraphSAGE(nn.Module):
    def __init__(self, in_dim=10, hidden_dim=32, out_dim=2):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.r1 = nn.Linear(in_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        row, col = edge_index
        adj = torch.sparse_coo_tensor(
            torch.stack([col, row]), edge_weight, (x.size(0), x.size(0))
        )
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-5).unsqueeze(-1)
        aggr = torch.sparse.mm(adj, x) / deg
        return self.out(F.relu(self.l1(x) + self.r1(aggr)))
```

### Generate Custom Synthetic Topologies
```python
from synthfin_aml_pkg.generator import FraudGraphGenerator

# Instantiate generator with custom agent counts and time horizons
gen = FraudGraphGenerator(seed=42)
gen.generate_transactions(agents=50000, days=30, fraud_ratio=0.015)

nodes_df, edges_df = gen.to_dataframes()
print(f"Generated {len(nodes_df)} nodes and {len(edges_df)} transactions.")
```

---

## 7. Hardware Requirements & Benchmarking Setup

* **LightGBM / Tabular Baselines:** Minimum 4 GB RAM, 2 CPU cores. Runtime: ~20 seconds.
* **GraphSAGE / PyG Baselines:** Minimum 8 GB RAM, 4 GB GPU VRAM (Tested on NVIDIA T4, RTX 3060, A100). Runtime: ~60–90 seconds for 50 epochs.
* **Storage Footprint:** Raw dataset ~35 MB compressed (`.zip`), ~140 MB uncompressed CSVs, ~85 MB processed PyG `.pt` snapshot tensors.

---

## 8. License & Privacy Compliance

SynthFin-AML is distributed under the **MIT License**.

All entities, balances, accounts, and transactions are **100% synthetically generated** via parametric mathematical models. The dataset contains **zero Personally Identifiable Information (PII)** and zero proprietary banking records, making it fully compliant with GDPR, CCPA, and institutional data governance frameworks. It is approved for both commercial R&D and open academic publication.

---

## 9. Citation

If you use SynthFin-AML in your research or production benchmarking, please cite:

```bibtex
@software{synthfin_aml_2026,
  author = {Oktay Valiyev},
  title = {SynthFin-AML: A Graph-Native Anti-Money Laundering Benchmark with Temporal Causality},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/valiyevoktay-cmd/synthfin-aml-}
}
```
