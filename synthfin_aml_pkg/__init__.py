"""
synthfin-aml: Synthetic Financial Transaction Graph Generator for AML Detection

Generate realistic transaction graphs with injected money laundering patterns
to train Graph Neural Networks (GNNs) for Anti-Money Laundering.

Usage:
    from synthfin_aml import FraudGraphGenerator

    gen = FraudGraphGenerator(seed=42)
    gen.generate_background(agents=10000, days=30)
    gen.inject_aml_ring(size=5, total_amount=50000)
    gen.inject_smurfing(mules=8, total_amount=100000)
    gen.inject_layering(depth=3, branching=2, total_amount=150000)

    nodes_df, txn_df = gen.to_dataframes()
"""

__version__ = "9.2.0"

from .generator import FraudGraphGenerator
from .dataset import SynthFinDataset

__all__ = ["FraudGraphGenerator", "SynthFinDataset"]
