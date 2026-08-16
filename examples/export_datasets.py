"""
export_datasets.py — Script to generate static reference datasets for HuggingFace.

This script runs the V9.1 generator to produce three benchmark datasets:
- synthfin-small (10k nodes)
- synthfin-medium (100k nodes)
- synthfin-large (1M nodes)

Outputs are saved to the 'datasets' directory as compressed CSVs.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "synthfin_aml_pkg")))
from generator import FraudGraphGenerator

def export_dataset(name: str, agents: int, days: int, seed: int = 42):
    print("=" * 60)
    print(f"Generating {name} ({agents:,} nodes, {days} days)")
    print("=" * 60)
    
    t_start = time.time()
    
    gen = FraudGraphGenerator(seed=seed)
    gen.generate_transactions(agents=agents, days=days)
    
    df_agents = gen.agents
    df_txns = gen.transactions
    
    # Save to CSV
    os.makedirs("datasets", exist_ok=True)
    
    nodes_path = f"datasets/{name}_nodes.csv"
    edges_path = f"datasets/{name}_edges.csv"
    
    print(f"Saving nodes to {nodes_path}...")
    df_agents.to_csv(nodes_path, index=False)
    
    print(f"Saving edges to {edges_path}...")
    df_txns.to_csv(edges_path, index=False)
    
    t_total = time.time() - t_start
    print(f"[{name}] Done in {t_total:.1f}s.")
    print(f"Nodes: {len(df_agents):,} | Edges: {len(df_txns):,}")
    print(f"Fraud ratio: {df_txns['is_fraud'].mean():.2%}\n")

if __name__ == "__main__":
    export_dataset("synthfin-small", agents=10_000, days=30, seed=42)
    export_dataset("synthfin-medium", agents=100_000, days=30, seed=42)
    # The 1M node graph might take a few minutes to generate and save, so do it last
    export_dataset("synthfin-large", agents=1_000_000, days=30, seed=42)
