"""
Example: Generate a fraud graph and visualize basic statistics.

Usage:
    python examples/quickstart.py
"""

import sys
sys.path.insert(0, "..")

from synthfin_aml_pkg import FraudGraphGenerator


def main():
    # 1. Create generator
    gen = FraudGraphGenerator(seed=42)

    # 2. Generate realistic background (10k agents, 30 days)
    gen.generate_background(agents=10_000, days=30)

    # 3. Inject fraud patterns
    gen.inject_aml_ring(size=5, total_amount=50_000)
    gen.inject_aml_ring(size=7, total_amount=120_000)
    gen.inject_smurfing(mules=8, total_amount=100_000)
    gen.inject_smurfing(mules=12, total_amount=200_000)
    gen.inject_layering(depth=3, branching=2, total_amount=150_000)

    # 4. Get data
    nodes_df, txn_df = gen.to_dataframes()

    # 5. Print summary
    summary = gen.summary()
    print("\n--- Dataset Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # 6. Show fraud vs clean distribution
    print("\n--- Transaction Type Breakdown ---")
    print(txn_df.groupby(["txn_type", "is_fraud"]).size().to_string())

    # 7. Export
    gen.to_csv("output_nodes.csv", "output_transactions.csv")
    print("\nDone! Files saved.")


if __name__ == "__main__":
    main()
