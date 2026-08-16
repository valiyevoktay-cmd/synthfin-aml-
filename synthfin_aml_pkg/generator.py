"""
synthfin-aml: Synthetic Financial Transaction Graph Generator for AML/Fraud Detection

V9.1: Calibrated Distributions & Symmetric Friction (LLM Council V9.1 Architecture)
Changes from V9:
- Amount Distribution Alignment: Fraud, P2P_Broker, and Escrow all use lognormal(mu=8.517, sigma=0.8).
- Volume Preponderance: Broker/Escrow generate >= fraud tx count in the high-budget segment.
- Temporal Burstiness: Broker/Escrow transactions are clustered into intraday activity bursts.
- Causal Layering: Fraud hop amounts are strictly derived from incoming amount (mass conservation).
- Symmetric Friction: Uniform 5% system failure for ALL tx classes + strict Balance < Amount.
- Structuring Signal: 30% of fraud chains intentionally structure amounts just below $10k to mimic real-world typologies.
- No asymmetric coin-flip NSF rates between classes.
"""

import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime, timedelta
import time
from typing import Optional, Tuple

class FraudGraphGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.agents: Optional[pd.DataFrame] = None
        self.transactions: Optional[pd.DataFrame] = None

    def generate_transactions(self, agents: int = 100000, days: int = 30,
                              start_date: str = "2024-01-01", fraud_ratio: float = 0.02) -> "FraudGraphGenerator":
        t0 = time.time()
        start = datetime.strptime(start_date, "%Y-%m-%d")
        start_ts = start.timestamp()
        end_ts = start_ts + days * 24 * 3600
        
        # Calculate approximate transaction volumes
        num_retail_tx = agents * 8  # Retail P2P (low-volume)
        num_fraud_chains = int(agents * 10 * fraud_ratio) // 5
        
        # 1. Generate Agents
        print("Generating Node Profiles...")
        profiles = self.rng.choice(
            ["retail", "merchant", "p2p_broker", "escrow"],
            size=agents,
            p=[0.80, 0.15, 0.04, 0.01]
        )
        is_fraud = np.zeros(agents, dtype=bool)
        
        retail_idx = np.where(profiles == "retail")[0]
        merchant_idx = np.where(profiles == "merchant")[0]
        broker_idx = np.where(profiles == "p2p_broker")[0]
        escrow_idx = np.where(profiles == "escrow")[0]
        broker_escrow_idx = np.concatenate([broker_idx, escrow_idx])
        
        # Root nodes (Salary sources)
        num_salary_roots = max(1, agents // 1000)
        salary_roots = retail_idx[:num_salary_roots]
        
        # Fraud roles
        num_mules = min(agents // 10, num_fraud_chains * 4)
        mules = self.rng.choice(retail_idx[num_salary_roots:], size=num_mules, replace=False)
        is_fraud[mules] = True
        
        # Initialize Seed Balances
        # Retail: moderate balances
        seed_balances = self.rng.lognormal(mean=7.0, sigma=1.0, size=agents)
        # Salary roots: infinite liquidity
        seed_balances[salary_roots] = 1e9
        # Brokers and Escrow: high liquidity (they handle large volumes legitimately)
        seed_balances[broker_escrow_idx] = self.rng.lognormal(mean=10.0, sigma=0.8, size=len(broker_escrow_idx))
        # Merchants: moderate-high
        seed_balances[merchant_idx] = self.rng.lognormal(mean=9.0, sigma=0.8, size=len(merchant_idx))
        
        # ══════════════════════════════════════════════════════════════
        # 2. Clean Intent Generation
        # ══════════════════════════════════════════════════════════════
        print("Generating Clean Intents (Calibrated V9.1)...")
        
        all_cl_sources = []
        all_cl_targets = []
        all_cl_amts = []
        all_cl_timestamps = []
        
        # 2a. Salary injections (10% of retail tx)
        salary_tx = int(num_retail_tx * 0.1)
        cl_sources_sal = self.rng.choice(salary_roots, size=salary_tx)
        cl_targets_sal = self.rng.choice(retail_idx, size=salary_tx)
        cl_amts_sal = self.rng.lognormal(mean=7.0, sigma=0.5, size=salary_tx)
        cl_ts_sal = self.rng.uniform(start_ts, end_ts, size=salary_tx)
        
        all_cl_sources.append(cl_sources_sal)
        all_cl_targets.append(cl_targets_sal)
        all_cl_amts.append(cl_amts_sal)
        all_cl_timestamps.append(cl_ts_sal)
        
        # 2b. Retail P2P (low-volume, everyday transfers)
        clean_retail = np.setdiff1d(retail_idx, mules)
        p2p_tx = num_retail_tx - salary_tx
        cl_sources_p2p = self.rng.choice(clean_retail, size=p2p_tx)
        cl_targets_p2p = self.rng.choice(retail_idx, size=p2p_tx)
        cl_amts_p2p = self.rng.lognormal(mean=4.0, sigma=1.0, size=p2p_tx)
        cl_ts_p2p = self.rng.uniform(start_ts, end_ts, size=p2p_tx)
        
        all_cl_sources.append(cl_sources_p2p)
        all_cl_targets.append(cl_targets_p2p)
        all_cl_amts.append(cl_amts_p2p)
        all_cl_timestamps.append(cl_ts_p2p)
        
        # 2c. Broker/Escrow high-volume legitimate traffic (CRITICAL: Volume Preponderance)
        # We need OVERWHELMING clean high-volume tx to blind amount-based classifiers.
        # Target: N(clean high-volume) >= 10 * N(fraud edges)
        num_fraud_edges_estimate = num_fraud_chains * 5  # ~5 hops per chain
        num_broker_outgoing = max(num_fraud_edges_estimate * 5, len(broker_escrow_idx) * 30)
        
        # Broker -> Retail/Merchant (outgoing high-volume)
        cl_sources_bk1 = self.rng.choice(broker_escrow_idx, size=num_broker_outgoing)
        target_pool = np.concatenate([retail_idx, merchant_idx, broker_escrow_idx])
        cl_targets_bk1 = self.rng.choice(target_pool, size=num_broker_outgoing)
        cl_amts_bk1 = self.rng.lognormal(mean=8.517, sigma=0.8, size=num_broker_outgoing)
        cl_ts_bk1 = self._generate_bursty_timestamps(num_broker_outgoing, start_ts, end_ts, days)
        
        all_cl_sources.append(cl_sources_bk1)
        all_cl_targets.append(cl_targets_bk1)
        all_cl_amts.append(cl_amts_bk1)
        all_cl_timestamps.append(cl_ts_bk1)
        
        # Retail -> Broker/Escrow (incoming high-volume: customers depositing/paying)
        num_broker_incoming = num_broker_outgoing
        cl_sources_bk2 = self.rng.choice(retail_idx, size=num_broker_incoming)
        cl_targets_bk2 = self.rng.choice(broker_escrow_idx, size=num_broker_incoming)
        cl_amts_bk2 = self.rng.lognormal(mean=8.517, sigma=0.8, size=num_broker_incoming)
        cl_ts_bk2 = self._generate_bursty_timestamps(num_broker_incoming, start_ts, end_ts, days)
        
        all_cl_sources.append(cl_sources_bk2)
        all_cl_targets.append(cl_targets_bk2)
        all_cl_amts.append(cl_amts_bk2)
        all_cl_timestamps.append(cl_ts_bk2)
        
        # Merchant high-volume (wholesale/B2B — same distribution)
        num_merchant_hv = num_fraud_edges_estimate * 3
        cl_sources_mhv = self.rng.choice(merchant_idx, size=num_merchant_hv)
        cl_targets_mhv = self.rng.choice(np.concatenate([merchant_idx, broker_escrow_idx]), size=num_merchant_hv)
        cl_amts_mhv = self.rng.lognormal(mean=8.517, sigma=0.8, size=num_merchant_hv)
        cl_ts_mhv = self.rng.uniform(start_ts, end_ts, size=num_merchant_hv)
        
        all_cl_sources.append(cl_sources_mhv)
        all_cl_targets.append(cl_targets_mhv)
        all_cl_amts.append(cl_amts_mhv)
        all_cl_timestamps.append(cl_ts_mhv)
        
        total_clean_hv = num_broker_outgoing + num_broker_incoming + num_merchant_hv
        print(f"  High-volume clean intents: {total_clean_hv:,} (vs ~{num_fraud_edges_estimate:,} fraud edges, ratio {total_clean_hv/max(1,num_fraud_edges_estimate):.1f}x)")
        
        # 2d. Merchant receiving (retail -> merchant purchases)
        merchant_tx = agents * 2
        cl_sources_merch = self.rng.choice(retail_idx, size=merchant_tx)
        cl_targets_merch = self.rng.choice(merchant_idx, size=merchant_tx)
        cl_amts_merch = self.rng.lognormal(mean=5.0, sigma=1.2, size=merchant_tx)
        cl_ts_merch = self.rng.uniform(start_ts, end_ts, size=merchant_tx)
        
        all_cl_sources.append(cl_sources_merch)
        all_cl_targets.append(cl_targets_merch)
        all_cl_amts.append(cl_amts_merch)
        all_cl_timestamps.append(cl_ts_merch)
        
        # 2e. Mule background traffic (CRITICAL: temporal camouflage)
        # Mules MUST have DENSE normal P2P activity so their inter-arrival times
        # match clean retail. We give them 3x the average retail rate to ensure
        # fraud bursts are drowned in background noise.
        avg_tx_per_node = num_retail_tx / max(1, len(clean_retail))
        mule_multiplier = 1.0  # Exactly same rate as clean retail (no volume leakage)
        num_mule_bg_tx = int(len(mules) * avg_tx_per_node * mule_multiplier)
        
        # Outgoing from mules (P2P)
        cl_sources_mule = self.rng.choice(mules, size=num_mule_bg_tx)
        cl_targets_mule = self.rng.choice(retail_idx, size=num_mule_bg_tx)
        cl_amts_mule = self.rng.lognormal(mean=4.0, sigma=1.0, size=num_mule_bg_tx)
        cl_ts_mule = self.rng.uniform(start_ts, end_ts, size=num_mule_bg_tx)
        
        all_cl_sources.append(cl_sources_mule)
        all_cl_targets.append(cl_targets_mule)
        all_cl_amts.append(cl_amts_mule)
        all_cl_timestamps.append(cl_ts_mule)
        
        # Incoming to mules (salary/P2P receipts — so their balance and in-flow look normal)
        num_mule_in_tx = int(num_mule_bg_tx * 0.8)
        cl_sources_mule_in = self.rng.choice(np.concatenate([salary_roots, clean_retail]), size=num_mule_in_tx)
        cl_targets_mule_in = self.rng.choice(mules, size=num_mule_in_tx)
        cl_amts_mule_in = self.rng.lognormal(mean=4.0, sigma=1.0, size=num_mule_in_tx)
        cl_ts_mule_in = self.rng.uniform(start_ts, end_ts, size=num_mule_in_tx)
        
        all_cl_sources.append(cl_sources_mule_in)
        all_cl_targets.append(cl_targets_mule_in)
        all_cl_amts.append(cl_amts_mule_in)
        all_cl_timestamps.append(cl_ts_mule_in)
        
        total_mule_bg = num_mule_bg_tx + num_mule_in_tx
        print(f"  Mule background cover: {total_mule_bg:,} clean tx ({avg_tx_per_node * mule_multiplier:.1f} tx/mule out + {avg_tx_per_node * mule_multiplier * 0.8:.1f} in)")
        
        # Concatenate all clean intents
        cl_sources = np.concatenate(all_cl_sources).astype(np.int32)
        cl_targets = np.concatenate(all_cl_targets).astype(np.int32)
        cl_amts = np.concatenate(all_cl_amts).astype(np.float64)
        cl_timestamps = np.concatenate(all_cl_timestamps).astype(np.float64)
        
        # Fix self-loops
        mask = cl_sources == cl_targets
        cl_targets[mask] = (cl_targets[mask] + 1) % agents
        
        num_clean_tx = len(cl_sources)
        cl_is_fraud = np.zeros(num_clean_tx, dtype=np.int32)
        
        # ══════════════════════════════════════════════════════════════
        # 3. Fraud Intent Generation (Causal Layering)
        # ══════════════════════════════════════════════════════════════
        print("Generating Fraud Intents (Causal Layering V9.1)...")
        fr_sources = []
        fr_targets = []
        fr_amts = []
        fr_timestamps = []
        
        fraud_source_nodes = self.rng.choice(retail_idx[num_salary_roots:], size=num_fraud_chains)
        
        # Fund fraud sources with the SAME distribution as broker/escrow seed balances
        # This prevents balance-based leakage
        for idx in range(num_fraud_chains):
            source_node = fraud_source_nodes[idx]
            # Use same lognormal(10.0, 0.8) as brokers — no special treatment
            seed_balances[source_node] = max(seed_balances[source_node],
                                             self.rng.lognormal(mean=10.0, sigma=0.8))
        
        for idx in range(num_fraud_chains):
            source_node = fraud_source_nodes[idx]
            chain_nodes = [source_node] + list(self.rng.choice(mules, size=4, replace=False)) + [self.rng.choice(broker_escrow_idx)]
            
            # IDENTICAL distribution to broker/escrow amounts mostly,
            # but inject 30% "structuring" behavior just below the $10,000 reporting threshold (real-world AML typology).
            if self.rng.random() < 0.3:
                base_amount = self.rng.uniform(9000.0, 9999.0)
            else:
                base_amount = self.rng.lognormal(mean=8.517, sigma=0.8)
            
            # Bursty base timestamp (same intraday clustering as brokers)
            base_t = self.rng.uniform(start_ts, start_ts + (days - 2) * 24 * 3600)
            
            # Truncation logic (20% off-graph leakage)
            steps = 3 if self.rng.random() < 0.2 else 5
            
            # CAUSAL LAYERING: each hop amount derives strictly from previous
            current_amt = base_amount
            current_t = base_t
            
            for s in range(steps):
                fr_sources.append(chain_nodes[s])
                fr_targets.append(chain_nodes[s+1])
                # Mass conservation: hop_amt = current_amt * U(0.95, 1.0)
                hop_amt = current_amt * self.rng.uniform(0.95, 1.0)
                fr_amts.append(hop_amt)
                
                # Delay variance: T_out > T_in (10 min to 1 hour)
                current_t += self.rng.uniform(600, 3600)
                fr_timestamps.append(current_t)
                current_amt = hop_amt
                
        num_fraud_edges = len(fr_sources)
        fr_sources = np.array(fr_sources, dtype=np.int32)
        fr_targets = np.array(fr_targets, dtype=np.int32)
        fr_amts = np.array(fr_amts, dtype=np.float64)
        fr_timestamps = np.array(fr_timestamps, dtype=np.float64)
        fr_is_fraud = np.ones(num_fraud_edges, dtype=np.int32)
        
        print(f"  Fraud edges generated: {num_fraud_edges}")
        
        # ══════════════════════════════════════════════════════════════
        # 4. Chronological Resolution (Symmetric Friction)
        # ══════════════════════════════════════════════════════════════
        print("Chronological Resolution Pass (Symmetric 5% Friction)...")
        all_sources = np.concatenate([cl_sources, fr_sources]).astype(np.int32)
        all_targets = np.concatenate([cl_targets, fr_targets]).astype(np.int32)
        all_amts = np.concatenate([cl_amts, fr_amts]).astype(np.float64)
        all_timestamps = np.concatenate([cl_timestamps, fr_timestamps]).astype(np.float64)
        all_is_fraud = np.concatenate([cl_is_fraud, fr_is_fraud]).astype(np.int32)
        
        # Sort by timestamp
        sort_idx = np.argsort(all_timestamps)
        all_sources = all_sources[sort_idx]
        all_targets = all_targets[sort_idx]
        all_amts = all_amts[sort_idx]
        all_timestamps = all_timestamps[sort_idx]
        all_is_fraud = all_is_fraud[sort_idx]
        
        # Generate uniform friction random numbers (one per transaction)
        friction_rng = np.random.default_rng(self.seed + 1)
        friction_rolls = friction_rng.random(len(all_sources)).astype(np.float64)
        
        # Execute fast Numba loop with symmetric friction
        approved_mask = self._resolve_balances_numba_v91(
            all_sources, all_targets, all_amts, seed_balances, friction_rolls, 0.05
        )
        
        # 5. Build Final DataFrames
        print("Building Final Outputs...")
        self.agents = pd.DataFrame({
            "agent_id": np.arange(agents),
            "profile": profiles,
            "is_fraud": is_fraud,
            "initial_balance": seed_balances
        })
        
        self.transactions = pd.DataFrame({
            "txn_id": np.arange(np.sum(approved_mask)),
            "source_id": all_sources[approved_mask],
            "target_id": all_targets[approved_mask],
            "timestamp": all_timestamps[approved_mask],
            "amount": all_amts[approved_mask],
            "is_fraud": all_is_fraud[approved_mask]
        })
        self.transactions["timestamp"] = pd.to_datetime(self.transactions["timestamp"], unit="s")
        
        # Diagnostics
        n_total = len(self.transactions)
        n_fraud = self.transactions.is_fraud.sum()
        n_clean = n_total - n_fraud
        elapsed = time.time() - t0
        
        print(f"[synthfin-aml V9.1] Generated {n_total:,} transactions ({n_fraud:,} fraud, {n_clean:,} clean). Elapsed: {elapsed:.1f}s")
        
        # Distribution diagnostics
        fraud_amts = self.transactions[self.transactions.is_fraud == 1].amount
        clean_high = self.transactions[(self.transactions.is_fraud == 0) & (self.transactions.amount > 1000)].amount
        if len(fraud_amts) > 0 and len(clean_high) > 0:
            print(f"  Amount diagnostics:")
            print(f"    Fraud mean: ${fraud_amts.mean():,.0f}, median: ${fraud_amts.median():,.0f}")
            print(f"    Clean (>$1k) mean: ${clean_high.mean():,.0f}, median: ${clean_high.median():,.0f}")
            print(f"    Clean (>$1k) count: {len(clean_high):,} vs Fraud count: {len(fraud_amts):,}")
        
        return self

    def _generate_bursty_timestamps(self, n: int, start_ts: float, end_ts: float, days: int) -> np.ndarray:
        """Generate bursty timestamps mimicking real broker activity patterns.
        
        Brokers don't transact uniformly. They have:
        - Business hour concentration (9am-6pm local)
        - Session clustering (bursts of 5-20 tx within 5-30 minutes)
        - Weekend drop-off
        """
        timestamps = np.empty(n, dtype=np.float64)
        idx = 0
        day_seconds = 24 * 3600
        
        while idx < n:
            # Pick a random day
            day_offset = self.rng.integers(0, days)
            day_start = start_ts + day_offset * day_seconds
            
            # Weekend probability reduction (Sat/Sun)
            weekday = (datetime.fromtimestamp(day_start).weekday())
            if weekday >= 5 and self.rng.random() < 0.7:  # 70% skip weekends
                continue
            
            # Business hours: 9am-6pm (32400 to 64800 seconds from midnight)
            # With some after-hours activity (10% chance)
            if self.rng.random() < 0.9:
                session_start = day_start + self.rng.uniform(32400, 61200)  # 9am-5pm
            else:
                session_start = day_start + self.rng.uniform(0, day_seconds)  # any time
            
            # Session burst: 3-15 transactions within 5-30 minutes
            burst_size = min(self.rng.integers(3, 16), n - idx)
            burst_duration = self.rng.uniform(300, 1800)  # 5-30 min
            
            burst_times = session_start + np.sort(self.rng.uniform(0, burst_duration, size=burst_size))
            
            # Clip to valid range
            burst_times = np.clip(burst_times, start_ts, end_ts)
            
            timestamps[idx:idx + burst_size] = burst_times
            idx += burst_size
        
        return timestamps[:n]

    @staticmethod
    @njit(cache=True)
    def _resolve_balances_numba_v91(sources, targets, amounts, balances, friction_rolls, friction_rate):
        """V9.1 Resolver: Strict Balance < Amount + Symmetric 5% system friction.
        
        Two rejection rules (identical for ALL transaction classes):
        1. Deterministic NSF: Balance_sender < Amount (conservation law)
        2. Stochastic friction: Uniform(0,1) < friction_rate (system failures, applies equally)
        """
        n = len(sources)
        approved = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            # Rule 2: Symmetric system friction (5% random failure for everyone)
            if friction_rolls[i] < friction_rate:
                continue
            
            s = sources[i]
            t = targets[i]
            amt = amounts[i]
            
            # Rule 1: Strict conservation — sender must have sufficient funds
            if balances[s] >= amt:
                balances[s] -= amt
                balances[t] += amt
                approved[i] = True
        return approved

    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.agents is None or self.transactions is None:
            raise RuntimeError("No data generated.")
        return self.agents.copy(), self.transactions.copy()

    def summary(self) -> dict:
        nodes_df, txn_df = self.to_dataframes()
        fraud_txns = txn_df[txn_df.is_fraud == 1]
        clean_txns = txn_df[txn_df.is_fraud == 0]
        return {
            "total_agents": len(nodes_df),
            "total_transactions": len(txn_df),
            "fraud_transactions": len(fraud_txns),
            "clean_transactions": len(clean_txns)
        }

    def to_csv(self, nodes_path: str = "nodes.csv", transactions_path: str = "transactions.csv") -> None:
        nodes_df, txn_df = self.to_dataframes()
        nodes_df.to_csv(nodes_path, index=False)
        txn_df.to_csv(transactions_path, index=False)
