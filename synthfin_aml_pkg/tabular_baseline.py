import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, classification_report
import time
import os
import gc

class TabularBaseline:
    def __init__(self, data_path: str = None, df: pl.DataFrame = None):
        if df is not None:
            self.df = df
        elif data_path:
            self.data_path = data_path
            self.df = pl.read_csv(data_path)
        else:
            raise ValueError("Provide either data_path or df")
            
        # Ensure timestamp is datetime
        if self.df.schema["timestamp"] in [pl.Float64, pl.Int64]:
            self.df = self.df.with_columns(
                pl.from_epoch("timestamp", time_unit="s").alias("timestamp")
            )
            
        self.df = self.df.sort("timestamp")
        
    def _compute_ego_features_sharded(self, K=10) -> pl.DataFrame:
        print(f"Computing Ego-Net Features (Deterministic Sharding K={K})...")
        start_t = time.time()
        
        # Melt to node level (source -> OUT, dest -> IN)
        df_out = self.df.select([
            pl.col("txn_id"),
            pl.col("timestamp"),
            pl.col("source_id").alias("node_id"),
            pl.col("target_id").alias("counterparty"),
            pl.col("amount").alias("amt_out"),
            pl.lit(0.0).alias("amt_in")
        ])
        
        df_in = self.df.select([
            pl.col("txn_id"),
            pl.col("timestamp"),
            pl.col("target_id").alias("node_id"),
            pl.col("source_id").alias("counterparty"),
            pl.lit(0.0).alias("amt_out"),
            pl.col("amount").alias("amt_in")
        ])
        
        df_nodes = pl.concat([df_out, df_in])
        
        # Apply deterministic sharding
        # Polars hash might be non-deterministic across runs, but consistent within a run.
        # For strict deterministic sharding: modulo on node_id
        df_nodes = df_nodes.with_columns(
            (pl.col("node_id") % K).alias("shard_id")
        )
        
        shard_results = []
        
        for k in range(K):
            print(f"  Processing Shard {k+1}/{K}...")
            # Extract shard
            shard_df = df_nodes.filter(pl.col("shard_id") == k).sort(["node_id", "timestamp"])
            
            # Now we can safely do eager rolling on this smaller shard
            rolling = shard_df.rolling(
                index_column="timestamp", 
                period="7d", 
                closed="left",
                by="node_id"
            ).agg([
                pl.col("amt_in").sum().alias("vol_in_7d"),
                pl.col("amt_out").sum().alias("vol_out_7d"),
                pl.col("counterparty").n_unique().alias("unique_counterparties_7d"),
                pl.col("timestamp").max().alias("last_txn_ts")
            ])
            
            shard_results.append(rolling)
            
            del shard_df
            gc.collect()
            
        print("Reassembling shards...")
        all_rolling = pl.concat(shard_results)
        
        # Join back to transactions for source
        df_feat = self.df.join(
            all_rolling, 
            left_on=["source_id", "timestamp"],
            right_on=["node_id", "timestamp"],
            how="left"
        ).rename({
            "vol_in_7d": "src_vol_in_7d",
            "vol_out_7d": "src_vol_out_7d",
            "unique_counterparties_7d": "src_unique_counterparties_7d",
            "last_txn_ts": "src_last_txn_ts"
        })
        
        # Calculate derived features
        epsilon = 1e-5
        df_feat = df_feat.with_columns([
            (pl.col("src_vol_in_7d").fill_null(0) / (pl.col("src_vol_out_7d").fill_null(0) + epsilon)).alias("src_in_out_ratio_7d"),
            (pl.col("timestamp") - pl.col("src_last_txn_ts")).dt.total_seconds().fill_null(86400 * 30).alias("src_time_since_last_txn")
        ])
        
        print(f"Feature engineering completed in {time.time() - start_t:.2f} seconds.")
        return df_feat
        
    def train_and_evaluate(self):
        df_feat = self._compute_ego_features_sharded()
        
        split_idx = int(df_feat.shape[0] * 0.8)
        train_df = df_feat.slice(0, split_idx)
        test_df = df_feat.slice(split_idx, df_feat.shape[0] - split_idx)
        
        feature_cols = [
            "amount", 
            "src_in_out_ratio_7d", 
            "src_unique_counterparties_7d",
            "src_time_since_last_txn",
            "src_vol_in_7d",
            "src_vol_out_7d"
        ]
        
        X_train = train_df.select(feature_cols).to_pandas().fillna(0)
        y_train = train_df.select("is_fraud").to_pandas().values.ravel()
        
        X_test = test_df.select(feature_cols).to_pandas().fillna(0)
        y_test = test_df.select("is_fraud").to_pandas().values.ravel()
        
        print(f"Training LightGBM on {len(X_train)} transactions (Features: {len(feature_cols)})...")
        
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        clf.fit(X_train, y_train)
        
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        print("\n============================================================")
        print("  TABULAR BASELINE (LightGBM) RESULTS")
        print("============================================================")
        print(f"PR-AUC (Test): {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["clean", "fraud"]))
        return pr_auc

    def ablation_no_temporal(self):
        df_feat = self._compute_ego_features_sharded()
        
        split_idx = int(df_feat.shape[0] * 0.8)
        train_df = df_feat.slice(0, split_idx)
        test_df = df_feat.slice(split_idx, df_feat.shape[0] - split_idx)
        
        feature_cols_no_time = [
            "amount",
            "src_in_out_ratio_7d",
            "src_unique_counterparties_7d",
            "src_vol_in_7d",
            "src_vol_out_7d"
        ]
        
        X_train = train_df.select(feature_cols_no_time).to_pandas().fillna(0)
        y_train = train_df.select("is_fraud").to_pandas().values.ravel()
        X_test = test_df.select(feature_cols_no_time).to_pandas().fillna(0)
        y_test = test_df.select("is_fraud").to_pandas().values.ravel()
        
        clf = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        print(f"PR-AUC (no temporal): {pr_auc:.4f}")
        return pr_auc

if __name__ == "__main__":
    from generator import FraudGraphGenerator
    gen = FraudGraphGenerator(seed=42)
    gen.generate_transactions(agents=10000, days=30)
    df = gen.transactions
    baseline = TabularBaseline(df=pl.from_pandas(df))
    baseline.train_and_evaluate()
    baseline.ablation_no_temporal()
