import polars as pl
import lightgbm as lgb
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, classification_report
import time

class TabularBaseline:
    def __init__(self, data_path: str = None, df: pl.DataFrame = None):
        if df is not None:
            self.df = df
        elif data_path:
            self.df = pl.read_csv(data_path)
        else:
            raise ValueError("Provide either data_path or df")
            
        # Ensure timestamp is datetime
        if self.df.schema["timestamp"] == pl.Float64:
            self.df = self.df.with_columns(
                pl.from_epoch("timestamp", time_unit="s").alias("timestamp")
            )
            
        self.df = self.df.sort("timestamp")
        
    def _compute_ego_features(self) -> pl.DataFrame:
        print("Computing Ego-Net Features (Strict Prior Windows)...")
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
        
        df_nodes = pl.concat([df_out, df_in]).sort(["node_id", "timestamp"])
        
        # Calculate Rolling 7d volumes using rolling to avoid memory explosion
        # We use closed="left" to strictly exclude the current transaction from its own prior window
        rolling = df_nodes.rolling(
            index_column="timestamp", 
            period="7d", 
            closed="left",
            by="node_id"
        ).agg([
            pl.col("amt_in").sum().alias("vol_in_7d"),
            pl.col("amt_out").sum().alias("vol_out_7d"),
            pl.col("counterparty").n_unique().alias("unique_counterparties_7d"),
            # Delta between last IN and current time (approximated by last timestamp in window)
            pl.col("timestamp").max().alias("last_txn_ts")
        ])
        
        # Join back to transactions for source and dest
        df_feat = self.df.join(
            rolling, 
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
        df_feat = self._compute_ego_features()
        
        # Strict OOT Split (last 20% by time is test)
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
        
        # Feature Importance
        importance = clf.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        print("\nTop Features:")
        for i in sorted_idx[:5]:
            print(f"{feature_cols[i]}: {importance[i]}")
            
        return pr_auc

    def ablation_no_temporal(self):
        """Ablation: train WITHOUT temporal features to check for temporal leakage."""
        df_feat = self._compute_ego_features()
        
        split_idx = int(df_feat.shape[0] * 0.8)
        train_df = df_feat.slice(0, split_idx)
        test_df = df_feat.slice(split_idx, df_feat.shape[0] - split_idx)
        
        # Only non-temporal features
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
        
        print(f"\n{'='*60}")
        print("  ABLATION: LightGBM WITHOUT temporal features")
        print(f"{'='*60}")
        print(f"Training on {len(X_train)} tx (Features: {len(feature_cols_no_time)})...")
        
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
        
        print(f"PR-AUC (no temporal): {pr_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["clean", "fraud"]))
        
        importance = clf.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        print("Top Features (no temporal):")
        for i in sorted_idx:
            print(f"  {feature_cols_no_time[i]}: {importance[i]}")
        
        return pr_auc

if __name__ == "__main__":
    from generator import FraudGraphGenerator
    import time
    
    gen = FraudGraphGenerator(seed=42)
    gen.generate_transactions(agents=10000, days=30)
    df = gen.transactions
    
    baseline = TabularBaseline(df=pl.from_pandas(df))
    pr_auc_full = baseline.train_and_evaluate()
    pr_auc_no_time = baseline.ablation_no_temporal()
    
    print(f"\n{'='*60}")
    print("  LEAKAGE DIAGNOSIS")
    print(f"{'='*60}")
    print(f"Full features PR-AUC:       {pr_auc_full:.4f}")
    print(f"No temporal PR-AUC:         {pr_auc_no_time:.4f}")
    delta = pr_auc_full - pr_auc_no_time
    print(f"Delta (temporal leak test):  {delta:+.4f}")
    if delta > 0.15:
        print("[!!] HIGH temporal leak detected -- temporal features contribute >15pp")
    elif delta > 0.05:
        print("[!]  MODERATE temporal signal -- may be legitimate AML pattern")
    else:
        print("[OK] LOW temporal dependency -- distributions are well-calibrated")
