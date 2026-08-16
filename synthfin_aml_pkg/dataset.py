import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
from typing import Optional, Callable
from torch_geometric.data import InMemoryDataset, Data

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError("Please install datasets library: pip install datasets")


class SynthFinDataset(InMemoryDataset):
    """
    synthfin-aml V9.1 Dataset for Anti-Money Laundering (AML).
    Downloads pre-generated nodes and edges from Hugging Face: ovvaliyev/synthfin-aml
    and constructs a transductive PyG graph for node classification.
    """
    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # We don't download raw files directly; we use HF datasets API in process()
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        print("Downloading dataset from Hugging Face (ovvaliyev/synthfin-aml)...")
        # Load from Hugging Face (separate configs)
        nodes_dataset = load_dataset("ovvaliyev/synthfin-aml", "nodes")
        edges_dataset = load_dataset("ovvaliyev/synthfin-aml", "edges")
        
        nodes_df = nodes_dataset['train'].to_pandas()
        edges_df = edges_dataset['train'].to_pandas()
        
        print("Computing topology features (Weighted PageRank & Neighbor Aggregates)...")
        # Compute the explicit topological features required for the benchmark
        od = edges_df.groupby('source_id').size().rename('out_degree')
        id_ = edges_df.groupby('target_id').size().rename('in_degree')
        ov = edges_df.groupby('source_id')['amount'].sum().rename('out_volume')
        iv = edges_df.groupby('target_id')['amount'].sum().rename('in_volume')
        om = edges_df.groupby('source_id')['amount'].max().rename('out_max_amt')
        im = edges_df.groupby('target_id')['amount'].max().rename('in_max_amt')

        ev = edges_df.merge(iv, left_on='target_id', right_index=True, how='left')
        ev = ev.merge(ov, left_on='source_id', right_index=True, how='left')
        ni = ev.groupby('source_id')['in_volume'].mean().rename('nbr_in_volume')
        no_ = ev.groupby('target_id')['out_volume'].mean().rename('nbr_out_volume')

        G = nx.from_pandas_edgelist(
            edges_df, 'source_id', 'target_id',
            edge_attr='amount', create_using=nx.DiGraph()
        )
        pr = pd.Series(nx.pagerank(G, weight='amount'), name='pagerank')

        feat = nodes_df.set_index('agent_id').copy()
        feat = feat.join([od, id_, ov, iv, om, im, ni, no_, pr]).fillna(0)

        # Log transform large monetary values for stability
        for c in ['initial_balance', 'out_volume', 'in_volume', 'out_max_amt',
                  'in_max_amt', 'nbr_in_volume', 'nbr_out_volume', 'pagerank']:
            feat[c] = np.log1p(feat[c])
            
        y_val = feat['is_fraud'].values
        X_df = feat.drop(columns=['profile', 'is_fraud'])
        X_val = X_df.values
        
        # Standardize features
        mu, sd = X_val.mean(0), X_val.std(0)
        X_s = (X_val - mu) / (sd + 1e-5)

        # Mapping node IDs to continuous range 0..N-1
        a2i = {a: i for i, a in enumerate(feat.index)}
        src = edges_df['source_id'].map(a2i).dropna().astype(int).values
        dst = edges_df['target_id'].map(a2i).dropna().astype(int).values
        ml = min(len(src), len(dst))
        
        edge_index = torch.tensor(np.vstack([src[:ml], dst[:ml]]), dtype=torch.long)
        edge_attr = torch.tensor(np.log1p(edges_df['amount'].values[:ml]), dtype=torch.float32)

        x = torch.tensor(X_s, dtype=torch.float32)
        y = torch.tensor(y_val, dtype=torch.long)
        
        # Create standard transductive masks (80% train / 20% test, stratified)
        # Using a fixed seed for reproducible benchmark splits
        np.random.seed(42)
        fraud_idx = np.where(y_val == 1)[0]
        clean_idx = np.where(y_val == 0)[0]
        
        np.random.shuffle(fraud_idx)
        np.random.shuffle(clean_idx)
        
        f_split = int(0.8 * len(fraud_idx))
        c_split = int(0.8 * len(clean_idx))
        
        train_idx = np.concatenate([fraud_idx[:f_split], clean_idx[:c_split]])
        test_idx = np.concatenate([fraud_idx[f_split:], clean_idx[c_split:]])
        
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    train_mask=train_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])
        print(f"Dataset successfully processed and saved to {self.processed_paths[0]}")
