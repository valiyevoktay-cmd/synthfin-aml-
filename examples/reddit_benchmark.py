"""
synthfin-aml Benchmark: LightGBM vs Pure PyTorch GraphSAGE
==========================================================
Methodology (Council-Verified):
- Static 10-day graph snapshot. Zero temporal ambiguity.
- Stratified 80/20 node split. Both models see the same graph.
- LightGBM gets 11 features INCLUDING Weighted PageRank + 1-hop neighbor aggregates.
- GraphSAGE: 40-line pure PyTorch, no PyG/DGL dependency.
- 5 seeds with mean±std for statistical rigor.
- Metrics: PR-AUC + Precision@Top-100 (industry standard).
"""
import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit
import lightgbm as lgb
import networkx as nx

class PureTorchGraphSAGE(nn.Module):
    """40-line GraphSAGE using only torch.sparse. No PyG needed."""
    def __init__(self, in_c, hid, out_c):
        super().__init__()
        self.l1 = nn.Linear(in_c, hid)
        self.r1 = nn.Linear(in_c, hid)
        self.out = nn.Linear(hid, out_c)

    def forward(self, x, ei, ew):
        row, col = ei
        adj = torch.sparse_coo_tensor(
            torch.stack([col, row]), ew, (x.size(0), x.size(0))
        )
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1e-5).unsqueeze(-1)
        aggr = torch.sparse.mm(adj, x) / deg
        return self.out(F.relu(self.l1(x) + self.r1(aggr)))


def build_features(nodes_df, edges_df):
    """Extract 11 features for all nodes: 8 base + 2 neighbor + PageRank."""
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

    y = feat['is_fraud'].values
    X_df = feat.drop(columns=['profile', 'is_fraud'])
    for c in ['initial_balance', 'out_volume', 'in_volume', 'out_max_amt',
              'in_max_amt', 'nbr_in_volume', 'nbr_out_volume', 'pagerank']:
        X_df[c] = np.log1p(X_df[c])
    X = X_df.values

    a2i = {a: i for i, a in enumerate(feat.index)}
    src = edges_df['source_id'].map(a2i).dropna().astype(int).values
    dst = edges_df['target_id'].map(a2i).dropna().astype(int).values
    ml = min(len(src), len(dst))
    ei = torch.tensor(np.vstack([src[:ml], dst[:ml]]), dtype=torch.long)
    ew = torch.tensor(np.log1p(edges_df['amount'].values[:ml]), dtype=torch.float32)

    return X, y, ei, ew


def run_one_seed(seed):
    from synthfin_aml_pkg.generator import FraudGraphGenerator

    gen = FraudGraphGenerator(seed=seed)
    gen.generate_transactions(agents=10000, days=10)
    nodes_df, edges_df = gen.to_dataframes()

    X, y, ei, ew = build_features(nodes_df, edges_df)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(sss.split(X, y))

    mu, sd = X[train_idx].mean(0), X[train_idx].std(0)
    X_s = (X - mu) / (sd + 1e-5)

    # --- LightGBM (tuned) ---
    clf = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=63, learning_rate=0.05,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbose=-1
    )
    clf.fit(X_s[train_idx], y[train_idx])
    p_lgb = clf.predict_proba(X_s[test_idx])[:, 1]
    pr_l, rc_l, _ = precision_recall_curve(y[test_idx], p_lgb)
    auc_lgb = auc(rc_l, pr_l)
    top100_lgb = np.argsort(p_lgb)[-100:]
    p100_lgb = y[test_idx][top100_lgb].sum() / 100

    # --- GraphSAGE ---
    torch.manual_seed(seed)
    model = PureTorchGraphSAGE(X_s.shape[1], 32, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    xt = torch.tensor(X_s, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    mask_train = torch.zeros(len(y), dtype=torch.bool)
    mask_train[train_idx] = True

    for _ in range(200):
        model.train(); opt.zero_grad()
        o = model(xt, ei, ew)
        F.cross_entropy(o[mask_train], yt[mask_train]).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        p_all = F.softmax(model(xt, ei, ew), dim=1)[:, 1].numpy()
    p_gnn = p_all[test_idx]
    pr_g, rc_g, _ = precision_recall_curve(y[test_idx], p_gnn)
    auc_gnn = auc(rc_g, pr_g)
    top100_gnn = np.argsort(p_gnn)[-100:]
    p100_gnn = y[test_idx][top100_gnn].sum() / 100

    return auc_lgb, auc_gnn, p100_lgb, p100_gnn


if __name__ == '__main__':
    results = []
    for s in [42, 123, 456, 789, 2024]:
        print(f'--- Seed {s} ---')
        a_l, a_g, p_l, p_g = run_one_seed(s)
        print(f'  LightGBM  PR-AUC={a_l:.4f}  P@100={p_l:.2f}')
        print(f'  GraphSAGE PR-AUC={a_g:.4f}  P@100={p_g:.2f}')
        results.append((a_l, a_g, p_l, p_g))

    r = np.array(results)
    print(f'\n{"="*50}')
    print(f'LightGBM  PR-AUC: {r[:,0].mean():.4f} ± {r[:,0].std():.4f}')
    print(f'GraphSAGE PR-AUC: {r[:,1].mean():.4f} ± {r[:,1].std():.4f}')
    print(f'LightGBM  P@100:  {r[:,2].mean():.2f} ± {r[:,2].std():.2f}')
    print(f'GraphSAGE P@100:  {r[:,3].mean():.2f} ± {r[:,3].std():.2f}')
    print(f'Δ PR-AUC: {(r[:,1]-r[:,0]).mean():+.4f}')
    print(f'Δ P@100:  {(r[:,3]-r[:,2]).mean():+.2f}')
