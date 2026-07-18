"""
Coreset selection for the expensive SubgraphX stability eval.

For each dataset except mutag (small enough to run in full), every graph is
bucketed by (predicted class, confidence decile, size decile) -- ~10 buckets
per dimension. Each bucket gets a budget proportional to its share of the
dataset, and within a bucket we pick a diverse subset via greedy k-center
(farthest-point) sampling over the GNTK graph kernel, so the coreset covers
both "what the model predicts and how confidently" and "structural
diversity" within each stratum.
"""
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from gntk_combined import GNTK, convert_pyg_to_s2v
from model import GIN
from training_proteins import GIN as GIN_proteins
from utils import ba2motif_dataset, bamultishapes_dataset, proteins_dataset

NUM_BUCKETS = 10
TOTAL_BUDGET = 200

DATASETS = {
    'ba2motif': ba2motif_dataset,
    'bamultishapes': bamultishapes_dataset,
    'proteins': proteins_dataset,
}


def load_model(dataset_str, dataset):
    if dataset_str == 'ba2motif':
        model = GIN(input_dim=dataset[0].x.shape[1], output_dim=1, multi=False)
        model.load_state_dict(torch.load('models/GIN_model_BA.pth', map_location='cpu', weights_only=True))
    elif dataset_str == 'bamultishapes':
        model = GIN(input_dim=dataset.num_node_features, hidden_dim=64, output_dim=2, multi=True)
        model.load_state_dict(torch.load('models/GIN_model_BA_SHAPES.pt', map_location='cpu', weights_only=True))
    elif dataset_str == 'proteins':
        model = GIN_proteins(input_dim=dataset.num_node_features, hidden_dim=128, output_dim=2, multi=True)
        model.load_state_dict(torch.load('models/GIN_model_PROTEINS.pt', map_location='cpu', weights_only=True))
    else:
        raise ValueError(f'Unknown dataset {dataset_str}')
    model.eval()
    return model


@torch.no_grad()
def get_probs(dataset_str, model, x, edge_index):
    out = model(x, edge_index)
    # GIN_proteins returns raw logits when multi=True (see training_proteins.py);
    # every other model already applies softmax/sigmoid internally.
    if dataset_str == 'proteins':
        out = F.softmax(out, dim=1)
    return out


@torch.no_grad()
def collect_graph_stats(dataset_str, dataset, model):
    """Returns a list of dicts: {index, class, confidence, size} for every graph."""
    stats = []
    for i in range(len(dataset)):
        data = dataset[i]
        probs = get_probs(dataset_str, model, data.x, data.edge_index)
        pred = probs.argmax(dim=1).item()
        stats.append({
            'index': i,
            'class': pred,
            'confidence': probs[0, pred].item(),
            'size': data.x.shape[0],
        })
    return stats


def assign_buckets(values, num_buckets):
    """Quantile-bin values into ~num_buckets buckets, returning a bucket id per value.
    Bins on rank rather than raw value so ties (e.g. ba2motif graphs are all the
    same size) still split into balanced buckets instead of breaking qcut."""
    ranks = pd.Series(values).rank(method='first')
    return pd.qcut(ranks, q=num_buckets, labels=False, duplicates='drop').to_numpy()


def build_buckets(stats, num_buckets):
    """3D stratification: (predicted class, confidence decile, size decile) -> [dataset indices]."""
    size_buckets = assign_buckets([s['size'] for s in stats], num_buckets)
    conf_buckets = assign_buckets([s['confidence'] for s in stats], num_buckets)

    buckets = {}
    for s, size_b, conf_b in zip(stats, size_buckets, conf_buckets):
        key = (s['class'], int(conf_b), int(size_b))
        buckets.setdefault(key, []).append(s['index'])
    return buckets


def allocate_budgets(buckets, total_dataset_size, total_budget):
    """Proportional (to bucket cardinality) budget per bucket, summing to total_budget
    via the largest-remainder method so the split is exact rather than rounded."""
    keys = list(buckets.keys())
    caps = np.array([len(buckets[k]) for k in keys])
    raw = total_budget * caps / total_dataset_size
    floors = np.minimum(np.floor(raw).astype(int), caps)

    remainder = total_budget - floors.sum()
    fractional_order = np.argsort(-(raw - floors))
    for idx in fractional_order:
        if remainder <= 0:
            break
        if floors[idx] < caps[idx]:
            floors[idx] += 1
            remainder -= 1

    return {k: int(b) for k, b in zip(keys, floors)}


class GNTKCache:
    """Precomputes each graph's S2V form, adjacency, and GNTK diagonal once so
    repeated pairwise similarity calls during farthest-point sampling don't
    redo per-graph work."""

    def __init__(self, dataset, num_layers=4, num_mlp_layers=2, jk=False, scale='degree'):
        self.dataset = dataset
        self.gntk = GNTK(num_layers=num_layers, num_mlp_layers=num_mlp_layers, jk=jk, scale=scale)
        self._cache = {}

    def _entry(self, idx):
        if idx not in self._cache:
            s2v, adj = convert_pyg_to_s2v(self.dataset[idx])
            diag = self.gntk.diag(s2v, adj)
            self._cache[idx] = (s2v, adj, diag)
        return self._cache[idx]

    def similarity(self, i, j):
        s2v_i, adj_i, diag_i = self._entry(i)
        s2v_j, adj_j, diag_j = self._entry(j)
        return self.gntk.gntk(s2v_i, s2v_j, diag_i, diag_j, adj_i, adj_j)


def farthest_point_sampling(candidate_indices, budget, gntk_cache):
    """Greedy k-center sampling over GNTK similarity: repeatedly picks the
    candidate whose worst-case (max) similarity to the already-selected set is
    smallest, maximizing structural diversity within the bucket."""
    if budget <= 0:
        return []
    if budget >= len(candidate_indices):
        return list(candidate_indices)

    candidates = list(candidate_indices)
    selected = [candidates.pop(np.random.randint(len(candidates)))]

    max_sim = {c: gntk_cache.similarity(c, selected[0]) for c in candidates}

    while len(selected) < budget and candidates:
        next_pos = min(range(len(candidates)), key=lambda k: max_sim[candidates[k]])
        next_point = candidates.pop(next_pos)
        selected.append(next_point)
        del max_sim[next_point]

        for c in candidates:
            sim = gntk_cache.similarity(c, next_point)
            if sim > max_sim[c]:
                max_sim[c] = sim

    return selected


def select_coreset(dataset_str, total_budget=TOTAL_BUDGET, num_buckets=NUM_BUCKETS):
    dataset = DATASETS[dataset_str]
    model = load_model(dataset_str, dataset)
    stats = collect_graph_stats(dataset_str, dataset, model)

    buckets = build_buckets(stats, num_buckets)
    bucket_budgets = allocate_budgets(buckets, len(dataset), total_budget)

    gntk_cache = GNTKCache(dataset)
    coreset = []
    for key, indices in buckets.items():
        budget = bucket_budgets[key]
        coreset.extend(farthest_point_sampling(indices, budget, gntk_cache))

    return np.array(sorted(coreset))


def main():
    parser = argparse.ArgumentParser(description='Select a stability-eval coreset per dataset.')
    parser.add_argument('--budget', type=int, default=TOTAL_BUDGET)
    parser.add_argument('--num_buckets', type=int, default=NUM_BUCKETS)
    parser.add_argument('--datasets', type=str, nargs='+', default=list(DATASETS.keys()),
                         help='Subset of {ba2motif, bamultishapes, proteins} to run. mutag is never included.')
    parser.add_argument('--out_dir', type=str, default='.')
    args = parser.parse_args()

    for dataset_str in args.datasets:
        print(f'Selecting coreset for {dataset_str}...')
        coreset = select_coreset(dataset_str, args.budget, args.num_buckets)
        out_path = f'{args.out_dir}/coreset_{dataset_str}.npy'
        np.save(out_path, coreset)
        print(f'  saved {len(coreset)} indices to {out_path}')


if __name__ == '__main__':
    main()
