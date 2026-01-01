"""
Analyze random subgraphs for interpretability normalization
Generates random connected subgraphs of different sizes and evaluates motif matching
"""
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
from networkx.algorithms.isomorphism import GraphMatcher
from torch_geometric.data import Data

import config
from utils import to_networkx_graph, ba2motif_dataset, mutag_dataset
from reward import compute_fidelity, similarity_score

# Configuration
NUM_SAMPLES = 25  # Number of random subgraphs to generate for each size
GRAPH_INDEX = config.graph_index # Index of the graph to analyze

# Set the dataset
dataset = mutag_dataset if config.dataset_str == "mutag" else ba2motif_dataset
# print(f"Analyzing graph {GRAPH_INDEX} from {DATASET} dataset...")

# Load the graph
data = dataset[GRAPH_INDEX]
x = data.x
edge_index = data.edge_index
edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float) if not hasattr(data, 'edge_attr') else data.edge_attr
MAX_SIZE = int(data.edge_index.shape[1] * 0.7)

'''
# Extract subgraph with nodes 0-5
target_nodes = list(range(6))  # Nodes 0 through 5

# Find edges where both endpoints are in our target nodes
target_edge_indices = []

for i, (src, dst) in enumerate(zip(edge_index[0], edge_index[1])):
    src_item, dst_item = src.item(), dst.item()
    if src_item in target_nodes and dst_item in target_nodes:
        target_edge_indices.append(i)

# Create new edge_index tensor with only the selected edges
edge_index = edge_index[:, target_edge_indices]

# Get node features for the target nodes
x = x[target_nodes]

# Get edge attributes if they exist
edge_attr = edge_attr[target_edge_indices] if hasattr(data, 'edge_attr') else edge_attr[target_edge_indices]

# Create the new Data object
data = Data(
    x=x,
    edge_index=edge_index,
    edge_attr=edge_attr
)
'''

# Create edge_list from edge_index
edge_list = []
for i in range(edge_index.size(1)):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    edge_list.append((src, dst))

# Convert to NetworkX for easier manipulation
full_graph = to_networkx_graph(data)

def generate_connected_random_subgraph(graph, edge_list, size):
    """Generate a random connected subgraph with 'size' edges"""
    if size <= 0 or size > len(edge_list):
        return set()
    
    # Start with a random edge
    all_edges = list(range(len(edge_list)))
    random.shuffle(all_edges)
    selected_edges = {all_edges[0]}
    
    # Keep track of connected nodes
    edge = edge_list[all_edges[0]]
    connected_nodes = {edge[0], edge[1]}
    
    # Expand subgraph by adding connected edges
    attempts = 0
    max_attempts = len(edge_list) * 5  # Avoid infinite loop
    
    while len(selected_edges) < size and attempts < max_attempts:
        # Find candidate edges that would maintain connectivity
        candidates = []
        for i, edge_idx in enumerate(all_edges):
            if edge_idx in selected_edges:
                continue
                
            src, dst = edge_list[edge_idx]
            # Edge is valid if at least one endpoint is in our connected set
            if src in connected_nodes or dst in connected_nodes:
                candidates.append((i, edge_idx))
        
        if not candidates:
            # No valid candidates, break
            break
            
        # Choose a random valid edge
        chosen_idx, chosen_edge_idx = random.choice(candidates)
        selected_edges.add(chosen_edge_idx)
        
        # Update connected nodes
        src, dst = edge_list[chosen_edge_idx]
        connected_nodes.add(src)
        connected_nodes.add(dst)
        
        # Remove the chosen edge from candidates
        all_edges.pop(chosen_idx)
        
        attempts += 1
    
    if len(selected_edges) < size:
        print(f"Warning: Could only generate subgraph with {len(selected_edges)} edges, not {size}")
    
    return selected_edges

all_results_stab = []

# print(f"Generating {NUM_SAMPLES} random connected subgraphs for each size 1-{MAX_SIZE}...")
for size in range(1, MAX_SIZE + 1):
    # print(f"Processing size {size}:")
    for _ in range(NUM_SAMPLES):
        # Generate random connected subgraph
        subgraph_edges = generate_connected_random_subgraph(full_graph, edge_list, size)
        all_results_stab.append(similarity_score(subgraph_edges))

config.stability_mean = np.mean(all_results_stab)
config.stability_std = np.std(all_results_stab)
 
