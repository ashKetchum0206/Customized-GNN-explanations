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
from reward import compute_fidelity

# Configuration
NUM_SAMPLES = 100  # Number of random subgraphs to generate for each size
MAX_SIZE = 12  # Maximum subgraph size (number of edges)
DATASET = "MUTAG"  # "MUTAG" or "BA2Motif"
GRAPH_INDEX = config.graph_index # Index of the graph to analyze

# Set the dataset
dataset = ba2motif_dataset if DATASET == "BA2Motif" else mutag_dataset
# print(f"Analyzing graph {GRAPH_INDEX} from {DATASET} dataset...")

# Load the graph
data = dataset[GRAPH_INDEX]
x = data.x
edge_index = data.edge_index
edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float) if not hasattr(data, 'edge_attr') else data.edge_attr

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
print(f"Graph has {full_graph.number_of_nodes()} nodes and {full_graph.number_of_edges()} edges")

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

def compute_subgraph_matches(selected_edges):
    """Count motif matches in subgraph"""
    # Create subgraph from selected edges
    target_edge_list = torch.zeros((2, len(selected_edges)), dtype=torch.long)
    last_filled = 0
    unique_nodes = set()

    for idx, edge in enumerate(edge_list):
        if idx not in selected_edges:
            continue
        target_edge_list[0][last_filled] = edge[0]
        target_edge_list[1][last_filled] = edge[1]
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])
        last_filled += 1

    # Skip empty subgraphs
    if last_filled == 0:
        return 0

    unique_nodes = sorted(list(unique_nodes))
    mapping = {node: idx for idx, node in enumerate(unique_nodes)}

    # Remap node indices
    for edge in range(target_edge_list.shape[1]):
        target_edge_list[0][edge] = mapping[target_edge_list[0][edge].item()]
        target_edge_list[1][edge] = mapping[target_edge_list[1][edge].item()]

    target_x = x[list(unique_nodes)]
    target_graph_data = Data(
        x=target_x, 
        edge_index=target_edge_list, 
        edge_attr=edge_attr[list(selected_edges)] if hasattr(data, 'edge_attr') else None
    )
    target_graph = to_networkx_graph(target_graph_data)

    # Count matches for all query graph motifs
    total_matches = 0
    for query_name, query_graph in config.query_graphs.items():
        matcher = GraphMatcher(
            target_graph,
            query_graph,
            node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item()
        )
        
        # Count isomorphisms
        matches = len(list(matcher.subgraph_isomorphisms_iter()))
        # if matches: print(len(selected_edges), matches, query_name)
        total_matches += matches
    
    return total_matches

# Generate and analyze random subgraphs
size_results = defaultdict(list)
all_results = []
all_results_fid = []

# print(f"Generating {NUM_SAMPLES} random connected subgraphs for each size 1-{MAX_SIZE}...")
for size in range(1, MAX_SIZE + 1):
    # print(f"Processing size {size}:")
    for _ in range(NUM_SAMPLES):
        # Generate random connected subgraph
        subgraph_edges = generate_connected_random_subgraph(full_graph, edge_list, size)
        
        # Count motif matches
        matches = compute_subgraph_matches(subgraph_edges)
        
        # Record results
        size_results[size].append(matches)
        all_results.append((size, matches))
        all_results_fid.append(compute_fidelity(subgraph_edges,config.fidelity_weights))

# Analyze results by size
# print("\nResults by subgraph size:")
# print(f"{'Size':>4} | {'Min':>5} | {'Max':>5} | {'Mean':>8} | {'Std Dev':>8} | {'Matches/Edge':>12}")
# print("-" * 55)

# for size in range(1, MAX_SIZE + 1):
#     if not size_results[size]:
#         continue
        
#     matches = size_results[size]
#     min_val = min(matches)
#     max_val = max(matches)
#     mean_val = np.mean(matches)
#     std_val = np.std(matches)
#     density = mean_val / size  # Matches per edge
    
#     print(f"{size:4d} | {min_val:5d} | {max_val:5d} | {mean_val:8.2f} | {std_val:8.2f} | {density:12.2f}")

# Calculate overall statistics
all_matches = [m for _, m in all_results]
overall_mean = np.mean(all_matches)
overall_std = np.std(all_matches)

# print("\nOverall statistics:")
# print(f"Mean: {overall_mean:.2f}")
# print(f"Std Dev: {overall_std:.2f}")
# print(f"Mean + Std: {overall_mean + overall_std:.2f}")
config.interpret_mean = overall_mean 
config.interpret_std = overall_std
config.fidelity_mean = np.mean(all_results_fid)
config.fidelity_std = np.std(all_results_fid)
