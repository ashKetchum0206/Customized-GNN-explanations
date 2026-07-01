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
from utils import to_networkx_graph, ba2motif_dataset, mutag_dataset, generate_random_walk_seed
from reward import compute_fidelity, similarity_score
from subgraph_matching import subgraph_score_learned

if(config.dataset_str == 'bamultishapes'):
    INTERPRET_SCALING = 1000 
else:
    INTERPRET_SCALING = 1 
    
FIDELITY_SCALING = 0.1 

# Configuration
NUM_SAMPLES = 0  # Number of random subgraphs to generate for each size
GRAPH_INDEX = config.graph_index # Index of the graph to analyze

# Set the dataset
if(dataset_str == 'mutag'): dataset = mutag_dataset
elif(dataset_str == 'ba2motif'): dataset = ba2motif_dataset
elif(dataset_str == 'bamultishapes'): dataset = bamultishapes_dataset
elif(dataset_str == 'proteins'): dataset = proteins_dataset
# print(f"Analyzing graph {GRAPH_INDEX} from {DATASET} dataset...")

# Load the graph
data = dataset[GRAPH_INDEX]
x = data.x
edge_index = data.edge_index
edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float) if not hasattr(data, 'edge_attr') else data.edge_attr
MAX_SIZE = int(data.edge_index.shape[1] * 1) # Maximum subgraph size (number of edges)

# Create edge_list from edge_index
edge_list = []
for i in range(edge_index.size(1)):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    edge_list.append((src, dst))

# Convert to NetworkX for easier manipulation
full_graph = to_networkx_graph(data)
print(f"Graph has {full_graph.number_of_nodes()} nodes and {full_graph.number_of_edges()} edges")

# Generate and analyze random subgraphs
config.size_results_inter = {}
config.size_results_fid = {}
all_results_inter = []
all_results_fid = []


# print(f"Generating {NUM_SAMPLES} random connected subgraphs for each size 1-{MAX_SIZE}...")
for size in range(1, MAX_SIZE + 1):
    # print(f"Processing size {size}:")
    config.size_results_inter[size] = []
    config.size_results_fid[size] = []
    # for _ in range(NUM_SAMPLES):
    #     # Generate random connected subgraph
    #     subgraph_edges = generate_random_walk_seed(data, size)
        
    #     # Count motif matches
    #     if(config.interp_index == 'hard'): matches = compute_subgraph_matches(subgraph_edges)
    #     elif(config.interp_index == 'learned'): matches = subgraph_score_learned(subgraph_edges)
        
    #     # Record results
    #     config.size_results_inter[size].append(matches)
    #     config.size_results_fid[size].append(compute_fidelity(subgraph_edges, config.fidelity_weights))
        


# all_results_inter = []
# all_results_fid = []

# for size, value in config.size_results_inter.items():
#     all_results_inter.extend(value) 

# for size, value in config.size_results_fid.items():
#     all_results_fid.extend(value)

# mean_inter = np.mean(all_results_inter)
# std_inter = np.std(all_results_inter)

# mean_fid = np.mean(all_results_fid)
# std_fid = np.std(all_results_fid)

# for size, value in config.size_results_inter.items():
#     config.size_results_inter[size] = {'mean' : mean_inter, 'std': std_inter}

# for size, value in config.size_results_fid.items():
#     config.size_results_fid[size] = {'mean' : mean_fid, 'std': std_fid}


# config.size_results_inter[0] = {'mean' : mean_inter, 'std': std_inter}
# config.size_results_fid[0] = {'mean' : mean_fid, 'std': std_fid}
    

for size, value in config.size_results_inter.items():
    # config.size_results_inter[size] = {'mean' : np.mean(value), 'std': np.std(value)}
    config.size_results_inter[size] = {'mean' : 0, 'std': INTERPRET_SCALING}
    
for size, value in config.size_results_fid.items():
    # config.size_results_fid[size] = {'mean' : np.mean(value), 'std': np.std(value)}
    config.size_results_fid[size] = {'mean' : 0, 'std': FIDELITY_SCALING}
    
config.size_results_fid[0] = {'mean' : 0, 'std': INTERPRET_SCALING}
config.size_results_inter[0] = {'mean' : 0, 'std': FIDELITY_SCALING}



 
