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
MAX_SIZE = int(data.edge_index.shape[1] * 1)

# Create edge_list from edge_index
edge_list = []
for i in range(edge_index.size(1)):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    edge_list.append((src, dst))

# Convert to NetworkX for easier manipulation
full_graph = to_networkx_graph(data)

config.size_results_stab = {}

# print(f"Generating {NUM_SAMPLES} random connected subgraphs for each size 1-{MAX_SIZE}...")
for size in range(1, MAX_SIZE + 1):
    # print(f"Processing size {size}:")
    config.size_results_stab[size] = []
    # for _ in range(NUM_SAMPLES):
    #     # Generate random connected subgraph
    #     subgraph_edges = generate_random_walk_seed(data, size)
    #     config.size_results_stab[size].append(similarity_score(subgraph_edges))
        
# all_results_stab = []

# for size, value in config.size_results_stab.items():
#     all_results_stab.extend(value) 

# mean_stab = np.mean(all_results_stab)
# std_stab = np.std(all_results_stab)

# for size, value in config.size_results_stab.items():
#     config.size_results_stab[size] = {'mean' : mean_stab, 'std': std_stab}

# config.size_results_stab[0] = {'mean' : mean_stab, 'std': std_stab}

std = 1000 if config.sim_index == 'gntk' else 1

for size, value in config.size_results_stab.items():
    # config.size_results_stab[size] = {'mean' : np.mean(value), 'std': np.std(value)}
    config.size_results_stab[size] = {'mean' : 0, 'std': std}
    
config.size_results_stab[0] = {'mean' : 0, 'std': std}
    

 
