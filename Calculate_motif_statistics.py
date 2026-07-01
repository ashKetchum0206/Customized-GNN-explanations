import math
import random
import config
from torch_geometric.data import Data
from reward import explanation_reward, combined_reward, most_similar_graphs
from constraint import constraint
from model import GCN_2l, GIN
import torch
import networkx as nx
import matplotlib.pyplot as plt
from MCTS_algo import MCTS
from utils import to_networkx_graph, mutag_dataset, ba2motif_dataset, bamultishapes_dataset, proteins_dataset
from subgraph_matching import subgraph_score
from networkx.algorithms.isomorphism import GraphMatcher
from tqdm import tqdm
import torch.nn.functional as F
from utils import to_pyg_data
from VGAE_pyG.model import DeepVGAE
from utils import create_ba2motif_query_graphs_pyg, create_mutag_query_graphs_pyg, create_proteins_query_graphs_pyg, create_bamultishapes_query_graphs_pyg 
import json

# dataset = mutag_dataset
# dataset = ba2motif_dataset
dataset = proteins_dataset
# dataset = bamultishapes_dataset


# create_mutag_query_graphs_pyg()
# create_ba2motif_query_graphs_pyg()
create_proteins_query_graphs_pyg()
# create_bamultishapes_query_graphs_pyg()

query_graphs = config.query_graphs

zero_statistics = {}
one_statistics = {}
zero_count = 0 
one_count = 0 

for query_name, _ in query_graphs.items():
    zero_statistics[query_name] = 0 
    one_statistics[query_name] = 0 

for query_name, query_graph in query_graphs.items(): 
    query_graphs[query_name] = to_networkx_graph(query_graph)

for i in range(len(dataset)):
    # Define which graph from MUTAG to analyze
    config.graph_index = i  # You can change this to analyze different molecules
    graph_index = config.graph_index
    print(f"Analyzing molecule {graph_index} from MUTAG dataset..")

    # Extract data from the selected graph
    x = dataset[graph_index].x
    edge_index = dataset[graph_index].edge_index
    print(x.shape, edge_index.shape)
    if(edge_index.shape[1] > 200): continue
    edge_attr = dataset[graph_index].edge_attr
    edge_list = []
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        edge_list.append((src, dst))

    target_graph = to_networkx_graph(dataset[graph_index])
    
    for query_name, query_graph in query_graphs.items():

        matcher = GraphMatcher(
            target_graph,
            query_graph,
            node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item()
            # edge_match=lambda e1, e2: torch.all(e1['weight'] == e2['weight']).item()
        )

        score = len(list(matcher.subgraph_isomorphisms_iter()))
        print(score)
        if(dataset[graph_index].y == 0):
            zero_statistics[query_name] += score 
            zero_count += 1 
            
        elif(dataset[graph_index].y == 1):
            one_statistics[query_name] += score 
            one_count += 1
            
for key, value in zero_statistics.items():
    zero_statistics[key] /= zero_count 
    
for key, value in one_statistics.items():
    one_statistics[key] /= one_count
    
# with open("zero_statistics_ba2motif.json", "w") as f:
#     json.dump(zero_statistics, f, indent=4)  # indent makes it pretty-printed 

# with open("zero_statistics_mutag.json", "w") as f:
#     # json.dump(zero_statistics, f, indent=4)  # indent makes it pretty-printed 

with open("zero_statistics_proteins.json", "w") as f:
    json.dump(zero_statistics, f, indent=4)  # indent makes it pretty-printed 

# with open("zero_statistics_bamultishapes.json", "w") as f:
    # json.dump(zero_statistics, f, indent=4)  # indent makes it pretty-printed 
    
# with open("one_statistics_ba2motif.json", "w") as f:
#     json.dump(one_statistics, f, indent=4)  # indent makes it pretty-printed

# with open("one_statistics_mutag.json", "w") as f:
#     json.dump(one_statistics, f, indent=4)  # indent makes it pretty-printed  

with open("one_statistics_proteins.json", "w") as f:
    json.dump(one_statistics, f, indent=4)  # indent makes it pretty-printed  

# with open("one_statistics_bamultishapes.json", "w") as f:
    # json.dump(one_statistics, f, indent=4)  # indent makes it pretty-printed  


        
    
    

   

