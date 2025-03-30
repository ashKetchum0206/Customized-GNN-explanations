import math
import random
import config
from torch_geometric.data import Data
from reward import explanation_reward
from constraint import constraint
from model import GCN_2l
import torch
import networkx as nx
from MCTS_algo import MCTS

def to_networkx_graph(graph_data):
    G = nx.Graph()

    # Add nodes with attributes
    for node_idx, node_attr in enumerate(graph_data.x):
        G.add_node(node_idx, label=node_attr)  # Assuming node_attr has atomic numbers or types

    # Add edges with attributes
    for src, dst in graph_data.edge_index.t().tolist():
        G.add_edge(src, dst)

    return G

x = torch.tensor([[1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0]], dtype = torch.float)

edge_list = [(0,1), (1,2), (0,2)]  # Example edge list

x_query = torch.tensor([[1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0]],
                dtype = torch.float)

edge_index_query = torch.tensor([[0],[1]], dtype = torch.long)
query = to_networkx_graph(Data(x=x_query, edge_index=edge_index_query) )
config.query_graphs.append(query)

edge_index = torch.zeros((2,len(edge_list)), dtype = torch.long)
for idx, edge in enumerate(edge_list):
    edge_index[0][idx] = edge[0]
    edge_index[1][idx] = edge[1]

metric_weights = {'sparse':1, 'interpret':1, 'fidelity':1}

main_model = GCN_2l()
mcts = MCTS(main_model, x , edge_list, edge_index, explanation_reward, metric_weights, constraint, C=1.4, num_simulations=500, rollout_depth=3)
best_subset = mcts.search()

print("Best edge indices:", best_subset)
print("Selected edges:", [edge_list[i] for i in best_subset])