import config
from torch_geometric.data import Data
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import torch
from utils import to_networkx_graph

# Function to compute the score of a subgraph based on the number of query graphs it matches
def subgraph_score(selected_edges):

    score = 0
    
    edge_list = config.edge_list
    query_graphs = config.query_graphs

    target_edge_list = torch.zeros(2,len(selected_edges))
    last_filled = 0 
    unique_nodes = set()

    for idx,edge in enumerate(edge_list):
        if(idx not in selected_edges): continue
        target_edge_list[0][last_filled] = edge[0]
        target_edge_list[1][last_filled] = edge[1]
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])
        last_filled+=1
    
    unique_nodes = sorted(list(unique_nodes))
    mapping = {}
    for idx, node in enumerate(unique_nodes):
        mapping[node] = idx

    for edge in range(target_edge_list.shape[1]):
        target_edge_list[0][edge] = mapping[target_edge_list[0][edge].item()]
        target_edge_list[1][edge] = mapping[target_edge_list[1][edge].item()]

    target_x = config.node_features[list(unique_nodes)]
    target_graph_data = Data(x=target_x, edge_index=target_edge_list, edge_attr=config.edge_attr[list(selected_edges)])
    target_graph = to_networkx_graph(target_graph_data)


    for query_name, query_graph in query_graphs.items():

        matcher = GraphMatcher(
            target_graph,
            query_graph,
            node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item(),
            edge_match=lambda e1, e2: torch.allclose(e1.get('weight', torch.tensor(1.0)), e2.get('weight', torch.tensor(1.0)))
        )

        score += len(list(matcher.subgraph_isomorphisms_iter()))

    return score



