import config
from torch_geometric.data import Data
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import torch

# Function to convert DIG graph object to NetworkX graph
def to_networkx_graph(graph_data):
    G = nx.Graph()

    # Add nodes with attributes
    for node_idx, node_attr in enumerate(graph_data.x):
        G.add_node(node_idx, label=node_attr)  # Assuming node_attr contains node features

    # Add edges with attributes
    edge_features = graph_data.edge_attr if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else None

    for edge_idx, (src, dst) in enumerate(graph_data.edge_index.t().tolist()):
        if edge_features is not None:
            G.add_edge(src, dst, weight=edge_features[edge_idx])  # Assuming edge_attr stores edge features
        else:
            G.add_edge(src, dst)

    return G

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


    for query in query_graphs:

        matcher = GraphMatcher(
            target_graph,
            query,
            node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item(),
            edge_match=lambda e1, e2: e1['weight'].item() == e2['weight'].item()
        )

        score += len(list(matcher.subgraph_isomorphisms_iter()))

    return score



