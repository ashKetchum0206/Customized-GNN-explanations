import config
from torch_geometric.data import Data
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

# Function to convert DIG graph object to NetworkX graph
def to_networkx_graph(graph_data):
    G = nx.Graph()

    # Add nodes with attributes
    for node_idx, node_attr in enumerate(graph_data.x):
        G.add_node(node_idx, label=node_attr)  # Assuming node_attr has atomic numbers or types

    # Add edges with attributes
    for src, dst in graph_data.edge_index.t().tolist():
        G.add_edge(src, dst)

    return G

def subgraph_score(selected_edges):

    score = 0
    
    edge_list = config.edge_list
    query_graphs = config.query_graphs

    target_edge_list = torch.zeros(2,len(selected_edges))
    unique_nodes = set()
    for idx,edge in enumerate(edge_list):
        if(idx not in selected_edges): continue
        target_edge_list[idx][0] = edge[0]
        target_edge_list[idx][1] = edge[1]
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])
    
    target_x = target_x[list(unique_nodes)]
    target_graph_data = Data(x=target_x, edge_index=target_edge_list) 
    target_graph = to_networkx_graph(target_graph_data)

    for query in query_graphs:
        query_graph = to_networkx_graph(query)
        matcher = GraphMatcher(target_graph, query_graph,
                    node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item())

        score += len(matcher.subgraph_isomorphisms_iter())
    
    return score



