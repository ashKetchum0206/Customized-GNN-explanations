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
    target_graph_data = Data(x=target_x, edge_index=target_edge_list)
    target_graph = to_networkx_graph(target_graph_data)


    for query_name, query_graph in query_graphs.items():

        matcher = GraphMatcher(
            target_graph,
            query_graph,
            node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item()
            # edge_match=lambda e1, e2: torch.all(e1['weight'] == e2['weight']).item()
        )

        score += len(list(matcher.subgraph_isomorphisms_iter())) * (config.correlation[config.y.item()][query_name] - config.correlation[1 - config.y.item()][query_name])
        # score += len(list(matcher.subgraph_isomorphisms_iter()))

    return score

import torch
from torch_geometric.data import Data

def subgraph_score_learned(selected_edges):
    # Edge case: No edges selected yet
    if len(selected_edges) == 0:
        return 0.0
        
    score = 0.0
    
    # ==========================================
    # 1. Efficient Subgraph Construction
    # ==========================================
    # Assuming config.edge_list is a list of [u, v] edges or an [E, 2] array.
    # We extract only the selected edges.
    selected_edge_tuples = [config.edge_list[i] for i in selected_edges]
    
    # Convert to standard PyG edge_index format: shape [2, num_selected_edges], type long
    target_edge_list = torch.tensor(selected_edge_tuples, dtype=torch.long).t().contiguous()
    
    # Vectorized node ID remapping (replaces the manual dict mapping loop)
    unique_nodes, remapped_edge_index = torch.unique(target_edge_list, return_inverse=True)
    
    # Extract features for the unique nodes present in the subgraph
    target_x = config.node_features[unique_nodes]
    
    # Create a batch vector (all zeros because this is a single graph)
    target_batch = torch.zeros(target_x.size(0), dtype=torch.long, device=target_x.device)

    # ==========================================
    # 2. Continuous Motif Matching (NeuroMatch)
    # ==========================================
    config.subgraph_matching_model.eval() # Ensure model is in evaluation mode
    
    with torch.no_grad(): # Disable gradient tracking to save memory and speed up MCTS
        
        # Pass x, edge_index, and batch explicitly
        target_embedding = config.subgraph_matching_model(target_x, remapped_edge_index, target_batch)
        
        for query_name, query_graph in config.query_graphs.items():
            
            # Ensure the query graph also has a batch vector
            if not hasattr(query_graph, 'batch') or query_graph.batch is None:
                query_batch = torch.zeros(query_graph.x.size(0), dtype=torch.long, device=query_graph.x.device)
            else:
                query_batch = query_graph.batch
                
            query_embedding = config.subgraph_matching_model(query_graph.x, query_graph.edge_index, query_batch)
            
            # Compute the continuous NeuroMatch violation energy
            # We want query_embedding <= target_embedding. Violation is how much it exceeds.
            violation = torch.relu(query_embedding - target_embedding)
            energy = torch.sum(violation ** 2)
            
            # Convert energy to a smooth soft score bounded between 0 and 1
            # (If energy is 0, score is 1. As energy increases, score decays towards 0)
            match_score = 1.0 / (1.0 + energy.item())
            
            # Apply your predefined correlation weights
            #(config.correlation[config.y.item()][query_name] - config.correlation[1 - config.y.item()][query_name])/(config.correlation[config.y.item()][query_name] + config.correlation[1 - config.y.item()][query_name])
            weight = config.correlation[config.y.item()][query_name] - config.correlation[1 - config.y.item()][query_name]
            score += match_score * weight

    return score
    
     
def is_present(query_graph, target_graph):

        matcher = GraphMatcher(
            to_networkx_graph(target_graph),
            to_networkx_graph(query_graph),
            node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item()
            # edge_match=lambda e1, e2: torch.all(e1['weight'] == e2['weight']).item()
        )
        return len(list(matcher.subgraph_isomorphisms_iter())) > 0
    



