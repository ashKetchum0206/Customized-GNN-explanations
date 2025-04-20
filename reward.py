"""
Reward calculation module for GNN explanation subgraph evaluation
Implements interpretability, sparsity, and fidelity metrics with configurable weighting
"""
from numpy import abs
from typing import Dict
import torch
import config
from config import model, edge_list, node_features, original_pred, edge_index
from subgraph_matching import subgraph_score
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import CosineSimilarity


def similarity(graph1, graph2):
    
    # embed1 = torch.mean(config.model(data=graph1, return_embeddings = True), dim = 0).unsqueeze(0)
    # embed2 = torch.mean(config.model(data=graph2, return_embeddings = True), dim = 0).unsqueeze(0)
    # return F.cosine_similarity(embed1, embed2).item()
    return len(graph1 & graph2)/len(graph1 | graph2)


def similarity_score(selected_edges,metric_weights=None):

    alter_graphs = config.alter_graphs
    # edge_list = config.edge_list
    score = 0 

    # target_edge_list = torch.zeros(2,len(selected_edges), dtype = torch.long)
    # last_filled = 0 
    # unique_nodes = set()

    # for idx,edge in enumerate(edge_list):
    #     if(idx not in selected_edges): continue
    #     target_edge_list[0][last_filled] = edge[0]
    #     target_edge_list[1][last_filled] = edge[1]
    #     unique_nodes.add(edge[0])
    #     unique_nodes.add(edge[1])
    #     last_filled+=1
    
    # unique_nodes = sorted(list(unique_nodes))
    # mapping = {}
    # for idx, node in enumerate(unique_nodes):
    #     mapping[node] = idx

    # for edge in range(target_edge_list.shape[1]):
    #     target_edge_list[0][edge] = mapping[target_edge_list[0][edge].item()]
    #     target_edge_list[1][edge] = mapping[target_edge_list[1][edge].item()]

    # target_x = config.node_features[list(unique_nodes)]
    # target_graph_data = Data(x=target_x, edge_index=target_edge_list, edge_attr=config.edge_attr[list(selected_edges)])

    for alter_graph in alter_graphs:
        score += alter_graph[1] * similarity(selected_edges, alter_graph[0])

    return [score]

def compute_interpretability(selected_edges: torch.Tensor) -> float:
    """
    Calculate motif-based interpretability score for selected subgraph

    Args:
        selected_edges: Tensor of edge indices forming explanation subgraph

    Returns:
        float: Motif matching score from external subgraph matching system
    """
    return subgraph_score(selected_edges)

def compute_sparsity(selected_edges: torch.Tensor) -> int:
    """
    Calculate sparsity metric as simple edge count

    Args:
        selected_edges: Tensor of edge indices forming explanation subgraph

    Returns:
        int: Number of edges in explanation subgraph
    """
    return len(selected_edges) # can try other sparsification metrics

def compute_fidelity(selected_edges: torch.Tensor, fidelity_weights: Dict[str, float]) -> float:
    """
    Calculate characterization score combining fidelity+ and fidelity-

    Args:
        selected_edges: Tensor of edge indices forming explanation subgraph
        fidelity_weights: Dictionary with 'plus' and 'minus' weights

    Returns:
        float: Combined characterization score per PyG documentation
    """
    # Create masks for subgraph and complement
    mask = torch.zeros(len(config.edge_list), dtype=torch.bool)
    mask[list(selected_edges)] = True
    
    with torch.no_grad():
        subgraph_pred = config.model(config.node_features, config.edge_index[:, mask])[0,config.original_pred].item()
        complement_pred = config.model(config.node_features, config.edge_index[:, ~mask])[0,config.original_pred].item()

    # Calculate fidelity components
    fidelity_plus = abs(config.original_prob - complement_pred)
    fidelity_minus = abs(config.original_prob - subgraph_pred)

    # Characterization formula
    numerator = fidelity_weights['plus'] + fidelity_weights['minus']
    denominator = (fidelity_weights['plus'] / (fidelity_plus + 1e-2)) + \
                  (fidelity_weights['minus'] / (1 - fidelity_minus + 1e-2))
    
    return numerator / denominator

def explanation_reward (selected_edges: torch.Tensor,
                        metric_weights: Dict[str, float] = config.metric_weights,
                        fidelity_weights: Dict[str, float] = {'plus': 0.7, 'minus': 0.3},
    ) -> float:
    """
    Calculate combined reward score for explanation subgraph

    Args:
        selected_edges: Tensor of edge indices forming explanation subgraph
        metric_weights: Dictionary with weights for:
            - 'interpret': Motif matching importance
            - 'sparse': Sparsity importance 
            - 'fidelity': Fidelity importance
        fidelity_weights: Dictionary with weights for:
            - 'plus': Fidelity+ component weight
            - 'minus': Fidelity- component weight

    Returns:
        float: Combined reward score. Returns -inf for empty subgraphs

    Raises:
        ValueError: If input tensors have mismatched dimensions
    """
    if len(selected_edges) == 0:
        return -float('inf')

    # Normalization constants (adjust based on dataset statistics)
    norm = {
        'expected_motifs': config.interpret_norm,        
        'max_edges': len(config.edge_list),          
        'baseline_fidelity': 1.0            # already scaled
    }

    # Calculate metric components
    interpret = compute_interpretability(selected_edges)
    sparsity = compute_sparsity(selected_edges)
    fidelity = compute_fidelity(selected_edges, config.fidelity_weights)

    # Normalize metrics
    interpret_norm = (interpret - config.interpret_mean)/config.interpret_std
    sparsity_norm = sparsity / norm['max_edges']
    fidelity_norm = (fidelity - config.fidelity_mean)/config.fidelity_std

    # print(interpret_norm, sparsity_norm, fidelity_norm)

    # Combine weighted metrics
    return (
        sparsity_norm,
        interpret_norm,
        fidelity_norm,
        -metric_weights['sparse'] * sparsity_norm+
        metric_weights['interpret'] * interpret_norm+
        metric_weights['fidelity'] * fidelity_norm
    )
