"""
Reward calculation module for GNN explanation subgraph evaluation
Implements interpretability, sparsity, and fidelity metrics with configurable weighting
"""
from numpy import abs
from typing import Dict
import torch
import config
from config import model, edge_list, node_features, original_pred, edge_index
from WL_kernel import wl_subtree_kernel_similarity_grakel
from subgraph_matching import subgraph_score
from torch_geometric.data import Data
import torch.nn.functional as F
from torch.nn import CosineSimilarity
from VGAE_pyG.VGAE_sim import compare_graphs_greedy_node_matching
from utils import to_pyg_data
from gntk_combined import calculate_gntk_similarity


def similarity_common_edges(graph1, graph2):
    
    graph1 = set(graph1)
    graph2 = set(graph2)
    return len(graph1 & graph2)/len(graph1 | graph2)

def most_similar_graphs(selected_edges, alter_graphs_all, num_graphs):

    scores = []
    if(config.sim_index == 'vgae'):
        data1 = to_pyg_data(selected_edges)
        for i in range(len(alter_graphs_all)):
            scores.append((compare_graphs_greedy_node_matching(config.vgae_model, data1, to_pyg_data(alter_graphs_all[i])), i))

    elif(config.sim_index == 'kernel'): 
        graphs = [selected_edges]
        for graph in alter_graphs_all:
            graphs.append(graph)
        
        indices, similarities = wl_subtree_kernel_similarity_grakel(graphs, config.edge_index, config.node_features)
        for i in range(similarities.shape[0] - 1):
            scores.append((similarities[0, i+1], indices[i]))

    elif(config.sim_index == 'common_edges'):
        for i, alter_graph in enumerate(alter_graphs_all):
            scores.append((similarity_common_edges(selected_edges, alter_graph),i))
    
    elif(config.sim_index == 'gntk'):
        data1 = to_pyg_data(selected_edges)
        for i in range(len(alter_graphs_all)):
            scores.append((calculate_gntk_similarity(data1, to_pyg_data(alter_graphs_all[i])),i))

    scores.sort(key=lambda item: item[0], reverse = True)
   
    result = []
    for score, i in scores:
        result.append(i)

    return result[:num_graphs]
    
def similarity_score(selected_edges, metric_weights=None):

    alter_graphs = config.alter_graphs
    # edge_list = config.edge_list
    score = 0 

    if(config.sim_index == 'vgae'):
        data1 = to_pyg_data(selected_edges)
        for i in range(len(alter_graphs)):
            score += alter_graphs[i][1] * compare_graphs_greedy_node_matching(config.vgae_model, data1, config.alter_graphs_pyg[i])

    elif(config.sim_index == 'kernel'): 
        graphs = [selected_edges]
        for graph,_ in alter_graphs:
            graphs.append(graph)
        
        indices, similarities = wl_subtree_kernel_similarity_grakel(graphs, config.edge_index, config.node_features)
        for i in range(similarities.shape[0] - 1):
            score += similarities[0, i+1]
        
        return score/len(indices)
             
    elif(config.sim_index == 'common_edges'):
        for alter_graph in alter_graphs:
            score += alter_graph[1] * similarity_common_edges(selected_edges, alter_graph[0])
    
    elif(config.sim_index == 'gntk'):
        data1 = to_pyg_data(selected_edges)
        for i in range(len(alter_graphs)):
            score += alter_graphs[i][1] * calculate_gntk_similarity(data1, config.alter_graphs_pyg[i])

    return score/len(config.alter_graphs)

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
        subgraph_pred = config.model(config.node_features, config.edge_index[:, mask])[0, config.original_pred].item()
        complement_pred = config.model(config.node_features, config.edge_index[:, ~mask])[0, config.original_pred].item()

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

def combined_reward(selected_edges: torch.Tensor,
                        metric_weights: Dict[str, float]
                        
    ) -> float:

    explanation_scores = explanation_reward(selected_edges, metric_weights)
    similarity_reward = (similarity_score(selected_edges) - config.stability_mean)/config.stability_std
    return [explanation_scores[0], explanation_scores[1], explanation_scores[2], similarity_reward, explanation_scores[3] + metric_weights['stability'] * similarity_reward]
