"""
Reward calculation module for GNN explanation subgraph evaluation
Implements interpretability, sparsity, and fidelity metrics with configurable weighting
"""

from typing import Dict
import torch
from config import model, edge_list, node_features, original_pred
from subgraph_matching import subgraph_score

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
    return selected_edges.size(0) # can try other sparsification metrics

def compute_fidelity(selected_edges: torch.Tensor, fidelity_weights: Dict[str, float]) -> float:
    """
    Calculate PyG-style characterization score combining fidelity+ and fidelity-

    Args:
        selected_edges: Tensor of edge indices forming explanation subgraph
        fidelity_weights: Dictionary with 'plus' and 'minus' weights

    Returns:
        float: Combined characterization score per PyG documentation
    """
    # Create masks for subgraph and complement
    mask = torch.zeros(edge_list.size(1), dtype=torch.bool)
    mask[selected_edges] = True
    
    with torch.no_grad():
        subgraph_pred = model(node_features, edge_list[:, mask])
        complement_pred = model(node_features, edge_list[:, ~mask])

    # Calculate fidelity components
    fidelity_plus = (original_pred - complement_pred).abs().item()
    fidelity_minus = (original_pred - subgraph_pred).abs().item()

    # PyG characterization formula
    numerator = fidelity_weights['plus'] + fidelity_weights['minus']
    denominator = (fidelity_weights['plus'] / fidelity_plus) + \
                  (fidelity_weights['minus'] / (1 - fidelity_minus))
    
    return numerator / denominator

def explanation_reward (selected_edges: torch.Tensor,
                        metric_weights: Dict[str, float],
                        fidelity_weights: Dict[str, float] = {'plus': 0.5, 'minus': 0.5},
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
    if selected_edges.size(0) == 0:
        return -float('inf')

    # Normalization constants (adjust based on dataset statistics)
    norm = {
        'max_motif': 1.0,        # define later
        'max_edges': 1,          # define later
        'baseline_fidelity': 1.0 # define later
    }

    # Calculate metric components
    interpret = compute_interpretability(selected_edges)
    sparsity = compute_sparsity(selected_edges)
    fidelity = compute_fidelity(selected_edges, fidelity_weights)

    # Normalize metrics
    interpret_norm = interpret / norm['max_motif']
    sparsity_norm = sparsity / norm['max_edges']
    fidelity_norm = fidelity / norm['baseline_fidelity']

    # Combine weighted metrics
    return (
        - metric_weights['sparse'] * sparsity_norm + 
        metric_weights['interpret'] * interpret_norm +
        metric_weights['fidelity'] * fidelity_norm
    )
