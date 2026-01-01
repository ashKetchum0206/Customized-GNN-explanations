import sys
import torch
from grakel.kernels import WeisfeilerLehman
from grakel import Graph
import numpy as np

# Define global edge_list and node_labels (fixed for all graphs)
edge_list_global = np.array([[0,1,2],[1,2,0]]) 
# Make sure edge_list indices match node_labels indices
node_labels_global = np.array([[0,0,1], # Node 0: label 2 (index 2 in one-hot)
                               [0,1,0], # Node 1: label 1 (index 1 in one-hot)
                               [1,0,0]]) # Node 2: label 0 (index 0 in one-hot)


def list_set_to_graph(edge_indices_subset: set, global_edge_list: np.ndarray, global_node_labels: np.ndarray):
    """
    Converts a subset of edge indices into a GraKeL Graph object.

    Args:
        edge_indices_subset (set): A set of indices referring to rows in global_edge_list.
        global_edge_list (np.ndarray): The master list of all possible edges (num_edges, 2).
        global_node_labels (np.ndarray): The master list of all node features/labels (num_nodes, num_features).

    Returns:
        grakel.Graph: A GraKeL Graph object.
    """
    # print(edge_indices_subset, global_edge_list, global_node_labels)
    # 1. Identify all unique nodes present in the selected edges
    unique_nodes = set()
    for edge_idx in edge_indices_subset:
        # Ensure edge_idx is valid
        if edge_idx >= global_edge_list.shape[1] or edge_idx < 0:
            raise ValueError(f"Edge index {edge_idx} out of bounds for global_edge_list")
        
        u, v = global_edge_list[0, edge_idx].item(), global_edge_list[1, edge_idx].item()
        unique_nodes.add(u)
        unique_nodes.add(v)
    
    # If no edges, the graph is empty
    if not unique_nodes:
        # GraKeL might have issues with truly empty graphs.
        # If this is expected, you might need to handle this case,
        # e.g., return a Graph with 0 nodes if GraKeL supports it,
        # or return a specific similarity score (e.g., 0).
        # For now, let's ensure it doesn't happen with valid inputs.
        print("Warning: Graph has no nodes or edges derived from the subset. Returning empty graph.")
        return Graph([], {})


    # 2. Create a mapping from original global node IDs to new local graph node IDs (0, 1, 2...)
    # This is important for GraKeL if original node IDs are not contiguous or start from 0.
    original_node_ids = sorted(list(unique_nodes))
    node_id_mapping = {original_id: local_id for local_id, original_id in enumerate(original_node_ids)}

    # 3. Prepare edges for GraKeL Graph object
    edges_for_grakel = []
    for edge_idx in edge_indices_subset:
        u_orig, v_orig = global_edge_list[0, edge_idx].item(), global_edge_list[1, edge_idx].item()
        u_local = node_id_mapping[u_orig]
        v_local = node_id_mapping[v_orig]
        edges_for_grakel.append((u_local, v_local))

    # 4. Prepare node labels for GraKeL Graph object
    # GraKeL expects a dictionary mapping local node IDs to their labels.
    # We are converting one-hot encoded labels to a single integer label using argmax.
    nodes_for_grakel = dict()
    for local_id, original_id in enumerate(original_node_ids):
        # Ensure original_id is within bounds for global_node_labels
        if original_id >= len(global_node_labels) or original_id < 0:
            raise ValueError(f"Original node ID {original_id} out of bounds for global_node_labels")
            
        # Get the one-hot vector and convert it to a single integer label
        label_vector = global_node_labels[original_id]
        nodes_for_grakel[local_id] = int(torch.argmax(label_vector))
        # print(f"Node original_id={original_id}, local_id={local_id}, label_vector={label_vector}, label={nodes_for_grakel[local_id]}")

    return Graph(edges_for_grakel, nodes_for_grakel)

def wl_subtree_kernel_similarity_grakel(
    graphs_edge_indices,
    global_edge_list: np.ndarray,
    global_node_labels: np.ndarray,
    n_iter: int = 5,
    normalize: bool = True,
) -> float:
    """
    Computes the Weisfeiler-Lehman subtree kernel similarity between two graphs
    defined by subsets of edge indices, using the GraKeL library.

    Args:
        graph1_edge_indices (set): A set of indices referring to rows in global_edge_list for graph 1.
        graph2_edge_indices (set): A set of indices referring to rows in global_edge_list for graph 2.
        global_edge_list (np.ndarray): The master list of all possible edges (num_edges, 2).
        global_node_labels (np.ndarray): The master list of all node features/labels (num_nodes, num_features).
        n_iter (int): The number of iterations for the WL algorithm.
        normalize (bool): Whether to normalize the kernel value.

    Returns:
        float: The WL subtree kernel similarity between the two graphs.
    """
    graph_list = []
    graph_list_indices = []
    for i, edge_indices in enumerate(graphs_edge_indices):
        try:
            g = list_set_to_graph(edge_indices, global_edge_list, global_node_labels)
            graph_list.append(g)
            graph_list_indices.append(i)
        except:
            continue

    wl_kernel = WeisfeilerLehman(n_iter=n_iter, normalize=normalize) # base_kernel=None is default but good to be explicit

    # Compute the kernel matrix
    K = wl_kernel.fit_transform(graph_list)
    return graph_list_indices, K

# --- Example Usage ---
if __name__ == "__main__":
    # Your example usage
    # graph1 refers to edges at index 0 and 1 in edge_list_global
    graph1_edges_subset = set([0, 1]) # Edges (0,1) and (1,2)
    # graph2 refers to edges at index 1 and 2 in edge_list_global
    graph2_edges_subset = set([1, 2]) # Edges (1,2) and (2,0)

    print("Comparing Graph 1 and Graph 2:")
    similarity = wl_subtree_kernel_similarity_grakel(
        [graph1_edges_subset, graph2_edges_subset], edge_list_global, node_labels_global
    )
    print(f"WL Kernel Similarity (Graph1, Graph2): {similarity}")

    # # Example: Graph with no edges (should return 0.0 or raise an error depending on GraKel behavior)
    # # This case is now handled in list_set_to_graph if unique_nodes is empty
    # graph_empty_subset = set()
    # similarity_empty = wl_subtree_kernel_similarity_grakel(
    #     graph_empty_subset, graph_empty_subset, edge_list_global, node_labels_global
    # )
    # print(f"\nWL Kernel Similarity (Empty Graph, Empty Graph): {similarity_empty}")

    # # Example: Graph with only one edge
    # graph_single_edge_subset = set([0])
    # similarity_single = wl_subtree_kernel_similarity_grakel(
    #     graph_single_edge_subset, graph_single_edge_subset, edge_list_global, node_labels_global
    # )
    # print(f"\nWL Kernel Similarity (Single Edge Graph, Single Edge Graph): {similarity_single}")