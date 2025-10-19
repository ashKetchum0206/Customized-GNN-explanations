import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import Data
import config


# Function to convert graph object to NetworkX graph
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

def to_pyg_data(selected_edges):

    target_edge_list = torch.zeros(2,len(selected_edges), dtype = torch.long)
    unique_nodes = set()

    for idx,edge in enumerate(selected_edges):

        target_edge_list[0][idx] = config.edge_index[0, edge]
        target_edge_list[1][idx] = config.edge_index[1, edge]
        unique_nodes.add(config.edge_index[0, edge].item())
        unique_nodes.add(config.edge_index[1, edge].item())
  
    
    unique_nodes = sorted(list(unique_nodes))
    mapping = {}
    for idx, node in enumerate(unique_nodes):
        mapping[node] = idx

    for edge in range(target_edge_list.shape[1]):
        target_edge_list[0, edge] = mapping[target_edge_list[0, edge].item()]
        target_edge_list[1, edge] = mapping[target_edge_list[1, edge].item()]

    target_x = config.node_features[list(unique_nodes)]
    target_graph_data = Data(x=target_x, edge_index=target_edge_list, edge_attr=config.edge_attr[list(selected_edges)])
    return target_graph_data

# Function to convert to an undirected graph representation
def convert_to_undirected(dataset):
    """
    Convert MUTAG dataset to use only unique edges (undirected graph)
    
    Args:
        dataset: PyTorch Geometric dataset with redundant edges
        
    Returns:
        List of modified Data objects with unique edges
    """
    from torch_geometric.data import Data
    
    undirected_dataset = []
    
    for idx in tqdm(range(len(dataset))):
        data = dataset[idx]
        
        # Dictionary to track unique edges
        unique_edges = {}
        
        for i in range(data.edge_index.size(1)):
            src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            edge_key = (min(src, dst), max(src, dst))  # Normalize order
            
            if edge_key not in unique_edges:
                unique_edges[edge_key] = (i, src, dst)  # Store index and direction
        
        # Create new edge_index and edge_attr tensors
        new_edge_index = torch.zeros((2, len(unique_edges)), dtype=torch.long)
        new_edge_attr = torch.zeros((len(unique_edges), data.edge_attr.size(1)), dtype=data.edge_attr.dtype)
        
        # Fill the new tensors with unique edges
        for i, (_, (orig_idx, src, dst)) in enumerate(unique_edges.items()):
            new_edge_index[0, i] = src
            new_edge_index[1, i] = dst
            new_edge_attr[i] = data.edge_attr[orig_idx]
        
        # Create a new Data object with unique edges
        undirected_data = Data(
            x=data.x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=data.y
        )
        
        undirected_dataset.append(undirected_data)
    
    # print(f"Converted {len(undirected_dataset)} graphs to undirected format")
    return undirected_dataset

# Load the MUTAG dataset
mutag_dataset = torch.load('datasets/mutag_modified.pt', weights_only=False)
# mutag_dataset = torch.load('/kaggle/input/mcts-code/MCTS/datasets/mutag_modified.pt', weights_only=False)

def create_submolecule(atom_indices, parent_graph_idx):
    """
    Create a submolecule from a subset of atoms in a parent molecule
    
    Args:
        atom_indices: List of atom indices to include in the submotif
        parent_graph_idx: Index of the parent graph in the dataset
        
    Returns:
        PyG Data object representing the submolecule
    """
    parent = mutag_dataset[parent_graph_idx]
    
    # Create a mapping from original indices to new indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(atom_indices)}
    
    # Extract subgraph node features
    x = parent.x[atom_indices]
    
    # Find edges where both endpoints are in atom_indices
    edge_indices = []
    edge_attrs = []
    
    for i in range(parent.edge_index.size(1)):
        src, dst = parent.edge_index[0, i].item(), parent.edge_index[1, i].item()
        if src in atom_indices and dst in atom_indices:
            # Remap indices
            new_src = index_map[src]
            new_dst = index_map[dst]
            edge_indices.append([new_src, new_dst])
            edge_attrs.append(parent.edge_attr[i].clone())
    
    # Create new edge_index tensor
    if edge_indices:
        edge_index = torch.tensor(edge_indices).t()
        edge_attr = torch.stack(edge_attrs) if edge_attrs else None
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, parent.edge_attr.size(1)), dtype=parent.edge_attr.dtype)
    
    # Create a new Data object for the submolecule
    submolecule = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=parent.y,  # Keep the same label as parent
        parent_idx=parent_graph_idx,  # Store parent info
        original_indices=torch.tensor(atom_indices)  # Store original indices
    )
    
    return to_networkx_graph(submolecule)

config.query_graphs = {
    "nitro_group": create_submolecule([14, 15, 16], 0),  # NO₂ group
    "benzene_ring": create_submolecule([0, 1, 2, 3, 4, 5], 0),  # Benzene ring
    "napthalene": create_submolecule(list(range(0,10)), 0),  # Napthalene ring
    "anthracene": create_submolecule(list(range(0,14)), 0),  # Anthracene ring
    "pyridine": create_submolecule(list(range(2,8)), 1),  # Pyridine
    "ethyl": create_submolecule([3,12], 3),  # Ethyl chain
    "fluoro": create_submolecule([5,6], 4),  # Fluoro
    "propyl": create_submolecule([8,10,11], 7),  # Propyl chain
    "ester_group": create_submolecule([2,3,1], 7),  # Ester group
    "aromatic_oxy": create_submolecule([3,4,5], 7),  # Oxy-Benzene (or [3,4,5,6] or [3,4,5,9])
    "imidazole": create_submolecule(list(range(2,7)), 8),  # Imidazole (pseudo)
    "amino_benzene": create_submolecule([5,6,17], 11),  # Amino-Benzene
    "ketone": create_submolecule([3,11,12,13], 17),  # Ketone
    "cyanide": create_submolecule([6,7], 19),  # Cyanide
    "iodo": create_submolecule([14,16], 21),  # Iodo
    "ethene": create_submolecule([18,19], 22),  # Ethene
    "chloro": create_submolecule([2,12], 25),  # chloro
    "ether": create_submolecule([2,1,0], 38),  # Ether
    "bromo": create_submolecule([14,16], 40),  # Bromo
    "dinitro": create_submolecule([1,9], 41),  # Two-Nitrogen
    "aromatic_amine": create_submolecule([10,9,3], 48),  # Oxy-amine
    "cyclic_butyl": create_submolecule(list(range(14,20)), 51),  # Cyclic Butyl on Benzene
}

# Load the BA2Motif dataset
ba2motif_dataset = torch.load('datasets/ba2motif.pt', weights_only=False)

def create_submotif(motif_type):
    """
    Extract a motif subgraph from the BA2Motif dataset.
    
    Args:
        motif_type: String, either 'house' or 'cycle' to indicate which motif to extract
        
    Returns:
        NetworkX graph of the extracted motif
    """
    # The last 5 nodes (indices 20-24) form the motif
    motif_indices = list(range(20, 25))
    
    # Choose a graph with the correct class
    # Class 0 (index ~100): house motif
    # Class 1 (index ~600): 5-cycle motif
    parent_idx = 100 if motif_type == 'house' else 600
    parent = ba2motif_dataset[parent_idx]
    
    # Verify we have the correct graph class
    expected_class = 0 if motif_type == 'house' else 1
    if parent.y.item() != expected_class:
        # If our guess was wrong, find a graph with the correct class
        for i in range(len(ba2motif_dataset)):
            if ba2motif_dataset[i].y.item() == expected_class:
                parent = ba2motif_dataset[i]
                parent_idx = i
                break
    
    # Create a mapping from original indices to new indices (0-4)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(motif_indices)}
    
    # Extract subgraph node features
    x = parent.x[motif_indices]
    
    # Find edges where both endpoints are in the motif indices
    edge_indices = []
    
    for i in range(parent.edge_index.size(1)):
        src, dst = parent.edge_index[0, i].item(), parent.edge_index[1, i].item()
        if src in motif_indices and dst in motif_indices:
            # Remap indices to 0-4
            new_src = index_map[src]
            new_dst = index_map[dst]
            edge_indices.append([new_src, new_dst])
    
    # Create new edge_index tensor
    if edge_indices:
        edge_index = torch.tensor(edge_indices).t()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Create a new Data object for the submotif
    submotif = Data(
        x=x,
        edge_index=edge_index,
        y=parent.y,  # Keep the same label as parent
        parent_idx=parent_idx,  # Store parent info
        original_indices=torch.tensor(motif_indices)  # Store original indices
    )
    
    # Convert to NetworkX graph
    return to_networkx_graph(submotif)


# # Add the motifs to config
# config.query_graphs = {
#     "house": create_submotif("house"),  # House-shaped motif (class 0)
#     "cycle": create_submotif("cycle")   # 5-cycle motif (class 1)
# }
