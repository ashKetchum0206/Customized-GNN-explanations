import networkx as nx
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from constraint import constraint
import config
import random
from torch_geometric.datasets import BAMultiShapesDataset
from torch_geometric.datasets import TUDataset


mutag_dataset = torch.load('datasets/mutag_modified.pt', weights_only=False)
ba2motif_dataset = torch.load('datasets/ba2motif.pt', weights_only=False)
bamultishapes_dataset = BAMultiShapesDataset(root='datasets/BAMultiShapes')
proteins_dataset = TUDataset(root='datasets/TUDataset', name='PROTEINS')


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

def get_induced_edges(node_state):
    
    induced_edges = []
    for i, edge in enumerate(config.edge_list):
        if(edge[0] in node_state and edge[1] in node_state): induced_edges.append(i)
    return induced_edges
        
    
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

def create_mutag_query_graphs():
    # Load the MUTAG dataset
    
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
    
def create_mutag_query_graphs_pyg():
    # Load the MUTAG dataset
    
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
        
        return submolecule

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

def create_ba2motif_query_graphs():
    # Load the BA2Motif dataset
    

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

    # Add the motifs to config
    config.query_graphs = {
        "house": create_submotif("house"),  # House-shaped motif (class 0)
        "cycle": create_submotif("cycle")   # 5-cycle motif (class 1)
    }
    
def create_ba2motif_query_graphs_pyg():
    # Load the BA2Motif dataset
    

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
        return submotif

    # Add the motifs to config
    config.query_graphs = {
        "house": create_submotif("house"),  # House-shaped motif (class 0)
        "cycle": create_submotif("cycle")   # 5-cycle motif (class 1)
    }
    
def generate_random_walk_seed(full_graph_data, seed_edge_count=5):
    """
    Generates a seed as a set of edge indices via random walk.
    """
    edges = full_graph_data.edge_index.t().tolist()
    num_edges = len(edges)
    available_edge_indices = set(range(num_edges))
    
    # Start at a random node that has at least one edge
    start_node = random.randint(0, full_graph_data.num_nodes - 1)
    
    selected_edge_indices = set()
    selected_nodes = set()
    selected_nodes.add(start_node)
    current_node = start_node
    
    # Walk until we have the required number of edges
    # We use a safety break to avoid infinite loops in tiny disconnected components
    attempts = 0
    while len(selected_edge_indices) < seed_edge_count and attempts < 100:
        # Find edges connected to the current node
        # print("Blah", current_node, len(selected_edge_indices), len(edges), attempts)
        possible_edges = [i for i in available_edge_indices if edges[i][0] == current_node or edges[i][1] == current_node]
        
        if not possible_edges: # Dead end, restart walk from random node
            current_node = random.choice(list(selected_nodes))
            attempts += 1
            continue

        edge_idx = random.choice(possible_edges)
        selected_edge_indices.add(edge_idx)
        
        # Move to the other node in the edge
        chosen_edge = edges[edge_idx]
        current_node = chosen_edge[1] if chosen_edge[0] == current_node else chosen_edge[0]
        attempts += 1
        selected_nodes.add(chosen_edge[0])
        selected_nodes.add(chosen_edge[1])
        
        available_edge_indices.remove(edge_idx)
        
    # print("Inside random walk", constraint(selected_edge_indices), len(selected_edge_indices))
    return tuple(sorted(list(selected_edge_indices)))

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data

def create_proteins_query_graphs_pyg(root_path='datasets/TUDataset'):
    # Load the PROTEINS dataset
    dataset = TUDataset(root=root_path, name='PROTEINS')

    def create_motif(node_indices, parent_graph_idx):
        """
        Extracts a specific motif from the PROTEINS dataset.
        """
        parent = dataset[parent_graph_idx]
        
        # Mapping old indices to new subgraph indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(node_indices)}
        
        # Node features for the motif
        x = parent.x[node_indices]
        
        # Edge extraction
        edge_indices = []
        for i in range(parent.edge_index.size(1)):
            src, dst = parent.edge_index[0, i].item(), parent.edge_index[1, i].item()
            if src in node_indices and dst in node_indices:
                edge_indices.append([index_map[src], index_map[dst]])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        return Data(
            x=x,
            edge_index=edge_index,
            y=parent.y,
            parent_idx=parent_graph_idx,
            original_indices=torch.tensor(node_indices)
        )

    # Dictionary of standard biological motifs in the PROTEINS dataset
    # Note: Indices are based on standard TUDataset ordering
    config.query_graphs = {
        # Sequential: Helix - Turn - Helix
        # "hth_motif": create_motif([0, 1, 2], 0), 
        
        # Sequential: Sheet - Helix - Sheet (Rossmann fold element)
        "bab_motif": create_motif([0, 1, 9], 2), # done
        
        # Sequential: Sheet - Turn - Sheet (Beta Hairpin)
        "beta_hairpin": create_motif([5, 6, 7], 3),
        
        # Topological: 3-node Clique (Possible Hydrophobic Core)
        "core_clique_helix": create_motif([3, 4, 5], 2), # done
        
        "core_clique_sheet": create_motif([14, 19, 23], 3), # done
        
        # Feature-based: Pure Alpha-Helix Bundle (4 connected helices)
        "helix_bundle": create_motif([4, 5, 6, 7], 3), # done
        
        # # Feature-based: Small Beta-Sheet segment (3 connected sheets)
        # "sheet_stack": create_motif([2, 3, 4], 7),
        
        # Topological: High-degree Hub (A central node connected to 4+ others)
        "hub_node_1": create_motif([3, 6, 7, 9, 10, 11, 23], 3), # done
        
        "hub_node_2": create_motif([12, 13, 14, 16, 18, 20, 22], 3), # done
        
        "hub_node_3": create_motif([1, 2, 3, 6, 7, 8], 4), # done
        
        "hub_node_4": create_motif([1, 6, 7, 8, 9, 10], 4), # done
        
        "hub_node_5": create_motif([0, 4, 5, 9, 10], 4), # done
        
        "hub_node_6": create_motif([12, 13, 14, 15, 16], 8), # done
        
        "hub_node_7": create_motif([2, 3, 4, 5, 9, 10], 9), # done
        
        "hub_node_8": create_motif([1, 2, 16, 17, 18, 19], 14), # done
           
    }


def create_bamultishapes_query_graphs_pyg():
    num_features = bamultishapes_dataset.num_node_features
    # If the dataset has 0 features, we default to 1.0 to give the model something to read
    feature_dim = num_features if num_features > 0 else 1

    def finalize_motif(num_nodes, edges):
        # Create features: [num_nodes, feature_dim] filled with 1.0
        x = torch.ones((num_nodes, feature_dim), dtype=torch.float32)/feature_dim
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Ensure the edge index is undirected (NeuroMatch usually expects this)
        return Data(x=x, edge_index=edge_index)

    # Define connectivity for each motif
    house_edges = [[0,1], [0,2], [1,3], [2,3], [2,4], [3,4]]
    wheel_edges = [[0,1], [0,2], [0,3], [0,4], [0,5], [1,2], [2,3], [3,4], [4,5], [5,1]]
    star_edges  = [[0,1], [0,2], [0,3], [0,4]]

    # Add reverse edges for symmetry
    def to_undirected(edges):
        return edges + [[v, u] for u, v in edges]

    config.query_graphs = {
        "house": finalize_motif(5, house_edges),
        "wheel": finalize_motif(6, wheel_edges),
        "star":  finalize_motif(5, star_edges)
    }

from torch_geometric.utils import to_networkx 
from matplotlib import pyplot as plt

def visualize_pyg_graph(data, dataset_name="Graph", node_size=800):
    """
    Visualizes a PyG Data object. Node labels show index + type.
    Node colors represent the node feature (SSE type, Atom type, etc.)
    """
    # 1. Convert PyG to NetworkX
    # We use to_undirected=True assuming these are physical/chemical structures
    G = to_networkx(data, to_undirected=True)
    
    # 2. Determine Node Types and Colors
    # Usually, data.x is a one-hot vector. argmax gives us the type index.
    node_types = []
    if data.x is not None:
        # Get index of the active feature (e.g., 0=Helix, 1=Sheet for PROTEINS)
        node_types = data.x.argmax(dim=1).tolist()
    else:
        # Fallback if no features exist
        node_types = [0] * data.num_nodes

    # Define color palette (CSS colors)
    palette = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#1A535C', '#F7FFF7', '#FF9F1C', '#95D5B2']
    node_colors = [palette[t % len(palette)] for t in node_types]

    # 3. Define Labels (Index + Type)
    # Mapping for readability based on your current datasets
    type_names = {
        'proteins': {0: "Helix", 1: "Sheet", 2: "Turn"},
        'mutag': {0: "C", 1: "N", 2: "O", 3: "F", 4: "I", 5: "Cl", 6: "Br"},
        'bamultishapes': {0: "Base", 1: "Motif"}
    }
    
    mapping = type_names.get(dataset_name.lower(), {})
    print(mapping)
    labels = {}
    for i in range(data.num_nodes):
        t_idx = node_types[i]
        print(t_idx)
        t_name = mapping.get(t_idx, f"T{t_idx}")
        print(t_name)
        labels[i] = f"{i}\n({t_name})"

    # 4. Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42) # Consistent layout
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')

    plt.title(f"Visualization: {dataset_name}", fontsize=15)
    plt.axis('off')
    plt.show()
    
# Usage
# motifs = create_bamultishapes_motifs_pyg()
# print(f"House Motif: {motifs['house']}")

if __name__ == "__main__":
    
    # root_path='datasets/TUDataset'
    # dataset = TUDataset(root=root_path, name='PROTEINS')
    dataset = bamultishapes_dataset
    parent_graph_idx = 1
    data = dataset[parent_graph_idx]
    print(data.x)
    print(data.x.shape)
    
    # visualize_pyg_graph(data, dataset_name="bamultishapes", node_size=800)
    create_bamultishapes_query_graphs_pyg()
    for query_name, query_graph in config.query_graphs.items():
        print(query_name)
        print(query_graph.x)
        print(query_graph.edge_index)
            
    # config.query_graphs = {
    #     # Sequential: Helix - Turn - Helix
    #     "hth_motif": create_motif([0, 1, 2], 0), 
        
    #     # Sequential: Sheet - Helix - Sheet (Rossmann fold element)
    #     "bab_motif": create_motif([0, 1, 2], 2), 
        
    #     # Sequential: Sheet - Turn - Sheet (Beta Hairpin)
    #     "beta_hairpin": create_motif([5, 6, 7], 3),
        
    #     # Topological: 3-node Clique (Possible Hydrophobic Core)
    #     "core_clique": create_motif([1, 2, 3], 1),
        
    #     # Feature-based: Pure Alpha-Helix Bundle (4 connected helices)
    #     "helix_bundle": create_motif([0, 1, 4, 5], 10),
        
    #     # Feature-based: Small Beta-Sheet segment (3 connected sheets)
    #     "sheet_stack": create_motif([2, 3, 4], 7),
        
    #     # Topological: High-degree Hub (A central node connected to 4+ others)
    #     "hub_node": create_motif([3, 1, 2, 4, 5], 5)
    # }
    
    # edge_list = []
    # for i in range(parent.edge_index.shape[1]):
    #     edge_list.append((parent.edge_index[0, i].item(), parent.edge_index[1, i].item()))
    
    # target_edges = []
    # nodes = (0, 1, 2)
    # for edge in edge_list:
    #     if(edge[0] in nodes and edge[1] in nodes):
    #         target_edges.append(edge)
    
    # print(parent.x[nodes, :])
    # print(target_edges)
    
    
    
    
    
    
    
    
    


