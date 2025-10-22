import torch
import networkx as nx
import numpy as np
import scipy as sp
import math
from torch_geometric.data import Data

# --- Code from util.py ---
# Minimal S2VGraph class needed for GNTK
class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy float tensor, used as input to neural nets
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.node_features = node_features
        self.neighbors = []
        self.max_neighbor = 0

# --- Code from gntk.py ---
class GNTK(object):
    """
    implement the Graph Neural Tangent Kernel
    """
    def __init__(self, num_layers, num_mlp_layers, jk, scale):
        """
        num_layers: number of layers in the neural networks (including the input layer)
        num_mlp_layers: number of MLP layers
        jk: a bool variable indicating whether to add jumping knowledge
        scale: the scale used aggregate neighbors [uniform, degree]
        """
        self.num_layers = num_layers
        self.num_mlp_layers = num_mlp_layers
        self.jk = jk
        self.scale = scale
        assert(self.scale in ['uniform', 'degree'])
    
    def __next_diag(self, S):
        """
        go through one normal layer, for diagonal element
        S: covariance of last layer
        """
        diag = np.sqrt(np.diag(S))
        S = S / diag[:, None] / diag[None, :]
        S = np.clip(S, -1, 1)
        # dot sigma
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag[:, None] * diag[None, :]
        return S, DS, diag

    def __adj_diag(self, S, adj_block, N, scale_mat):
        """
        go through one adj layer
        S: the covariance
        adj_block: the adjacency relation
        N: number of vertices
        scale_mat: scaling matrix
        """
        # Ensure adj_block is sparse
        if not sp.sparse.issparse(adj_block):
            adj_block = sp.sparse.csr_matrix(adj_block)
        return adj_block.dot(S.reshape(-1)).reshape(N, N) * scale_mat

    def __next(self, S, diag1, diag2):
        """
        go through one normal layer, for all elements
        """
        S = S / diag1[:, None] / diag2[None, :]
        S = np.clip(S, -1, 1)
        DS = (math.pi - np.arccos(S)) / math.pi
        S = (S * (math.pi - np.arccos(S)) + np.sqrt(1 - S * S)) / np.pi
        S = S * diag1[:, None] / diag2[None, :]
        return S, DS
    
    def __adj(self, S, adj_block, N1, N2, scale_mat):
        """
        go through one adj layer, for all elements
        """
        # Ensure adj_block is sparse
        if not sp.sparse.issparse(adj_block):
            adj_block = sp.sparse.csr_matrix(adj_block)
        return adj_block.dot(S.reshape(-1)).reshape(N1, N2) * scale_mat
      
    def diag(self, g, A):
        """
        compute the diagonal element of GNTK for graph `g` with adjacency matrix `A`
        g: graph g (S2VGraph)
        A: adjacency matrix (numpy array)
        """
        N = A.shape[0]
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            # Ensure degrees are non-zero to avoid division by zero
            degrees = np.sum(A, axis=1)
            degrees[degrees == 0] = 1
            scale_mat = 1. / (degrees[:, None] * degrees[None, :])

        diag_list = []
        adj_block = sp.sparse.kron(A, A)

        # input covariance
        sigma = np.matmul(g.node_features, g.node_features.T)
        sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
        ntk = np.copy(sigma)
        
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma, diag = self.__next_diag(sigma)
                diag_list.append(diag)
                ntk = ntk * dot_sigma + sigma
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj_diag(sigma, adj_block, N, scale_mat)
                ntk = self.__adj_diag(ntk, adj_block, N, scale_mat)
        return diag_list

    def gntk(self, g1, g2, diag_list1, diag_list2, A1, A2):
        
        """
        compute the GNTK value \Theta(g1, g2)
        g1: graph1 (S2VGraph)
        g2: graph2 (S2VGraph)
        diag_list1, diag_list2: g1, g2's the diagonal elements of covariance matrix in all layers
        A1, A2: g1, g2's adjacency matrix (numpy array)
        """
        
        n1 = A1.shape[0]
        n2 = A2.shape[0]
        
        if self.scale == 'uniform':
            scale_mat = 1.
        else:
            # Ensure degrees are non-zero
            degrees1 = np.sum(A1, axis=1)
            degrees1[degrees1 == 0] = 1
            degrees2 = np.sum(A2, axis=0) # use axis=0 for g2 to match shape
            degrees2[degrees2 == 0] = 1
            scale_mat = 1. / (degrees1[:, None] * degrees2[None, :])
        
        adj_block = sp.sparse.kron(A1, A2)
        
        jump_ntk = 0
        sigma = np.matmul(g1.node_features, g2.node_features.T)
        jump_ntk += sigma
        sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
        ntk = np.copy(sigma)
        
        for layer in range(1, self.num_layers):
            for mlp_layer in range(self.num_mlp_layers):
                sigma, dot_sigma = self.__next(sigma, 
                                    diag_list1[(layer - 1) * self.num_mlp_layers + mlp_layer],
                                    diag_list2[(layer - 1) * self.num_mlp_layers + mlp_layer])
                ntk = ntk * dot_sigma + sigma
            jump_ntk += ntk
            # if not last layer
            if layer != self.num_layers - 1:
                sigma = self.__adj(sigma, adj_block, n1, n2, scale_mat)
                ntk = self.__adj(ntk, adj_block, n1, n2, scale_mat)
        if self.jk:
            return np.sum(jump_ntk) * 2
        else:
            return np.sum(ntk) * 2

# --- New Bridge Function ---

def convert_pyg_to_s2v(pyg_graph: Data):
    """
    Converts a PyTorch Geometric (pyg) Data object into the S2VGraph
    format and its corresponding adjacency matrix.
    
    Args:
        pyg_graph: A torch_geometric.data.Data object with
                   'x' (node features) and 'edge_index'.

    Returns:
        (S2VGraph, np.ndarray): A tuple containing the S2VGraph
                                object and its numpy adjacency matrix.
    """
    num_nodes = pyg_graph.num_nodes
    
    # Create NetworkX graph from edge_index
    # .t().cpu().numpy() converts [2, E] tensor to [E, 2] numpy array
    edges = pyg_graph.edge_index.t().cpu().numpy()
    nx_g = nx.Graph()
    nx_g.add_nodes_from(range(num_nodes))
    nx_g.add_edges_from(edges)
    
    # Get graph label (default to 0 if not present)
    label = 0
    if pyg_graph.y is not None:
        if isinstance(pyg_graph.y, torch.Tensor):
            label = pyg_graph.y.item()
        else:
            label = int(pyg_graph.y)
            
    # Get node features
    node_features = pyg_graph.x.cpu().numpy()
    
    # Create S2VGraph
    s2v_graph = S2VGraph(nx_g, label, node_features=node_features)
    
    # Get adjacency matrix
    # Ensure nodes are in order 0..N-1
    adj_matrix = nx.to_numpy_array(nx_g, nodelist=sorted(nx_g.nodes()))
    
    return s2v_graph, adj_matrix

def calculate_gntk_similarity(pyg_graph1: Data, pyg_graph2: Data, 
                              num_layers=4, num_mlp_layers=2, 
                              jk=False, scale='degree'):
    """
    Calculates the GNTK similarity between two PyG Data objects.
    
    Args:
        pyg_graph1: The first PyG graph.
        pyg_graph2: The second PyG graph.
        num_layers: GNTK 'num_layers' parameter.
        num_mlp_layers: GNTK 'num_mlp_layers' parameter.
        jk: GNTK 'jk' (jumping knowledge) parameter.
        scale: GNTK 'scale' parameter ('uniform' or 'degree').
        
    Returns:
        float: The GNTK similarity score.
    """
    
    # 1. Convert graphs to S2VGraph and Adjacency Matrix format
    s2v_g1, A1 = convert_pyg_to_s2v(pyg_graph1)
    s2v_g2, A2 = convert_pyg_to_s2v(pyg_graph2)
    
    # 2. Initialize GNTK model with default parameters
    gntk_model = GNTK(
        num_layers=num_layers,
        num_mlp_layers=num_mlp_layers,
        jk=jk,
        scale=scale
    )
    
    # 3. Compute diagonal elements for both graphs
    diag_list1 = gntk_model.diag(s2v_g1, A1)
    diag_list2 = gntk_model.diag(s2v_g2, A2)
    
    # 4. Compute the final GNTK similarity score
    similarity = gntk_model.gntk(s2v_g1, s2v_g2, diag_list1, diag_list2, A1, A2)
    
    return similarity

# --- Example Usage ---
if __name__ == "__main__":
    
    # Create Graph 1: A triangle (3 nodes)
    # Features: [1, 0], [1, 0], [0, 1]
    edge_index1 = torch.tensor([
        [0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]
    ], dtype=torch.long).t().contiguous()
    x1 = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float)
    g1 = Data(x=x1, edge_index=edge_index1, y=0)

    # Create Graph 2: A square (4 nodes)
    # Features: [1, 0], [0, 1], [1, 0], [0, 1]
    edge_index2 = torch.tensor([
        [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]
    ], dtype=torch.long).t().contiguous()
    x2 = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float)
    g2 = Data(x=x2, edge_index=edge_index2, y=1)

    # Create Graph 3: Another triangle (isomorphic to g1, different features)
    # Features: [1, 0], [1, 0], [1, 0]
    edge_index3 = torch.tensor([
        [0, 1], [1, 0], [1, 2], [2, 1], [0, 2], [2, 0]
    ], dtype=torch.long).t().contiguous()
    x3 = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ], dtype=torch.float)
    g3 = Data(x=x3, edge_index=edge_index3, y=0)

    # --- Calculate Similarities ---
    
    # Compare g1 (triangle) and g2 (square)
    sim_1_2 = calculate_gntk_similarity(g1, g2)
    print(f"GNTK Similarity (Triangle vs Square): {sim_1_2}")
    
    # Compare g1 (triangle) and g3 (triangle, different features)
    sim_1_3 = calculate_gntk_similarity(g1, g3)
    print(f"GNTK Similarity (Triangle 1 vs Triangle 2): {sim_1_3}")

    # Compare g1 with itself (should be highest)
    sim_1_1 = calculate_gntk_similarity(g1, g1)
    print(f"GNTK Similarity (Triangle 1 vs Triangle 1): {sim_1_1}")
