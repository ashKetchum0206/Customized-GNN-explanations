import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import Data, Batch
import random

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        
        # GNN 1: Global Context Encoder
        self.global_encoder = self._build_gin(input_dim, hidden_dim)
        
        # GNN 2: Seed Encoder
        self.seed_encoder = self._build_gin(input_dim, hidden_dim)
        
        # Final MLP Predictor
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def _build_gin(self, in_dim, h_dim):
        nn_model = nn.Sequential(nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))
        return GINConv(nn_model)

    def forward(self, global_data, seed_data):
        # 1. Global context embedding
        g_x = self.global_encoder(global_data.x, global_data.edge_index)
        g_emb = global_add_pool(g_x, global_data.batch)
        
        # 2. Seed embedding
        s_x = self.seed_encoder(seed_data.x, seed_data.edge_index)
        s_emb = global_add_pool(s_x, seed_data.batch)
        
        # 3. Concatenate and predict
        combined = torch.cat([g_emb, s_emb], dim=-1)
        return self.mlp(combined).squeeze(-1)
    

def generate_random_walk_seed(full_graph_data, seed_edge_count=5):
    """
    Generates a seed as a set of edge indices via random walk.
    """
    edges = full_graph_data.edge_index.t().tolist()
    num_edges = len(edges)
    
    # Start at a random node that has at least one edge
    start_node = random.randint(0, full_graph_data.num_nodes - 1)
    
    selected_edge_indices = set()
    current_node = start_node
    
    # Walk until we have the required number of edges
    # We use a safety break to avoid infinite loops in tiny disconnected components
    attempts = 0
    while len(selected_edge_indices) < seed_edge_count and attempts < 100:
        # Find edges connected to the current node
        possible_edges = [i for i, e in enumerate(edges) if e[0] == current_node or e[1] == current_node]
        
        if not possible_edges: # Dead end, restart walk from random node
            current_node = random.randint(0, full_graph_data.num_nodes - 1)
            continue
            
        edge_idx = random.choice(possible_edges)
        selected_edge_indices.add(edge_idx)
        
        # Move to the other node in the edge
        chosen_edge = edges[edge_idx]
        current_node = chosen_edge[1] if chosen_edge[0] == current_node else chosen_edge[0]
        attempts += 1
        
    return tuple(sorted(list(selected_edge_indices)))

def build_pyg_seed(full_graph_data, edge_indices):
    """
    Converts a set of edge indices into a standalone PyG Data object.
    """
    # 1. Extract the specific edges
    edge_index = full_graph_data.edge_index[:, list(edge_indices)]
    
    # 2. Identify and remap nodes to keep the graph compact
    unique_nodes, remapped_edges = torch.unique(edge_index, return_inverse=True)
    x = full_graph_data.x[unique_nodes]
    
    return Data(x=x, edge_index=remapped_edges)

def get_eventual_reward(full_graph, seed_indices):
    """
    Mock function: In your real code, this would be the output 
    of a full MCTS run starting from this seed.
    """
    
    return torch.tensor(random.random()) # Placeholder

def train_warm_start(model, full_graphs, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for graph in full_graphs:
            # 1. Generate 5 unique seeds for this graph
            seeds_indices = [generate_random_walk_seed(graph) for _ in range(5)]
            
            # 2. Prepare data for batching
            global_list = []
            seed_list = []
            target_rewards = []
            
            for idx_tuple in seeds_indices:
                reward = get_eventual_reward(graph, idx_tuple)
                seed_pyg = build_pyg_seed(graph, idx_tuple)
                
                global_list.append(graph)
                seed_list.append(seed_pyg)
                target_rewards.append(reward)
            
            # 3. Create PyG Batches
            batch_global = Batch.from_data_list(global_list)
            batch_seed = Batch.from_data_list(seed_list)
            y = torch.stack(target_rewards)
            
            # 4. Step
            optimizer.zero_grad()
            prediction = model(batch_global, batch_seed)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(full_graphs):.4f}")

# Example Initialization
# model = ValueNetwork(input_dim=num_node_features, hidden_dim=64)
# train_warm_start(model, my_dataset)

