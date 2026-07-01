import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch.utils.data import DataLoader, Dataset
from utils import to_networkx_graph, mutag_dataset, ba2motif_dataset, bamultishapes_dataset, proteins_dataset
from utils import create_mutag_query_graphs, create_ba2motif_query_graphs
from utils import create_ba2motif_query_graphs_pyg, create_mutag_query_graphs_pyg, create_proteins_query_graphs_pyg, create_bamultishapes_query_graphs_pyg 
from subgraph_matching import is_present
import config
import random

# ==========================================
# 1. Model Architecture
# ==========================================
class NeuroMatchEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(NeuroMatchEncoder, self).__init__()
        self.convs = nn.ModuleList()
        
        # Using GIN (Graph Isomorphism Network) for strong expressive power
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            nn_model = nn.Sequential(
                nn.Linear(in_channels, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(nn_model))
            
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x, edge_index, batch):
        # Message Passing
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            
        # Global pooling to get graph-level embedding
        x = global_mean_pool(x, batch)
        
        # Post-message passing MLP
        emb = self.post_mp(x)
        
        # Ensure embeddings are non-negative for order constraints (optional but helpful)
        return F.relu(emb)

# ==========================================
# 2. Order Embedding Loss Function
# ==========================================
def order_embedding_loss(z_motif, z_target, labels, margin=1.0):
    """
    Computes the max-margin loss for subgraph order embeddings.
    If M is a subgraph of G, we want z_motif <= z_target in all dimensions.
    
    Args:
        z_motif: Tensor of shape (batch_size, dim) for query motifs.
        z_target: Tensor of shape (batch_size, dim) for target graphs.
        labels: Tensor of shape (batch_size,). 1 if M is subgraph of G, 0 otherwise.
        margin: The margin for negative examples.
    """
    # Calculate the violation: how much is z_motif > z_target?
    # If z_motif <= z_target, this is 0.
    violation = torch.relu(z_motif - z_target)
    
    # Square the violations and sum across the embedding dimensions
    energy = torch.sum(violation ** 2, dim=1)
    
    # Positive pairs (labels == 1): We want energy to be 0
    loss_pos = labels * energy
    
    # Negative pairs (labels == 0): We want energy to be greater than the margin
    loss_neg = (1.0 - labels) * torch.relu(margin - energy)
    
    return torch.mean(loss_pos + loss_neg)

# ==========================================
# 3. Training Loop Example
# ==========================================
def train_neuromatch(model, optimizer, dataloader, device, epochs=50):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        # The dataloader should yield pairs of graphs (motif, target) and a label
        for batch_motif, batch_target, labels in dataloader:
            batch_motif = batch_motif.to(device)
            batch_target = batch_target.to(device)
            labels = labels.to(device).float()
            
            optimizer.zero_grad()
            
            # 1. Get embeddings
            z_motif = model(batch_motif.x, batch_motif.edge_index, batch_motif.batch)
            z_target = model(batch_target.x, batch_target.edge_index, batch_target.batch)
            
            # 2. Compute Loss
            loss = order_embedding_loss(z_motif, z_target, labels, margin=1.0)
            
            # 3. Backprop
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
        
class MotifGraphDataset(Dataset):
    def __init__(self, motifs, graphs, is_subgraph_fn, num_samples_per_epoch=1000):
        """
        Args:
            motifs: List of PyG Data objects representing query motifs.
            graphs: List of PyG Data objects representing target graphs.
            is_subgraph_fn: Function that takes (motif, graph) and returns a boolean.
            num_samples_per_epoch: Number of random pairs to generate per epoch.
        """
        self.motifs = motifs
        self.graphs = graphs
        self.is_subgraph_fn = is_subgraph_fn
        self.num_samples = num_samples_per_epoch

    def __len__(self):
        # Defines the size of one epoch
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly sample one motif and one target graph
        motif = random.choice(self.motifs)
        graph = random.choice(self.graphs)
        
        # Determine if the motif is present in the graph
        is_present = self.is_subgraph_fn(motif, graph)
        
        # Convert boolean to float label (1.0 for positive, 0.0 for negative)
        label = 1.0 if is_present else 0.0
        
        return motif, graph, label

# ==========================================
# 2. The Custom Collate Function
# ==========================================
def pair_collate(data_list):
    """
    Takes a list of tuples (motif, graph, label) and batches them.
    PyG requires its own Batch object to correctly merge disjoint graphs.
    """
    motifs, graphs, labels = zip(*data_list)
    
    # Batch the motifs and graphs separately using PyG's built-in Batch
    batch_motif = Batch.from_data_list(motifs)
    batch_target = Batch.from_data_list(graphs)
    
    # Convert labels to a standard tensor
    batch_labels = torch.tensor(labels, dtype=torch.float32)
    
    return batch_motif, batch_target, batch_labels

# ==========================================
# 4. Dummy Data Setup (To make script runnable)
# ==========================================
if __name__ == "__main__":
    
    dataset_str = 'proteins'
    if(dataset_str == 'mutag'): dataset = mutag_dataset
    elif(dataset_str == 'ba2motif'): dataset = ba2motif_dataset
    elif(dataset_str == 'bamultishapes'): dataset = bamultishapes_dataset
    elif(dataset_str == 'proteins'): dataset = proteins_dataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    query_graphs_list = []
    
    if(dataset_str == 'mutag'): 
        create_mutag_query_graphs_pyg()
        for _, value in config.query_graphs.items():
            query_graphs_list.append(value)
            
    elif(dataset_str == "ba2motif"):
        create_ba2motif_query_graphs_pyg()
        for _, value in config.query_graphs.items():
            query_graphs_list.append(value)        

    elif(dataset_str == "bamultishapes"):
        create_bamultishapes_query_graphs_pyg()
        for _, value in config.query_graphs.items():
            query_graphs_list.append(value)  
            
    elif(dataset_str == "proteins"):
        create_proteins_query_graphs_pyg()
        for key, value in config.query_graphs.items():
            query_graphs_list.append(value)
    
    
    print(query_graphs_list)
    # Hyperparameters
    NODE_FEATURE_DIM = dataset.num_node_features
    HIDDEN_DIM = 64
    EMBEDDING_DIM = 64
    BATCH_SIZE = 16
    
    data_graphs_list = []
    for i in range(len(dataset)):
        data_graphs_list.append(dataset[i])
        
    dataset = MotifGraphDataset(query_graphs_list, data_graphs_list, is_present, num_samples_per_epoch=1000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=pair_collate)
    
    # Initialize Model & Optimizer 
    model = NeuroMatchEncoder(input_dim=NODE_FEATURE_DIM, 
                              hidden_dim=HIDDEN_DIM, 
                              output_dim=EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    train_neuromatch(model, optimizer, dataloader, device, epochs=20)
    if(dataset_str == 'mutag'):torch.save(model.state_dict(), "mutag_subgraph_matching.pt")
    elif(dataset_str == 'ba2motif'):torch.save(model.state_dict(), "ba2motif_subgraph_matching.pt")
    elif(dataset_str == 'bamultishapes'):torch.save(model.state_dict(), "bamultishapes_subgraph_matching.pt")
    elif(dataset_str == 'proteins'):torch.save(model.state_dict(), "proteins_subgraph_matching.pt")
    