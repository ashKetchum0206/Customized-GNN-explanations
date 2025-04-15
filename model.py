from torch import nn
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.conv import GCNConv
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GINConv, global_add_pool


class GCN_2l(nn.Module):

    def __init__(self, in_features=7, hidden_dim=300):
        """
        GCN with 2 layers that can handle different input feature dimensions.
        
        Args:
            in_features: Number of input node features (7 for MUTAG, 10 for BA2Motif)
            hidden_dim: Hidden dimension size
        """
        super(GCN_2l, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim)])
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU()])
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, data=None, return_embeddings=False, num_classes=2):
        # Handling data input if provided
        if data is not None:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply first convolutional layer
        x = self.conv1(x, edge_index, edge_weight=edge_attr)  # Pass edge_attr to the conv layer if supported
        x = self.relu1(x)

        # Pass through additional layers
        for conv, relu in zip(self.convs, self.relus):
            x = conv(x, edge_index, edge_weight=edge_attr)  # Pass edge_attr to conv layers if supported
            x = relu(x)

        # Handle batch if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Return embeddings if requested
        if return_embeddings:
            return x

        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.ffn(x)

        return F.softmax(x, dim=1)  # Convert logits to probabilities
    

# Define GIN model
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=300, output_dim=1,multi=False):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.lin = Linear(hidden_dim, output_dim)
        self.multi = multi

    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, data=None):
        # Handling data input if provided
        if data is not None:
            x, edge_index = data.x, data.edge_index

        # Handle batch if not provided
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)  # Graph-level pooling

        logit = self.lin(x)

        if not self.multi: return torch.cat([1-torch.sigmoid(logit), torch.sigmoid(logit)], dim=1)
        
        return F.softmax(logit, dim = 1)

        