from torch import nn
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.conv import GCNConv
import torch


class GCN_2l(nn.Module):

    def __init__(self):
        super(GCN_2l, self).__init__()
        self.conv1 = GCNConv(7, 300)
        self.convs = nn.ModuleList([GCNConv(300, 300)])
        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU()])
        self.ffn = nn.Sequential(
            nn.Linear(300, 2)
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x=None, edge_index=None, batch=None, data=None, return_embeddings=False, num_classes=2):
    # def forward(self, data=None, edge_mask=None, return_embeddings=False)
        if data is not None:
            x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.relu1(x)
        for conv, relu in zip(self.convs, self.relus):
            x = conv(x, edge_index)
            x = relu(x)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if return_embeddings:
            return x  # Return embeddings for explainability

        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.ffn(x)
        return x