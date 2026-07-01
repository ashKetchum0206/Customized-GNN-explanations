import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import BAMultiShapesDataset
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------
# 1. User's GIN Model (With Pooling & Batch Fix Included)
# ---------------------------------------------------------
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=300, output_dim=1, multi=False):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Sequential(Linear(input_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(Sequential(Linear(hidden_dim, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)))
        self.lin = Linear(hidden_dim, output_dim)
        self.multi = multi

    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, data=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
            # Safely extract the batch vector
            if hasattr(data, 'batch'):
                batch = data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Graph-level pooling is BACK because BAMultiShapes is graph classification
        x = global_add_pool(x, batch)  

        logit = self.lin(x)

        if not self.multi: 
            return torch.cat([1-torch.sigmoid(logit), torch.sigmoid(logit)], dim=1)
        
        return F.softmax(logit, dim=1)

# ---------------------------------------------------------
# 2. Dataset Loading and Preprocessing
# ---------------------------------------------------------
print("Loading BAMultiShapes dataset...")
dataset = BAMultiShapesDataset(root='datasets/BAMultiShapes')

# Shuffle the dataset
torch.manual_seed(12345)
dataset = dataset.shuffle()

# BAMultiShapes has exactly 1000 graphs. 
# We'll use an 80/20 split (800 for training, 200 for testing).
train_dataset = dataset[:800]
test_dataset = dataset[800:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 3. Initialization
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# BAMultiShapes is binary classification (Class 0 vs Class 1)
model = GIN(
    input_dim=dataset.num_node_features, 
    hidden_dim=64, # Scaled down for faster local training
    output_dim=2, 
    multi=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------------------------
# 4. Training and Evaluation Loops
# ---------------------------------------------------------
def train():
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data=data)
        
        # NLLLoss expects log-probabilities
        loss = F.nll_loss(torch.log(out + 1e-7), data.y)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    
    for data in loader:
        data = data.to(device)
        out = model(data=data)
        pred = out.argmax(dim=1) 
        correct += int((pred == data.y).sum())
        
    return correct / len(loader.dataset)

# ---------------------------------------------------------
# 5. Execution
# ---------------------------------------------------------
print(f"Starting training on {device}...")
for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
        
torch.save(model.state_dict(), 'models/GIN_model_BA_SHAPES.pt')