import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# ---------------------------------------------------------
# 1. Standardized GIN Model Definition (With Normalization)
# ---------------------------------------------------------
class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=300, output_dim=1, multi=False):
        super(GIN, self).__init__()
        
        # Adding BatchNorm1d makes global_add_pool stable across variable graph sizes
        self.conv1 = GINConv(Sequential(
            Linear(input_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)
        ))
        self.conv3 = GINConv(Sequential(
            Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim)
        ))
        
        self.lin = Linear(hidden_dim, output_dim)
        self.multi = multi

    def forward(self, x=None, edge_index=None, edge_attr=None, batch=None, data=None):
        if data is not None:
            x, edge_index = data.x, data.edge_index
            if hasattr(data, 'batch'):
                batch = data.batch

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Cast to float to prevent type mismatch issues
        x = x.float()

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        x = global_add_pool(x, batch)  
        logit = self.lin(x)

        # CRITICAL FIX: For training stability, we return raw logits when multi=True.
        # We handle activation layers inside our training/evaluation functions.
        if not self.multi: 
            prob = torch.sigmoid(logit)
            return torch.cat([1 - prob, prob], dim=1)
        
        return logit 

# ---------------------------------------------------------
# 2. Dataset Loading and Preprocessing
# ---------------------------------------------------------
print("Loading PROTEINS dataset...")
dataset = TUDataset(root='datasets/TUDataset', name='PROTEINS')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:890]
test_dataset = dataset[890:]

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------------------------------------
# 3. Initialization
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GIN(
    input_dim=dataset.num_node_features, 
    hidden_dim=128, # Drop dim slightly from 300 to 128 for easier convergence on PROTEINS
    output_dim=2, 
    multi=True
).to(device)

# Drop learning rate to a stable 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ---------------------------------------------------------
# 4. Training and Evaluation Loops
# ---------------------------------------------------------
def train():
    model.train()
    total_loss = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        logits = model(data=data)
        
        # Use cross_entropy on raw logits. It combines LogSoftmax + NLLLoss stably.
        loss = F.cross_entropy(logits, data.y)
        
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
        logits = model(data=data)
        
        # Apply softmax only during inference if you need probabilities, 
        # but argmax works perfectly fine directly on raw logits!
        pred = logits.argmax(dim=1)  
        correct += int((pred == data.y).sum())

    
    return correct / len(loader.dataset)

@torch.no_grad()
def test_cm(loader):
    model.eval()
    correct = 0
    preds = []
    targets = []
    
    for data in loader:
        data = data.to(device)
        logits = model(data=data)
        
        # Apply softmax only during inference if you need probabilities, 
        # but argmax works perfectly fine directly on raw logits!
        pred = logits.argmax(dim=1)  
        correct += int((pred == data.y).sum())
        targets.append(data.y.item())
        preds.append(pred.item())
    
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    ConfusionMatrixDisplay.from_predictions(targets, preds)
    plt.show()
    
    return correct / len(loader.dataset)

# ---------------------------------------------------------
# 5. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Starting training on {device}...")
    for epoch in range(1, 101):
        loss = train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f'Epoch: {epoch:03d} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')
            
    import os
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_cm(train_loader)
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/GIN_model_PROTEINS.pt')
    print("Model saved successfully.")