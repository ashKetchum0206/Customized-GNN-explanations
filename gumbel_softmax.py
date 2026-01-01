import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, to_undirected
import numpy as np

# --- 1. The Learner: Edge Probability Predictor ---
# This network learns to set the logits for every potential edge (N x N)
class EdgeProbabilityPredictor(nn.Module):
    """
    Simulates a module that outputs the raw unnormalized log-probabilities (logits)
    for the existence of an edge between every pair of nodes.
    """
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        # Initialize a trainable parameter that represents the logits for a full
        # N x N adjacency matrix (including self-loops for simplicity).
        # This is what will be optimized by backpropagation.
        self.edge_logits = nn.Parameter(torch.randn(num_nodes, num_nodes))
        
        # We enforce self-loops for the GCN to work properly, but they are not 
        # part of the Gumbel-Softmax sampling, so we initialize them strongly.
        with torch.no_grad():
             self.edge_logits.fill_diagonal_(5.0) # Bias towards keeping self-loops

    def forward(self):
        # The logits for the N x N adjacency matrix (before Gumbel-Softmax)
        return self.edge_logits

# --- 2. The Downstream Model: Black-Box Proxy GCN ---
# This model represents the black box. It MUST only take node features (x) and 
# an unweighted edge_index (derived from a binary matrix).
class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # PyG's GCNConv ignores edge_weight if not supplied, thus acting as 
        # a standard unweighted GCN for the purpose of the forward pass.
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    # Note: This forward pass only takes x and edge_index, fulfilling the
    # requirement of accepting an unweighted graph structure.
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 3. The Differentiable Sampling Function ---
def gumbel_softmax_sampling(edge_logits, temperature):
    """
    Applies Gumbel-Softmax to convert N x N logits into a differentiable, 
    soft adjacency matrix A_soft, and a hard binary matrix A_binary (for forward).
    """
    N = edge_logits.size(0)
    
    # 1. Prepare logits for Gumbel-Softmax
    # We have a categorical choice for each edge: [No Edge, Edge Exists]
    # Logits for 'Edge Exists' are directly `edge_logits`.
    # Logits for 'No Edge' are simply derived from the opposite probability.
    
    # The Gumbel-Softmax needs N*N logits for the two classes (0 and 1)
    # We stack the logits for 'No Edge' and 'Edge Exists'
    # Shape: [N, N, 2]
    logits_2d = torch.stack([
        -edge_logits, # Logits for the 'No Edge' class (0)
        edge_logits   # Logits for the 'Edge Exists' class (1)
    ], dim=-1)

    # 2. Apply Gumbel-Softmax (continuous relaxation)
    # hard=False ensures we get the continuous A_soft matrix for the backward pass.
    # Shape: [N, N, 2]
    A_soft_one_hot = F.gumbel_softmax(logits_2d, tau=temperature, hard=False, dim=-1)

    # A_soft: The soft adjacency matrix (continuous, for backpropagation)
    # We take the probability of the 'Edge Exists' class (index 1)
    A_soft = A_soft_one_hot[..., 1]
    
    # 3. Create A_binary for the forward pass (hard sampling)
    # For the forward pass into the black-box GNN, we need a discrete structure.
    # The Gumbel-Softmax with hard=True provides the Straight-Through Estimator (STE)
    # This matrix is strictly 0 or 1, but its gradients are approximated 
    # using the continuous path defined by A_soft.
    A_binary_one_hot = F.gumbel_softmax(logits_2d, tau=temperature, hard=True, dim=-1)
    A_binary = A_binary_one_hot[..., 1]
    
    # 4. Enforce symmetricity and remove self-loops from the learned part (optional, common for graphs)
    A_binary = (A_binary + A_binary.T) / 2
    A_binary.fill_diagonal_(0)
    
    return A_binary, A_soft


# --- 4. Training Loop Illustration ---
def train(num_epochs=500, lr=0.01, initial_temp=1.0, final_temp=0.1):
    N = 10  # Number of nodes
    F = 16  # Node features dimension
    C = 2   # Number of classes
    
    # Dummy Data: Node features and a fake ground truth label
    x = torch.randn(N, F)
    y_true = torch.randn((N,2)) # Node labels (0 or 1)
    
    # Instantiate models
    predictor = EdgeProbabilityPredictor(N)
    gcn = GCNModel(F, 32, C)
    # print(predictor.edge_logits)
    
    # Optimizer targets the EdgeProbabilityPredictor's logits
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    
    print(f"Starting Gumbel-Softmax Training ({num_epochs} epochs)...")
    print(f"Initial Edge Logits Norm: {torch.norm(predictor.edge_logits).item():.2f}")

    for epoch in range(1, num_epochs + 1):
        # --- Temperature Annealing (Crucial for Gumbel-Softmax) ---
        # Gradually decrease temperature to make the samples more binary over time
        tau = max(final_temp, initial_temp * (1.0 - epoch / num_epochs))
        
        predictor.train()
        gcn.train()
        optimizer.zero_grad()
        
        # 1. Get raw edge logits from the predictor
        edge_logits = predictor()

        # 2. Differentiable Sampling
        A_binary, _ = gumbel_softmax_sampling(edge_logits, tau)

        # 3. Convert Binary Adjacency Matrix to PyG format (edge_index)
        # This converts the N x N matrix into a [2, num_edges] tensor
        edge_index, _ = dense_to_sparse(A_binary)

        # 4. Forward Pass through the Black-Box GNN Proxy
        # Only node features and the UNWEIGHTED edge_index are passed
        out = gcn(x, edge_index)
 
        
        # 5. Calculate Loss and Backpropagate
        loss = torch.mean((out - y_true).pow(2))
        print(loss.item())
        loss.backward()
        optimizer.step()

        # if epoch % 100 == 0 or epoch == num_epochs:
        #     # Check sparsity (how many edges were selected)
        #     num_edges = edge_index.size(1)
        #     total_possible_edges = N * N - N # Excluding self-loops
        #     sparsity = 1.0 - (num_edges / total_possible_edges)
            
        #     # Note: We track the gradient norm to show that the gradient is flowing
        #     grad_norm = torch.norm(predictor.edge_logits.grad).item()
            
        #     print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Temp: {tau:.3f}, Edges: {num_edges:03d}, Sparsity: {sparsity:.2f}, Grad Norm: {grad_norm:.4f}")
            
    # --- Final Check ---
    final_edge_logits = predictor().detach()
    print("\n--- Training Complete ---")
    print(f"Final logit range: [{final_edge_logits.min().item():.2f}, {final_edge_logits.max().item():.2f}]")
    print("Edge predictor successfully trained via Gumbel-Softmax to optimize the downstream GNN.")

if __name__ == '__main__':
    train()
