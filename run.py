import math
import random
import config
from torch_geometric.data import Data
from reward import explanation_reward
from constraint import constraint
from model import GCN_2l
import torch
import networkx as nx
import matplotlib.pyplot as plt
from MCTS_algo import MCTS
from utils import to_networkx_graph, mutag_dataset

print(len(config.query_graphs))

# Load the pre-trained model
main_model = GCN_2l()
main_model.load_state_dict(torch.load('GCN_model.pth', map_location=torch.device('cpu'), weights_only=True))

# Define which graph from MUTAG to analyze
graph_index = 0  # You can change this to analyze different molecules

# Extract data from the selected graph
x = mutag_dataset[graph_index].x
edge_index = mutag_dataset[graph_index].edge_index
# edge_attr = mutag_dataset[graph_index].edge_attr
edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

# Create edge_list from edge_index
edge_list = []
for i in range(edge_index.size(1)):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    edge_list.append((src, dst))

# Set edge_attr in config (needed by reward function)
config.edge_attr = edge_attr

# Define metric weights
metric_weights = {'sparse': 1, 'interpret': 1, 'fidelity': 0.5}

# Initialize and run MCTS
print(f"Analyzing molecule {graph_index} from MUTAG dataset")
mcts = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
            constraint, C=1.4, num_simulations=100, rollout_depth=len(edge_list))
best_subset = mcts.search()

# Print results
print("Best edge indices:", best_subset)
print("Selected edges:", [edge_list[i] for i in best_subset])

''' Visualize Results '''
# Create full graph (but convert tensor attributes to simple values)
data = mutag_dataset[graph_index]
G = nx.Graph()

# Add nodes with simplified attributes
for i in range(data.x.size(0)):
    # Convert tensor attributes to simple values (e.g. first element or sum)
    label = data.x[i].sum().item() if data.x[i].numel() > 0 else 0
    G.add_node(i, label=label)

# Add edges with simplified attributes
for i in range(data.edge_index.size(1)):
    src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
    # Use a simple scalar weight instead of the full tensor
    weight = 1.0
    G.add_edge(src, dst, weight=weight)

# Create explanation subgraph
explanation_edges = [edge_list[i] for i in best_subset]
explanation_graph = nx.Graph()

# Add all nodes
for node in G.nodes():
    explanation_graph.add_node(node, **G.nodes[node])

# Add only the selected edges
for src, dst in explanation_edges:
    if G.has_edge(src, dst):  # Make sure edge exists in original graph
        explanation_graph.add_edge(src, dst, weight=1.0)

# Plot both graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Graph")
pos = nx.spring_layout(G, seed=42)  # Fixed layout for consistency
nx.draw(G, pos=pos, with_labels=True, node_color='lightblue')

plt.subplot(1, 2, 2)
plt.title("Explanation Subgraph")
nx.draw(explanation_graph, pos=pos, with_labels=True, node_color='lightgreen')

plt.tight_layout()
plt.savefig(f"explanation_graph_{graph_index}.png")
print(f"Visualization saved as explanation_graph_{graph_index}.png")