import math
import random
import config
from torch_geometric.data import Data
from reward import explanation_reward, similarity_score
from constraint import constraint
from model import GCN_2l, GIN
import torch
import networkx as nx
import matplotlib.pyplot as plt
from MCTS_algo import MCTS
from utils import to_networkx_graph, mutag_dataset, ba2motif_dataset
from subgraph_matching import subgraph_score
from networkx.algorithms.isomorphism import GraphMatcher
from tqdm import tqdm
import torch.nn.functional as F

dataset = ba2motif_dataset

'''
# Load the pre-trained model for MUTAG
main_model = GCN_2l()
main_model.load_state_dict(torch.load('models/GCN_model.pth', map_location=torch.device('cpu'), weights_only=True))

# Define which graph from MUTAG to analyze
graph_index = 57  # You can change this to analyze different molecules
print(f"Analyzing molecule {graph_index} from MUTAG dataset..")

# Extract data from the selected graph
x = dataset[graph_index].x
edge_index = dataset[graph_index].edge_index
# edge_attr = dataset[graph_index].edge_attr
'''

# Load the pre-trained GIN model for BA2Motif
main_model = GIN(input_dim=10)  # BA2Motif has 10 node features
main_model.load_state_dict(torch.load('models/GIN_model_BA.pth', map_location=torch.device('cpu'), weights_only=True))

# Define which graph from BA2Motif to analyze
graph_index = 42  # You can change this to analyze different graphs
print(f"Analyzing graph {graph_index} from BA2Motif dataset (Class: {dataset[graph_index].y.item()})...")

# Extract data from the selected graph
x = dataset[graph_index].x
edge_index = dataset[graph_index].edge_index

edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)
# Create edge_list from edge_index
edge_list = []
for i in range(edge_index.size(1)):
    src, dst = edge_index[0, i].item(), edge_index[1, i].item()
    edge_list.append((src, dst))

# Set edge_attr in config (needed by reward function)
config.edge_attr = edge_attr

# Define metric weights
metric_weights = {'sparse': 1, 'interpret': 1, 'fidelity': 1}
config.metric_weights = metric_weights

for query_name, query_graph in config.query_graphs.items():

    matcher = GraphMatcher(
        to_networkx_graph(dataset[graph_index]),
        query_graph,
        node_match=lambda n1, n2: torch.all(n1['label'] == n2['label']).item()
        # edge_match=lambda e1, e2: torch.allclose(e1.get('weight', torch.tensor(1.0)), e2.get('weight', torch.tensor(1.0)))
    )

    config.max_score += len(list(matcher.subgraph_isomorphisms_iter()))

# Initialize and run MCTS
config.max_edges = 12
config.allowed = range(len(edge_list))

mcts = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
            constraint, C=10, num_simulations=50, rollout_depth=100)
present_state = set()
best_subset = ()
best_reward = [0,0,0,0]

for _ in range(config.max_edges):
    result = mcts.search(present_state).state
    present_state = result
    reward = explanation_reward(present_state, metric_weights)
    if(reward[-1] >= best_reward[-1]):
        best_reward = reward
        best_subset = present_state

target_edge_list = torch.zeros((2,len(best_subset)), dtype = torch.long)
last_filled = 0 
unique_nodes = set()

for idx,edge in enumerate(edge_list):
    if(idx not in best_subset): continue
    target_edge_list[0][last_filled] = edge[0]
    target_edge_list[1][last_filled] = edge[1]
    unique_nodes.add(edge[0])
    unique_nodes.add(edge[1])
    last_filled+=1

unique_nodes = sorted(list(unique_nodes))
mapping = {}
for idx, node in enumerate(unique_nodes):
    mapping[node] = idx

for edge in range(target_edge_list.shape[1]):
    target_edge_list[0][edge] = mapping[target_edge_list[0][edge].item()]
    target_edge_list[1][edge] = mapping[target_edge_list[1][edge].item()]

target_x = config.node_features[list(unique_nodes)]
target_graph_data = Data(x=target_x, edge_index=target_edge_list, edge_attr=config.edge_attr[list(best_subset)])
# config.alter_graphs.append(target_graph_data)
config.alter_graphs.append((best_subset,best_reward[-1]))


print('Stage 1 complete.')
print(f'Interpret:{best_reward[1]}, Fidelity:{best_reward[2]}, Prob:{F.softmax(config.model(data=target_graph_data),dim = 1)[:,config.original_pred].item()}')


# Sample random graphs and get their explanations with the same user metrics preference
for i in tqdm(range(10)):

    k = 0.9
    sampled_indices = random.sample(range(len(edge_list)), int(k*len(edge_list)))
    config.allowed = sampled_indices

    present_state = set()
    best_subset = ()
    best_reward = [0,0,0,0]

    mcts = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
            constraint, C=10, num_simulations=50, rollout_depth=100)

    for _ in range(config.max_edges):
        result = mcts.search(present_state).state
        present_state = result
        reward = explanation_reward(present_state, metric_weights)
        if(reward[-1] >= best_reward[-1]):
            best_reward = reward
            best_subset = present_state

    # target_edge_list = torch.zeros((2,len(best_subset)), dtype = torch.long)
    # last_filled = 0 
    # unique_nodes = set()

    # for idx,edge in enumerate(edge_list):
    #     if(idx not in best_subset): continue
    #     target_edge_list[0][last_filled] = edge[0]
    #     target_edge_list[1][last_filled] = edge[1]
    #     unique_nodes.add(edge[0])
    #     unique_nodes.add(edge[1])
    #     last_filled+=1
    
    # unique_nodes = sorted(list(unique_nodes))
    # mapping = {}
    # for idx, node in enumerate(unique_nodes):
    #     mapping[node] = idx

    # for edge in range(target_edge_list.shape[1]):
    #     target_edge_list[0][edge] = mapping[target_edge_list[0][edge].item()]
    #     target_edge_list[1][edge] = mapping[target_edge_list[1][edge].item()]

    # target_x = config.node_features[list(unique_nodes)]
    # target_graph_data = Data(x=target_x, edge_index=target_edge_list, edge_attr=config.edge_attr[list(best_subset)])
    # config.alter_graphs.append(target_graph_data)
    config.alter_graphs.append((best_subset,best_reward[-1]))

print("Beginning Stage 2..")
# Run MCTS with updated reward function
config.allowed = range(len(edge_list))
present_state = set()
best_subset = ()
best_reward = [0,0,0,0]

mcts = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
            constraint, C=10, num_simulations=50, rollout_depth=100)

for _ in tqdm(range(config.max_edges)):
    result = mcts.search(present_state).state
    present_state = result
    reward = similarity_score(present_state)
    if(reward[-1] >= best_reward[-1]):
        best_reward = reward
        best_subset = present_state

target_edge_list = torch.zeros((2,len(best_subset)), dtype = torch.long)
last_filled = 0 
unique_nodes = set()

for idx,edge in enumerate(edge_list):
    if(idx not in best_subset): continue
    target_edge_list[0][last_filled] = edge[0]
    target_edge_list[1][last_filled] = edge[1]
    unique_nodes.add(edge[0])
    unique_nodes.add(edge[1])
    last_filled+=1

unique_nodes = sorted(list(unique_nodes))
mapping = {}
for idx, node in enumerate(unique_nodes):
    mapping[node] = idx

for edge in range(target_edge_list.shape[1]):
    target_edge_list[0][edge] = mapping[target_edge_list[0][edge].item()]
    target_edge_list[1][edge] = mapping[target_edge_list[1][edge].item()]

target_x = config.node_features[list(unique_nodes)]
target_graph_data = Data(x=target_x, edge_index=target_edge_list, edge_attr=config.edge_attr[list(best_subset)])

reward_tuple = explanation_reward(best_subset, metric_weights)
print("Stage 2 complete.")
print(f'Stability:{best_reward[0]}, Interpret:{reward_tuple[1]}, Fidelity:{reward_tuple[2]}, Prob:{F.softmax(config.model(data=target_graph_data),dim = 1)[:,config.original_pred].item()}')

# Print results
print("Best edge indices:", best_subset)
print("Selected edges:", [edge_list[i] for i in best_subset])

''' Visualize Results '''
# Create full graph (but convert tensor attributes to simple values)
data = dataset[graph_index]
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
plt.savefig(f"explanations/explanation_graph_{graph_index}.png")
print(f"Visualization saved as explanation_graph_{graph_index}.png")