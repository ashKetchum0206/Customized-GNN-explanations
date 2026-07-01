import math
import random
import config
from torch_geometric.data import Data
from reward import explanation_reward, combined_reward, most_similar_graphs
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
from utils import to_pyg_data
from VGAE_pyG.model import DeepVGAE
from utils import create_mutag_query_graphs, create_ba2motif_query_graphs
from utils import create_mutag_query_graphs_pyg, create_ba2motif_query_graphs_pyg
from smooth_subgraph_matching import NeuroMatchEncoder
# --- [NEW] Import argparse for command-line arguments ---
import argparse
import os
import json
import ast
# -----------------------------------------------------

def parse_tuple(arg_string):
    try:
        # Safely evaluate the string into a Python object
        parsed = ast.literal_eval(arg_string)
        if isinstance(parsed, tuple):
            return parsed
        else:
            raise argparse.ArgumentTypeError(f"Expected a tuple, but got {type(parsed).__name__}")
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError("Invalid tuple syntax. Use format: (1, 'a', 2.5)")

# --- [NEW] Argument Parsing Functionality ---
def parse_args():
    parser = argparse.ArgumentParser(description="GNN Explanation using MCTS for a range of graphs.")
    parser.add_argument('--graph_index', type=int, required=True)
    parser.add_argument('--sim_index', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--kaggle', type=int)
    parser.add_argument('--interp_index', type=str) # learned or hard
    parser.add_argument('--initial_state', type=parse_tuple, required=True) # pass the tuple
    return parser.parse_args()

args = parse_args()
sim_index = args.sim_index
dataset_str = args.dataset
config.dataset_str = dataset_str
kaggle = args.kaggle
interp_index = args.interp_index
initial_state = args.initial_state
graph_index = args.graph_index
# ------------------------------------------

if(dataset_str == 'mutag'): dataset = mutag_dataset
elif(dataset_str == 'ba2motif'): dataset = ba2motif_dataset

metric_weights = {'sparse': 1, 'interpret': 1, 'fidelity': 1, 'stability': 1}
config.metric_weights = metric_weights
fidelity_weights = {'plus': 0.5, 'minus': 0.5}
config.fidelity_weights = fidelity_weights
config.sim_index = sim_index
config.interp_index = interp_index

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if dataset_str == 'mutag':
    main_model = GIN(input_dim = dataset[0].x.shape[1], output_dim = 2, multi=True)
    main_model.load_state_dict(torch.load('models/GIN_model_MUTAG.pth', map_location=torch.device('cpu'), weights_only=True))
    enc_in_channels = dataset[0].x.shape[1]
    enc_hidden_channels = 64
    enc_out_channels = 32
    vgae_model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
    vgae_model.load_state_dict(torch.load('VGAE_model_MUTAG.pt'))
    
    
    if(interp_index == 'learned'):
        NODE_FEATURE_DIM = dataset[0].x.shape[1]
        HIDDEN_DIM = 64
        EMBEDDING_DIM = 64
    
        matching_model = NeuroMatchEncoder(input_dim=NODE_FEATURE_DIM, 
                                hidden_dim=HIDDEN_DIM, 
                                output_dim=EMBEDDING_DIM).to(device)

        matching_model.load_state_dict(torch.load('mutag_subgraph_matching.pt'))
        config.subgraph_matching_model = matching_model
        create_mutag_query_graphs_pyg()
    elif(interp_index == 'hard'):
        create_mutag_query_graphs()
    
    with open("motif statistics/zero_statistics_mutag.json", "r") as f:
        zero_statistics = json.load(f)
    with open("motif statistics/one_statistics_mutag.json", "r") as f:
        one_statistics = json.load(f)
    config.correlation[1] = one_statistics 
    config.correlation[0] = zero_statistics

elif(dataset_str == 'ba2motif'):
    main_model = GIN(input_dim = dataset[0].x.shape[1], output_dim = 1, multi=False)
    main_model.load_state_dict(torch.load('models/GIN_model_BA.pth', map_location=torch.device('cpu'), weights_only=True))
    enc_in_channels = dataset[0].x.shape[1]
    enc_hidden_channels = 64
    enc_out_channels = 32
    vgae_model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
    vgae_model.load_state_dict(torch.load('VGAE_model_BA.pt'))

    if(interp_index == 'learned'):
        NODE_FEATURE_DIM = dataset[0].x.shape[1]
        HIDDEN_DIM = 64
        EMBEDDING_DIM = 64
    
        matching_model = NeuroMatchEncoder(input_dim=NODE_FEATURE_DIM, 
                                hidden_dim=HIDDEN_DIM, 
                                output_dim=EMBEDDING_DIM).to(device)

        matching_model.load_state_dict(torch.load('ba2motif_subgraph_matching.pt'))
        config.subgraph_matching_model = matching_model
        create_ba2motif_query_graphs_pyg()
    elif(interp_index == 'hard'):
        create_ba2motif_query_graphs()
    
    with open("motif statistics/zero_statistics_ba2motif.json", "r") as f:
        zero_statistics = json.load(f)
    with open("motif statistics/one_statistics_ba2motif.json", "r") as f:
        one_statistics = json.load(f)
    config.correlation[1] = one_statistics 
    config.correlation[0] = zero_statistics

config.vgae_model = vgae_model

net_stability = 0
net_interpret = 0 
net_fidelity = 0        
# --- [MODIFIED] Use the end_index from arguments for total graph count ---
# We calculate the actual number of graphs processed based on the input range

# The original variable name 'num_graphs' is not strictly necessary for the loop, but helpful for the final division.
# We'll use the variable 'num_graphs_to_process' for the division later.
# -------------------------------------------------------------------------
max_edges = 0
for k in range(len(dataset)):
    edge_index = dataset[k].edge_index
    max_edges = max(max_edges, edge_index.shape[1])
    

explanations = None 
done = False 
# --- [MODIFIED] Use range(begin_index, end_index) for the loop ---
for k in tqdm(range(graph_index, graph_index+1)):
# -----------------------------------------------------------------
    try:  
        config.graph_index = k
        graph_index = config.graph_index
        print(f"Analyzing graph {graph_index}")
        
        # Extract data from the selected graph
        x = dataset[graph_index].x
        edge_index = dataset[graph_index].edge_index
        edge_attr = dataset[graph_index].edge_attr
        edge_list = []
        
        config.y = dataset[graph_index].y
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_list.append((src, dst))

        # Set edge_attr in config (needed by reward function)
        config.edge_attr = edge_attr
        if config.edge_attr is None:
            config.edge_attr = torch.ones(len(edge_list),1)

        # Initialize and run MCTS
        config.max_edges = 10
        config.allowed = range(len(edge_list))

        mcts_exp = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
                    constraint, C=10, num_simulations=100, rollout_depth=100)

        exec(open("interpret_norm.py").read(), globals())
        config.alter_graphs = []
        config.alter_graphs_pyg = []

        print(f'Prediction probability:{config.original_prob}')

        present_state = set()
        best_subset = set()
        best_reward = [0,0,0,-100]

        for _ in range(config.max_edges):
            result = mcts_exp.search(present_state).state
            present_state = result
            reward = explanation_reward(present_state, metric_weights)
            if(reward[-1] >= best_reward[-1]):
                best_reward = reward
                best_subset = present_state

        target_graph_data = to_pyg_data(best_subset)
        
        config.alter_graphs_pyg.append(target_graph_data)
        config.alter_graphs.append((best_subset,best_reward[-1]))


        print('Stage 1 complete.')
        # print(f'Interpret:{best_reward[1]}, Fidelity:{best_reward[2]}, Prob:{F.softmax(config.model(data=target_graph_data),dim = 1)[:,config.original_pred].item()}')

        alter_graphs_all = []
        # Sample random graphs and get their explanations with the same user metrics preference
        for i in range(25):

            k = 0.8
            sampled_indices = random.sample(range(len(edge_list)), int(k*len(edge_list)))
            alter_graphs_all.append(sampled_indices)

        similar_graphs = most_similar_graphs(list(range(len(edge_list))), alter_graphs_all, 10) # return indices of the most similar graphs

        for similar_graph in similar_graphs:
            
            config.allowed = alter_graphs_all[similar_graph]

            present_state = set()
            best_subset = set()
            best_reward = [0,0,0,-100]

            # mcts = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
            #         constraint, C=10, num_simulations=100, rollout_depth=100)

            for _ in range(config.max_edges):
                try:
                    result = mcts_exp.search(present_state).state
                    present_state = result
                    reward = explanation_reward(present_state, metric_weights)
                    if(reward[-1] >= best_reward[-1]):
                        best_reward = reward
                        best_subset = present_state
                except:
                    break
            
            if (len(best_subset) > 0):
                config.alter_graphs.append((best_subset, best_reward[-1]))
                config.alter_graphs_pyg.append(to_pyg_data(best_subset))

        # print(f'{len(config.alter_graphs)} smoothening graphs')
        print("Beginning Stage 2..")
        # Run MCTS with updated reward function

        exec(open("stability_norm.py").read(), globals())
        print("Stability parameters estimation complete")

        config.allowed = range(len(edge_list))
        present_state = set()
        best_subset = set()
        best_reward = [0,0,0,0,-100]

        mcts_comb = MCTS(main_model, x, edge_list, edge_index, combined_reward, metric_weights, 
                    constraint, C=10, num_simulations=100, rollout_depth=100)

        for _ in range(config.max_edges):
            result = mcts_comb.search(present_state).state
            present_state = result
            reward = combined_reward(present_state, metric_weights)
            
            if(reward[-1] >= best_reward[-1]):
                best_reward = reward
                best_subset = present_state

        best_subset_main = best_subset.copy()

        net_interpret += best_reward[1] * config.interpret_std + config.interpret_mean
        net_fidelity += best_reward[2] * config.fidelity_std + config.fidelity_mean

        config.alter_graphs = []
        config.alter_graphs_pyg = []
        alter_graphs_all = []

        for i in range(25):

            k = 0.8
            sampled_indices = random.sample(range(len(edge_list)), int(k*len(edge_list)))
            alter_graphs_all.append(sampled_indices)

        similar_graphs = most_similar_graphs(list(range(len(edge_list))), alter_graphs_all, 10) # return indices of the most similar graphs
        
        for similar_graph in similar_graphs:

            config.allowed = alter_graphs_all[similar_graph]

            present_state = set()
            best_subset = set()
            best_reward = [0,0,0,-100] #arbitrary constant

            # mcts = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
            #         constraint, C=10, num_simulations=100, rollout_depth=100)

            for _ in range(config.max_edges):
                try:
                    result = mcts_exp.search(present_state).state
                    present_state = result
                    reward = explanation_reward(present_state, metric_weights)
                    if(reward[-1] >= best_reward[-1]):
                        best_reward = reward
                        best_subset = present_state
                except:
                    break
            
            if(len(best_subset) > 0):
                config.alter_graphs.append((best_subset, best_reward[-1]))
                config.alter_graphs_pyg.append(to_pyg_data(best_subset))

        net_stability += combined_reward(best_subset_main, metric_weights)[-1]
        done = True  
    
    except:
        continue

# --- [MODIFIED] Use the calculated number of graphs for division ---
if done:
    print(f"Overall score: {net_stability}")
else:
    print("No graphs were processed (begin_index >= end_index).")
# -----------------------------------------------------------------