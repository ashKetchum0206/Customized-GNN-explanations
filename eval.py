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
from utils import to_networkx_graph, mutag_dataset, ba2motif_dataset, bamultishapes_dataset, proteins_dataset
from subgraph_matching import subgraph_score
from networkx.algorithms.isomorphism import GraphMatcher
from tqdm import tqdm
import torch.nn.functional as F
from utils import to_pyg_data
from VGAE_pyG.model import DeepVGAE
from utils import create_mutag_query_graphs, create_ba2motif_query_graphs
from utils import create_ba2motif_query_graphs_pyg, create_mutag_query_graphs_pyg, create_proteins_query_graphs_pyg, create_bamultishapes_query_graphs_pyg, generate_random_walk_seed
from utils import get_induced_edges
from smooth_subgraph_matching import NeuroMatchEncoder
from training_proteins import GIN as GIN_proteins
# --- [NEW] Import argparse for command-line arguments ---
import argparse
import os
import json
import numpy as np
# -----------------------------------------------------

# --- [NEW] Argument Parsing Functionality ---
def parse_args():
    parser = argparse.ArgumentParser(description="GNN Explanation using MCTS for a range of graphs.")
    parser.add_argument('--begin', type=int, default=0,
                        help='The starting index (inclusive) of the graph range to analyze.')
    parser.add_argument('--end', type=int, default=80,
                        help='The ending index (exclusive) of the graph range to analyze.')
    parser.add_argument('--sim_index', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--kaggle', type=int)
    parser.add_argument('--interp_index', type=str) # learned or hard
    parser.add_argument('--max_edges', type=int)
    return parser.parse_args()

args = parse_args()
begin_index = args.begin
end_index = args.end
sim_index = args.sim_index
dataset_str = args.dataset
config.dataset_str = dataset_str
kaggle = args.kaggle
interp_index = args.interp_index
# ------------------------------------------

if(dataset_str == 'mutag'): dataset = mutag_dataset
elif(dataset_str == 'ba2motif'): dataset = ba2motif_dataset
elif(dataset_str == 'bamultishapes'): dataset = bamultishapes_dataset
elif(dataset_str == 'proteins'): dataset = proteins_dataset


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
    
elif(dataset_str == 'proteins'):
    main_model = GIN_proteins(
    input_dim=dataset.num_node_features, 
    hidden_dim=128, # Drop dim slightly from 300 to 128 for easier convergence on PROTEINS
    output_dim=2, 
    multi=True
    )
    main_model.load_state_dict(torch.load('models/GIN_model_PROTEINS.pt', map_location=torch.device('cpu'), weights_only=True))
    enc_in_channels = dataset[0].x.shape[1]
    enc_hidden_channels = 64
    enc_out_channels = 32
    vgae_model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
    vgae_model.load_state_dict(torch.load('VGAE_model_PROTEINS.pt'))

    if(interp_index == 'learned'):
        NODE_FEATURE_DIM = dataset[0].x.shape[1]
        HIDDEN_DIM = 64
        EMBEDDING_DIM = 64
    
        matching_model = NeuroMatchEncoder(input_dim=NODE_FEATURE_DIM, 
                                hidden_dim=HIDDEN_DIM, 
                                output_dim=EMBEDDING_DIM).to(device)

        matching_model.load_state_dict(torch.load('proteins_subgraph_matching.pt'))
        config.subgraph_matching_model = matching_model
        create_proteins_query_graphs_pyg()
    elif(interp_index == 'hard'):
        create_ba2motif_query_graphs()
    
    with open("motif statistics/zero_statistics_proteins.json", "r") as f:
        zero_statistics = json.load(f)
    with open("motif statistics/one_statistics_proteins.json", "r") as f:
        one_statistics = json.load(f)
    config.correlation[1] = one_statistics 
    config.correlation[0] = zero_statistics
    
elif(dataset_str == 'bamultishapes'):
    
    main_model = GIN(
    input_dim=dataset.num_node_features, 
    hidden_dim=64, # Scaled down for faster local training
    output_dim=2, 
    multi=True)
    
    main_model.load_state_dict(torch.load('models/GIN_model_BA_SHAPES.pt', map_location=torch.device('cpu'), weights_only=True))
    enc_in_channels = dataset[0].x.shape[1]
    enc_hidden_channels = 64
    enc_out_channels = 32
    vgae_model = DeepVGAE(enc_in_channels, enc_hidden_channels, enc_out_channels).to(device)
    vgae_model.load_state_dict(torch.load('VGAE_model_BAMultiShapes.pt'))

    if(interp_index == 'learned'):
        NODE_FEATURE_DIM = dataset[0].x.shape[1]
        HIDDEN_DIM = 64
        EMBEDDING_DIM = 64
    
        matching_model = NeuroMatchEncoder(input_dim=NODE_FEATURE_DIM, 
                                hidden_dim=HIDDEN_DIM, 
                                output_dim=EMBEDDING_DIM).to(device)

        matching_model.load_state_dict(torch.load('bamultishapes_subgraph_matching.pt'))
        config.subgraph_matching_model = matching_model
        create_bamultishapes_query_graphs_pyg()
    elif(interp_index == 'hard'):
        create_ba2motif_query_graphs()
    
    with open("motif statistics/zero_statistics_bamultishapes.json", "r") as f:
        zero_statistics = json.load(f)
    with open("motif statistics/one_statistics_bamultishapes.json", "r") as f:
        one_statistics = json.load(f)
    config.correlation[1] = one_statistics 
    config.correlation[0] = zero_statistics

config.vgae_model = vgae_model

net_stability = 0
net_interpret = 0 
net_fidelity = 0        
# --- [MODIFIED] Use the end_index from arguments for total graph count ---
# We calculate the actual number of graphs processed based on the input range
num_graphs_processed = 0 
# The original variable name 'num_graphs' is not strictly necessary for the loop, but helpful for the final division.
# We'll use the variable 'num_graphs_to_process' for the division later.
# -------------------------------------------------------------------------
max_edges = 0
for k in range(len(dataset)):
    edge_index = dataset[k].edge_index
    max_edges = max(max_edges, edge_index.shape[1])
    

explanations = None 
# --- [MODIFIED] Use range(begin_index, end_index) for the loop ---
for k in tqdm(range(begin_index, end_index)):
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
        config.max_edges = args.max_edges
        config.allowed = range(len(edge_list))

        mcts_exp = MCTS(main_model, x, edge_list, edge_index, explanation_reward, metric_weights, 
                    constraint, C=10, num_simulations=20, rollout_depth=100)

        exec(open("interpret_norm.py").read(), globals())
        config.alter_graphs = []
        config.alter_graphs_pyg = []
    
        print(f'Prediction probability:{config.original_prob} and original predicted class:{config.original_pred}')
        
        present_node_state = set()
        present_edge_state = set()
        best_edge_state = set()
        best_reward = [0,0,0,-100]

        for _ in range(config.max_edges):
            result = mcts_exp.search(present_node_state)
            present_node_state = result.node_state
            present_edge_state = get_induced_edges(present_node_state)
            reward = explanation_reward(present_edge_state, metric_weights)
            print(present_node_state, reward[-1])
            if(reward[-1] >= best_reward[-1]):
                best_reward = reward
                best_edge_state = present_edge_state

        print(best_edge_state)
        target_graph_data = to_pyg_data(best_edge_state)
    
        config.alter_graphs_pyg.append(target_graph_data)
        config.alter_graphs.append((best_edge_state,best_reward[-1]))
        # print("Best subset in the first stage:", len(best_subset))

        print('Stage 1 complete.')
        # print(f'Interpret:{best_reward[0]}, Fidelity:{best_reward[1]}, Prob:{F.softmax(config.model(data=target_graph_data),dim = 1)[:,config.original_pred].item()}')

        alter_graphs_all = []
        choices = np.linspace(0.6, 0.8, 10)
        # Sample random graphs and get their explanations with the same user metrics preference
        for i in range(25):

            k = np.random.choice(choices)
            sampled_indices = generate_random_walk_seed(dataset[graph_index], seed_edge_count = int(k * x.shape[0]))
            alter_graphs_all.append(sampled_indices)

        similar_graphs = most_similar_graphs(list(range(len(edge_list))), alter_graphs_all, 10) # return indices of the most similar graphs
        print(similar_graphs)
        for similar_graph in similar_graphs:
            
            config.allowed = alter_graphs_all[similar_graph]

            present_node_state = set()
            present_edge_state = set()
            best_edge_state = set()
            best_reward = [0,0,0,-100]

            for _ in range(config.max_edges):
                result = mcts_exp.search(present_node_state)
                present_node_state = result.node_state
                present_edge_state = get_induced_edges(present_node_state)
                reward = explanation_reward(present_edge_state, metric_weights)
                if(reward[-1] >= best_reward[-1]):
                    best_reward = reward
                    best_edge_state = present_edge_state

            print(best_edge_state)
            target_graph_data = to_pyg_data(best_edge_state)
            if (len(best_edge_state) > 0):
                config.alter_graphs.append((best_edge_state, best_reward[-1]))
                # print("Best subset inside perturbed graphs", len(best_subset))
                config.alter_graphs_pyg.append(to_pyg_data(best_edge_state))


        # print(f'{len(config.alter_graphs)} smoothening graphs')
        print("Beginning Stage 2..")
        # Run MCTS with updated reward function
        exec(open("stability_norm.py").read(), globals())
        print("Stability parameters estimation complete")


        config.allowed = range(len(edge_list))
        present_node_state = set()
        present_edge_state = set()
        best_edge_state = set()
        best_reward = [0,0,0,0,-100]

        mcts_comb = MCTS(main_model, x, edge_list, edge_index, combined_reward, metric_weights, 
                    constraint, C=10, num_simulations=20, rollout_depth=100)

        for _ in range(config.max_edges):
            result = mcts_comb.search(present_node_state)
            present_node_state = result.node_state
            present_edge_state = get_induced_edges(present_node_state)
            reward = combined_reward(present_edge_state, metric_weights)
            
            if(reward[-1] >= best_reward[-1]):
                best_reward = reward
                best_edge_state = present_edge_state

        print(best_edge_state)
        best_subset_main = best_edge_state.copy()
        # print("After full graph processing", len(best_subset_main))

        print("Stage 2 complete.")
        # print(f'Stability:{best_reward[3]}, Interpret:{best_reward[1]}, Fidelity:{best_reward[2]}, Prob:{F.softmax(config.model(data=target_graph_data),dim = 1)[:,config.original_pred].item()}')

        size = len(best_subset_main)
        net_interpret += best_reward[0] * config.size_results_inter[size]['std'] + config.size_results_inter[size]['mean']
        net_fidelity += best_reward[1] * config.size_results_fid[size]['std'] + config.size_results_fid[size]['mean']

        config.alter_graphs = []
        config.alter_graphs_pyg = []
        alter_graphs_all = []

        for i in range(25):

            k = np.random.choice(choices)
            sampled_indices = generate_random_walk_seed(dataset[graph_index], seed_edge_count = k * x.shape[0])
            alter_graphs_all.append(sampled_indices)

        similar_graphs = most_similar_graphs(list(range(len(edge_list))), alter_graphs_all, 10) # return indices of the most similar graphs
        
        for similar_graph in similar_graphs:

            config.allowed = alter_graphs_all[similar_graph]

            present_node_state = set()
            present_edge_state = set()
            best_edge_state = set()
            best_reward = [0,0,0,-100]

            for _ in range(config.max_edges):
                result = mcts_exp.search(present_node_state)
                present_node_state = result.node_state
                present_edge_state = get_induced_edges(present_node_state)
                reward = explanation_reward(present_edge_state, metric_weights)
                if(reward[-1] >= best_reward[-1]):
                    best_reward = reward
                    best_edge_state = present_edge_state
            
            print(present_edge_state)
            if(len(present_edge_state) > 0):
                config.alter_graphs.append((present_edge_state, best_reward[-1]))
                config.alter_graphs_pyg.append(to_pyg_data(present_edge_state))

        net_stability += combined_reward(best_subset_main, metric_weights)[2] * config.size_results_stab[size]['std'] + config.size_results_stab[size]['mean']
        num_graphs_processed += 1
        
        # final_subset_vector = np.zeros((1,max_edges))
        # for i in best_subset_main:
        #     final_subset_vector[0,i] = 1
            
        # if explanations is None:
        #     explanations = final_subset_vector
        
        # else:
        #     explanations = np.concatenate((explanations, final_subset_vector))

    except Exception as e:
        print(e)
        continue

# --- [MODIFIED] Use the calculated number of graphs for division ---
if num_graphs_processed > 0:
    
    with open(f'results_{begin_index}-{end_index-1}_{dataset_str}_{sim_index}_{config.max_edges}.txt', 'w') as f:
        f.write(f"--- Results for Graph Indices {begin_index} to {end_index-1} ---" + '\n')
        f.write(f"Total Graphs Processed: {num_graphs_processed}" + '\n')
        f.write(f"Stability score: {net_stability/num_graphs_processed}" + '\n')
        f.write(f"Fidelity score: {net_fidelity/num_graphs_processed}" + '\n')
        f.write(f"Interpretability score: {net_interpret/num_graphs_processed}" + '\n')
    
    # if(kaggle):
    #     os.chdir("/kaggle/working")
        
    # np.save(f'{begin_index}-{end_index-1}_{dataset_str}_{sim_index}', explanations)
else:
    print("No graphs were processed (begin_index >= end_index).")
# -----------------------------------------------------------------