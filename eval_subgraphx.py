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
from utils import create_mutag_query_graphs_pyg, create_ba2motif_query_graphs_pyg, generate_random_walk_seed
from utils import create_proteins_query_graphs_pyg, create_bamultishapes_query_graphs_pyg
from utils import get_induced_edges
from smooth_subgraph_matching import NeuroMatchEncoder
from training_proteins import GIN as GIN_proteins
# --- [NEW] Import argparse for command-line arguments ---
import argparse
import os
import json
from dgl.nn.pytorch.explain import SubgraphX
import dgl
import numpy as np
# -----------------------------------------------------

class PyGModelAdapter(torch.nn.Module):
    """
    Wraps a PyG model so it looks like a DGL model to the SubgraphXExplainer.
    Accepts (graph, feat), converts graph to edge_index, and calls the PyG model.
    """
    def __init__(self, pyg_model):
        super().__init__()
        self.pyg_model = pyg_model

    def forward(self, graph, feat, **kwargs):
        # Convert DGL graph structure to PyG edge_index
        src, dst = graph.edges()
        # Ensure correct device and type
        edge_index = torch.stack([src, dst], dim=0).long().to(feat.device)
        
        # Call the original PyG model
        # Note: We assume your model forward signature is (x, edge_index)
        return self.pyg_model(feat, edge_index)

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
    parser.add_argument('--max_nodes', type=int)
    parser.add_argument('--eval_stab', type=int)
    parser.add_argument('--indices_path', type=str, default=None,
                        help='Path to a .npy file of explicit graph indices to evaluate. '
                             'When given, overrides --begin/--end.')
    return parser.parse_args()

args = parse_args()
begin_index = args.begin
end_index = args.end
sim_index = args.sim_index
dataset_str = args.dataset
config.dataset_str = dataset_str
kaggle = args.kaggle
interp_index = args.interp_index
max_nodes = args.max_nodes
eval_stab = args.eval_stab

if args.indices_path is not None:
    graph_indices = np.load(args.indices_path).tolist()
else:
    graph_indices = list(range(begin_index, end_index))
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
# max_edges = 0
# for k in range(len(dataset)):
#     edge_index = dataset[k].edge_index
#     max_edges = max(max_edges, edge_index.shape[1])
    
 # Initialize and run MCTS
config.max_edges = int(max_nodes * 0.67)
dgl_model_wrapper = PyGModelAdapter(main_model)
dgl_explainer = SubgraphX(
    dgl_model_wrapper, 
    num_hops = 3,
    node_min = config.max_edges
)

def get_subgraphx_prediction(x, edge_index, num_nodes):

    src, dst = edge_index
    g_dgl = dgl.graph((src, dst), num_nodes=num_nodes).to(device)
    g_dgl.ndata['x'] = x

    explanation_graph = dgl_explainer.explain_graph(
        g_dgl, 
        g_dgl.ndata['x'], 
        target_class=config.original_pred
    )
    
    exp_indices = set(list(np.array(explanation_graph)))
  
    exp_indices_main_edge = set()

    for k in range(edge_index.shape[1]):
        if edge_index[0][k].item() in exp_indices or edge_index[1][k].item() in exp_indices: exp_indices_main_edge.add(k)
    
    best_edge_state = list(exp_indices_main_edge)
    print(len(best_edge_state))
    return best_edge_state
    
    
explanations = None
# --- [MODIFIED] Iterate over graph_indices (range or explicit --indices_path list) ---
for k in tqdm(graph_indices):
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

        config.edge_list = edge_list
        config.node_features = x
        config.edge_index = edge_index
        config.model = main_model
        config.original_pred = main_model(x, edge_index).argmax(dim=1).item()
        config.original_prob = main_model(x, edge_index)[0, config.original_pred].item()

        exec(open("interpret_norm.py").read(), globals())

        best_edge_state_main = get_subgraphx_prediction(x, edge_index, x.shape[0])
        print(best_edge_state_main)
        reward = explanation_reward(best_edge_state_main, metric_weights)
        net_interpret += reward[0] * config.size_results_inter[size]['std'] + config.size_results_inter[size]['mean']
        net_fidelity += reward[1] * config.size_results_fid[size]['std'] + config.size_results_fid[size]['mean']
        
        if not eval_stab:
            num_graphs_processed += 1
            continue 
            
        config.alter_graphs = []
        config.alter_graphs_pyg = []
        alter_graphs_all = []
        choices = np.linspace(0.6, 0.8, 10)

        for i in range(25):

            k = np.random.choice(choices)
            sampled_indices = generate_random_walk_seed(dataset[graph_index], seed_edge_count = k * x.shape[0])
            alter_graphs_all.append(sampled_indices)

        similar_graphs = most_similar_graphs(list(range(len(edge_list))), alter_graphs_all, 10) # return indices of the most similar graphs
        proc_done = 0
            
        for similar_graph in similar_graphs:
            try:
                edge_indices = alter_graphs_all[similar_graph]
                stab_graph_data = to_pyg_data(edge_indices)
                best_edge_state = get_subgraphx_prediction(stab_graph_data.x, stab_graph_data.edge_index, stab_graph_data.x.shape[0])
                print(best_edge_state)
                reward = explanation_reward(best_edge_state, metric_weights)
                if (len(best_edge_state) > 0):
                    config.alter_graphs.append((best_edge_state, reward[-1]))
                    # print("Best subset inside perturbed graphs", len(best_subset))
                    config.alter_graphs_pyg.append(to_pyg_data(best_edge_state))
                proc_done += 1
            except Exception as e:
                print(e)
                continue 

        exec(open("stability_norm.py").read(), globals())
        print("proc_done ", proc_done)
        net_stability += combined_reward(best_edge_state_main, metric_weights)[2] * config.size_results_stab[size]['std'] + config.size_results_stab[size]['mean']
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
if args.indices_path is not None:
    range_desc = f"indices file {args.indices_path} ({len(graph_indices)} graphs)"
else:
    range_desc = f"Graph Indices {begin_index} to {end_index-1}"

if num_graphs_processed > 0:
    print(f"--- Results for {range_desc} ---")
    print(f"Total Graphs Processed: {num_graphs_processed}")
    print(f"Stability score: {net_stability/num_graphs_processed}")
    print(f"Fidelity score: {net_fidelity/num_graphs_processed}")
    print(f"Interpretability score: {net_interpret/num_graphs_processed}")

    # if(kaggle):
    #     os.chdir("/kaggle/working")

    # np.save(f'{begin_index}-{end_index-1}_{dataset_str}_{sim_index}', explanations)
else:
    print(f"No graphs were processed ({range_desc}).")
# -----------------------------------------------------------------