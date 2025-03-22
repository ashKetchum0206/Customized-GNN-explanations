from subgraph_matching import subgraph_score
import config

def reward(selected_edges):

    cardinality = len(selected_edges)
    edge_list = config.edge_list
    return cardinality + subgraph_score(selected_edges)



