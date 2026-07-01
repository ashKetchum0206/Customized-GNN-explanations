# import math
# import random
# from collections import defaultdict
# import config
# import numpy as np

# class MCTSNode:
#     def __init__(self, state, parent=None):
#         self.state = state  # Selected edge indices
#         self.parent = parent
#         self.children = {}
#         self.visits = 0
#         self.value = 0.0

#     def is_fully_expanded(self, all_edges, constraint_fn):
#         """Check if all valid edges have been added to this state."""
#         return all(edge in self.state or not constraint_fn(self.state | {edge}) for edge in all_edges)

#     def best_child(self, exploration_weight=1):
#         """Select the best child using UCT (Upper Confidence Bound for Trees)."""
#         return max(
#             self.children.values(),
#             key=lambda node: node.value / (node.visits + 1e-6) + 
#                              exploration_weight * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6))
#         )

# class MCTS:
#     def __init__(self, main_model, x, edge_list, edge_index, reward_function, metric_weights, constraint_function, stable = False, C=1.4, num_simulations=1000, rollout_depth=5):

#         self.main_model = main_model
#         self.edge_list = edge_list
#         self.edge_index = edge_index
#         self.x = x
#         self.reward_function = reward_function
#         self.constraint_function = constraint_function
#         self.C = C
#         self.num_simulations = num_simulations
#         self.rollout_depth = rollout_depth
#         self.metric_weights = metric_weights
#         self.stable = stable
#         self.tolerance = 10 
#         self.edges = edge_index.t().tolist()

#         if not stable:
#             self.best = [set(), (0,0,0,-np.inf)]
#         else:
#             self.best = [set(), -np.inf]

#         config.edge_list = self.edge_list
#         config.node_features = self.x
#         config.edge_index = self.edge_index
#         config.model = self.main_model
#         config.original_pred = self.main_model(self.x, self.edge_index).argmax(dim = 1).item()
#         config.original_prob = self.main_model(self.x, self.edge_index)[0,config.original_pred].item()

#     def select(self, node):
#         """Selection step: Traverse tree using UCT until an expandable node is found."""
#         while node.is_fully_expanded(range(len(self.edge_list)), self.constraint_function):
#             node = node.best_child(self.C)
#         return node

#     def expand(self, node):
#         """Expand a node by adding a new child corresponding to an unselected edge."""
#         available_actions = {edge for edge in range(len(self.edge_list)) if edge not in node.state}
#         valid_actions = [edge for edge in available_actions if self.constraint_function(node.state | {edge})]

#         if not valid_actions:
#             return node  # No more valid expansions

#         new_edge = random.choice(valid_actions)
#         new_state = node.state | {new_edge}
#         child_node = MCTSNode(new_state, parent=node)
#         node.children[new_edge] = child_node
#         return child_node

#     def simulate(self, node):
#         """Simulate a rollout from the given state using a random policy while respecting constraints."""
#         current_state = set(node.state)
#         available_actions = [edge for edge in range(len(self.edge_list)) if edge not in current_state]  
#         best_reward = 0
#         same_count = 1
#         prev_action = -1
  
       
#         for _ in range(self.rollout_depth):
            
#             if not available_actions or len(current_state) == config.max_edges:
#                 break
            
           
#             action = random.choice(available_actions)
#             if self.constraint_function(current_state | {action}):

#                 current_state.add(action)
#                 # best_reward = max(best_reward, self.reward_function(current_state, self.metric_weights)[-1])

#                 # if not self.stable: 
#                 #     reward_tuple = self.reward_function(current_state, self.metric_weights)
#                 #     current_reward = reward_tuple[3]
#                 #     if(current_reward > self.best[1][3]):
#                 #         self.best[1] = reward_tuple
#                 #         self.best[0] = current_state

#                 # else:
#                 #     reward = self.reward_function(current_state)
#                 #     if(reward > self.best[1]):
#                 #         self.best[1] = reward
#                 #         self.best[0] = current_state

#                 available_actions.remove(action) # since an action can be useful later on, don't remove it unless used. 
            
#             if(action == prev_action):
#                 same_count += 1
#             else:
#                 same_count = 1
                
#             prev_action = action
#             if(same_count == self.tolerance): 
#                 available_actions.remove(action) # Need to remove this action to avoid the network from getting stuck 
                
#         # print("Inside MCTS", len(current_state))
#         best_reward = max(best_reward, self.reward_function(current_state, self.metric_weights)[-1])
#         return best_reward
    


#     def backpropagate(self, node, reward):
#         """Backpropagate reward to update value estimates."""
#         while node:
#             node.visits += 1
#             node.value += reward
#             node = node.parent

#     def search(self, initial_state=set()):
#         """Run MCTS search and return the best set of edges found."""
#         root = MCTSNode(initial_state)
        
#         for _ in range(self.num_simulations):
#             node = self.select(root)  # Selection
#             node = self.expand(node)  # Expansion
#             reward = self.simulate(node)  # Simulation
#             self.backpropagate(node, reward)  # Backpropagation
        
#         # Return the best edge subset found
#         best_node = root.best_child(self.C)
#         return best_node

import math
import random
import numpy as np
import config

class MCTSNode:
    def __init__(self, node_state, parent=None):
        self.node_state = node_state  # Set of node indices (Pure Node State Space)
        self.parent = parent
        self.children = {}            # Actions are now node indices
        self.visits = 0
        self.value = 0.0

    def best_child(self, exploration_weight=1):
        """Select the best child using UCT."""
        return max(
            self.children.values(),
            key=lambda node: node.value / (node.visits + 1e-6) + 
                             exploration_weight * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6))
        )

class MCTS:
    def __init__(self, main_model, x, edge_list, edge_index, reward_function, metric_weights, constraint_function, stable=False, C=1.4, num_simulations=1000, rollout_depth=5):
        self.main_model = main_model
        self.edge_list = edge_list
        self.edge_index = edge_index
        self.x = x
        self.reward_function = reward_function
        self.constraint_function = constraint_function
        self.C = C
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth
        self.metric_weights = metric_weights
        self.stable = stable
        self.edges = edge_index.t().tolist()
        self.num_nodes = x.shape[0]

        # Fast Adjacency Lookup to rapidly build induced subgraphs
        self.node_adj = {i: [] for i in range(self.num_nodes)}
        for e_idx, (u, v) in enumerate(self.edges):
            self.node_adj[u].append((v, e_idx))
            self.node_adj[v].append((u, e_idx))

        if not stable:
            self.best = [set(), (0,0,0,-np.inf)]
        else:
            self.best = [set(), -np.inf]

        config.edge_list = self.edge_list
        config.node_features = self.x
        config.edge_index = self.edge_index
        config.model = self.main_model
        config.original_pred = self.main_model(self.x, self.edge_index).argmax(dim=1).item()
        config.original_prob = self.main_model(self.x, self.edge_index)[0, config.original_pred].item()

    def get_induced_edges(self, node_set):
        """
        Converts a purely node-based state into its corresponding edge-based state.
        Only returns edges where BOTH endpoints are in the selected node_set.
        """
        induced_edges = set()
        for u in node_set:
            for v, e_idx in self.node_adj[u]:
                if v in node_set:
                    induced_edges.add(e_idx)
        return induced_edges

    def get_valid_next_nodes(self, current_nodes):
        """
        Finds nodes that can be added to the current subgraph while maintaining connectivity
        and respecting the external constraint function.
        """
        if not current_nodes:
            # If the tree is entirely empty, any node is valid as a starting point.
            # We filter it through the constraint function just in case.
            return [n for n in range(self.num_nodes)]

        candidate_nodes = set()
        for n in current_nodes:
            for neighbor, _ in self.node_adj[n]:
                if neighbor not in current_nodes:
                    candidate_nodes.add(neighbor)

        valid_nodes = []
        for n in candidate_nodes:
            proposed_nodes = current_nodes | {n}
            proposed_edges = self.get_induced_edges(proposed_nodes)
            
            # Constraint is evaluated on the resulting backward-compatible edge state
            if self.constraint_function(proposed_edges):
                valid_nodes.append(n)
                
        return valid_nodes

    def select(self, node):
        """Selection step: Traverse tree using UCT."""
        while True:
            valid_actions = self.get_valid_next_nodes(node.node_state)
            unexpanded_actions = [n for n in valid_actions if n not in node.children]
            
            if unexpanded_actions or not node.children:
                break
            
            node = node.best_child(self.C)
        return node

    def expand(self, node):
        """Expand purely by adding a valid adjacent node."""
        valid_actions = self.get_valid_next_nodes(node.node_state)
        unexpanded_actions = [n for n in valid_actions if n not in node.children]

        if not unexpanded_actions:
            return node

        new_node_action = random.choice(unexpanded_actions)
        
        # Pure node transition
        new_node_state = node.node_state | {new_node_action}
        # Backward compatible edge conversion
        new_edge_state = self.get_induced_edges(new_node_state)
        child_node = MCTSNode(node_state = new_node_state, parent=node)
        node.children[new_node_action] = child_node
        return child_node

    def simulate(self, node):
        """Simulate a rollout by adding random adjacent nodes."""
        current_nodes = set(node.node_state)
        
        for _ in range(self.rollout_depth):
            valid_actions = self.get_valid_next_nodes(current_nodes)
            if not valid_actions:
                break
            
            action_node = random.choice(valid_actions)
            current_nodes.add(action_node)
                
        # Evaluate reward using the backward-compatible edge state of the rollout
        final_edge_state = self.get_induced_edges(current_nodes)
        return self.reward_function(final_edge_state, self.metric_weights)[-1]

    def backpropagate(self, node, reward):
        """Backpropagate reward up the pure node tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, initial_state=None):
        """
        Entry point. Takes node-based input and runs MCTS
        """
        if initial_state is None:
            initial_state = set()
            
        # --- Backward Compatibility Conversion (Edges -> Nodes) ---
        # initial_nodes = set()
        # for edge_idx in initial_state:
        #     u, v = self.edges[edge_idx]
        #     initial_nodes.add(u)
        #     initial_nodes.add(v)
            
        root = MCTSNode(node_state=initial_state)
        
        for _ in range(self.num_simulations):
            node = self.select(root)
            node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        
        if not root.children:
            return root
        
        # The best child retains the expected .state (edge indices) property
        return root.best_child(self.C)  