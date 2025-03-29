import math
import random
from collections import defaultdict
import config
from reward import reward
from constraint import constraint
from torch_geometric.data import Data

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state  # Selected edge indices
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self, all_edges, constraint_fn):
        """Check if all valid edges have been added to this state."""
        return all(edge in self.state or not constraint_fn(self.state | {edge}) for edge in all_edges)

    def best_child(self, exploration_weight=1.0):
        """Select the best child using UCT (Upper Confidence Bound for Trees)."""
        return max(
            self.children.values(),
            key=lambda node: node.value / (node.visits + 1e-6) + 
                             exploration_weight * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6))
        )

class MCTS:
    def __init__(self, x, edge_list, reward_function, constraint_function, C=1.4, num_simulations=1000, rollout_depth=5):
        self.edge_list = edge_list
        self.x = x
        self.reward_function = reward_function
        self.constraint_function = constraint_function
        self.C = C
        self.num_simulations = num_simulations
        self.rollout_depth = rollout_depth

        config.edge_list = self.edge_list
        config.x = self.x

    def select(self, node):
        """Selection step: Traverse tree using UCT until an expandable node is found."""
        while node.is_fully_expanded(range(len(self.edge_list)), self.constraint_function):
            node = node.best_child(self.C)
        return node

    def expand(self, node):
        """Expand a node by adding a new child corresponding to an unselected edge."""
        available_actions = {edge for edge in range(len(self.edge_list)) if edge not in node.state}
        valid_actions = [edge for edge in available_actions if self.constraint_function(node.state | {edge})]

        if not valid_actions:
            return node  # No more valid expansions

        new_edge = random.choice(valid_actions)
        new_state = node.state | {new_edge}
        child_node = MCTSNode(new_state, parent=node)
        node.children[new_edge] = child_node
        return child_node

    def simulate(self, node):
        """Simulate a rollout from the given state using a random policy while respecting constraints."""
        current_state = set(node.state)
        available_actions = [edge for edge in range(len(self.edge_list)) if edge not in current_state]

        for _ in range(self.rollout_depth):
            if not available_actions:
                break
            action = random.choice(available_actions)
            if self.constraint_function(current_state | {action}):
                current_state.add(action)
            available_actions.remove(action)

        return self.reward_function(current_state)

    def backpropagate(self, node, reward):
        """Backpropagate reward to update value estimates."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, initial_state=set()):
        """Run MCTS search and return the best set of edges found."""
        root = MCTSNode(initial_state)
        
        for _ in range(self.num_simulations):
            node = self.select(root)  # Selection
            node = self.expand(node)  # Expansion
            reward = self.simulate(node)  # Simulation
            self.backpropagate(node, reward)  # Backpropagation
        
        # Return the best edge subset found
        best_node = root.best_child(0)  # Set exploration weight to 0 for exploitation
        return best_node.state

x = torch.tensor([[1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0]], dtype = torch.float)

edge_list = [(0,1), (1,2), (0,2)]  # Example edge list

x_query = torch.tensor([[1,0,0,0,0,0,0],
                  [1,0,0,0,0,0,0]],
                dtype = torch.float)

edge_index_query = torch.tensor([[0],[1]], dtype = torch.long)
query = Data(x=target_x, edge_index=target_edge_list) 
query_graphs.append(query)

mcts = MCTS(x , edge_list, reward, constraint, C=1.4, num_simulations=500, rollout_depth=3)
best_subset = mcts.search()

print("Best edge indices:", best_subset)
print("Selected edges:", [edge_list[i] for i in best_subset])