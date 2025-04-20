from collections import defaultdict
import config 

def constraint(selected_edges,log=False):

    for edge in selected_edges:
        if edge not in config.allowed: return False
    edge_list = config.edge_list
    if not selected_edges:
        return False  # No edges selected means no connectivity
    
    # Step 1: Construct adjacency list from the selected edges
    graph = defaultdict(set)
    nodes = set()  # To track unique nodes in the selected edges
    
    for idx in selected_edges:
        u, v = edge_list[idx]  # Get actual edge
        graph[u].add(v)
        graph[v].add(u)
        nodes.update([u, v])
    
    # Step 2: Perform BFS/DFS to check if all nodes are reachable
    visited = set()
    start_node = next(iter(nodes))  # Start from any node in the selected set
    
    def dfs(node):
        """Recursive DFS to mark all connected nodes"""
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)
    
    dfs(start_node)  # Start traversal
    # Step 3: Check if all nodes in the selected edges are visited
    if log:
        print(visited, nodes)
    return visited == nodes and len(selected_edges) <= config.max_edges
