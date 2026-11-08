import heapq
from maze.maze_utils import get_valid_neighbors

def ucs(grid, start, goal):
    """
    Uniform Cost Search (UCS) algorithm.
    Uses a Priority Queue to expand the node with the lowest cumulative cost g(n).
    """
    pq = [(0, start, [start])]
    visited_costs = {start: 0}
    explored_nodes = []
    
    while pq:
        current_cost, current, path = heapq.heappop(pq)
        
        if current in explored_nodes:
            continue
        explored_nodes.append(current)
        
        if current == goal:
            return {
                "success": True,
                "path": path,
                "explored_nodes": explored_nodes,
                "path_cost": current_cost
            }
            
        for neighbor, move_cost in get_valid_neighbors(grid, current):
            new_cost = current_cost + move_cost
            
            if neighbor not in visited_costs or new_cost < visited_costs[neighbor]:
                visited_costs[neighbor] = new_cost
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (new_cost, neighbor, new_path))
                
    return {
        "success": False,
        "path": [],
        "explored_nodes": explored_nodes,
        "path_cost": 0
    }
