import heapq
from maze.maze_utils import manhattan_distance, get_valid_neighbors

def astar(grid, start, goal):
    """
    A* Search algorithm using Manhattan Distance heuristic.
    f(n) = g(n) + h(n)
    """
    h_start = manhattan_distance(start, goal)
    pq = [(h_start, 0, start, [start])]
    g_scores = {start: 0}
    explored_nodes = []
    
    while pq:
        f, g, current, path = heapq.heappop(pq)
        
        if current in explored_nodes:
            continue
        explored_nodes.append(current)
        
        if current == goal:
            return {
                "success": True,
                "path": path,
                "explored_nodes": explored_nodes,
                "path_cost": g
            }
            
        for neighbor, move_cost in get_valid_neighbors(grid, current):
            tentative_g = g + move_cost
            
            if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                g_scores[neighbor] = tentative_g
                h = manhattan_distance(neighbor, goal)
                f_new = tentative_g + h
                new_path = list(path)
                new_path.append(neighbor)
                heapq.heappush(pq, (f_new, tentative_g, neighbor, new_path))
                
    return {
        "success": False,
        "path": [],
        "explored_nodes": explored_nodes,
        "path_cost": 0
    }
