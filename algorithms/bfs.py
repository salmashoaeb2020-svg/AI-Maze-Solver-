from collections import deque
from maze.maze_utils import get_valid_neighbors

def bfs(grid, start, goal):
    """
    Breadth-First Search (BFS) algorithm.
    Uses a Queue data structure to explore nodes level by level.
    """
    queue = deque([[start]])
    visited = {start}
    explored_nodes = []
    
    while queue:
        path = queue.popleft()
        current = path[-1]
        explored_nodes.append(current)
        
        if current == goal:
            return {
                "success": True,
                "path": path,
                "explored_nodes": explored_nodes,
                "path_cost": len(path) - 1
            }
            
        for neighbor, cost in get_valid_neighbors(grid, current):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
                
    return {
        "success": False,
        "path": [],
        "explored_nodes": explored_nodes,
        "path_cost": 0
    }
