import numpy as np

def manhattan_distance(pos1, pos2):
    """Calculates the Manhattan distance between two grid coordinates."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_valid_neighbors(grid, current_pos):
    """
    Returns valid neighboring coordinates (up, down, left, right)
    and their movement costs.
    Wall cells (1) are impassable.
    Normal cells (0, Start 2, Goal 3) cost 1.
    Weighted cells (4) cost 5.
    """
    height, width = grid.shape
    r, c = current_pos
    neighbors = []
    
    # 4-directional movement
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            cell_value = grid[nr, nc]
            if cell_value != 1:  # Not a wall
                # Determine movement cost
                cost = 5 if cell_value == 4 else 1
                neighbors.append(((nr, nc), cost))
                
    return neighbors
