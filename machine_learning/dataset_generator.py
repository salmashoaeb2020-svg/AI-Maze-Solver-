import numpy as np
from maze.maze_generator import MazeGenerator
from algorithms.bfs import bfs
from maze.maze_utils import manhattan_distance

def generate_ml_dataset(num_mazes=5, width=10, height=10):
    """
    Generates training data for the Decision Tree Regressor.
    Extracts features like position, goal, manhattan distance, and dimensions,
    using BFS ground-truth shortest path cost as the target label.
    """
    X = []
    y = []
    
    for _ in range(num_mazes):
        gen = MazeGenerator(width=width, height=height)
        grid = gen.generate_random_maze(wall_probability=0.2)
        
        result = bfs(grid, gen.start, gen.goal)
        if not result["success"]:
            continue
            
        path = result["path"]
        path_length = len(path)
        for idx, pos in enumerate(path):
            remaining_cost = path_length - 1 - idx
            r, c = pos
            manh = manhattan_distance(pos, gen.goal)
            
            features = [r, c, gen.goal[0], gen.goal[1], manh, width, height]
            X.append(features)
            y.append(remaining_cost)
            
    return np.array(X), np.array(y)
