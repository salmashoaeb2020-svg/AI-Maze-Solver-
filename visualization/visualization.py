import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def render_maze_grid(grid, path=None, explored=None):
    """
    Renders the maze grid using Matplotlib with distinct colors for:
    - White: Empty (0)
    - Black: Wall (1)
    - Green: Start (2)
    - Red: Goal (3)
    - Blue: Explored nodes
    - Yellow: Final path
    - Orange: Weighted cells (4)
    """
    height, width = grid.shape
    img = np.ones((height, width, 3), dtype=float)
    
    COLOR_EMPTY = [1.0, 1.0, 1.0]       # White
    COLOR_WALL = [0.1, 0.1, 0.1]        # Black/Dark Gray
    COLOR_START = [0.1, 0.8, 0.2]       # Green
    COLOR_GOAL = [0.9, 0.2, 0.2]        # Red
    COLOR_WEIGHTED = [1.0, 0.6, 0.2]    # Orange
    COLOR_EXPLORED = [0.4, 0.7, 1.0]    # Light Blue
    COLOR_PATH = [1.0, 0.9, 0.1]        # Yellow
    
    for r in range(height):
        for c in range(width):
            val = grid[r, c]
            if val == 1:
                img[r, c] = COLOR_WALL
            elif val == 2:
                img[r, c] = COLOR_START
            elif val == 3:
                img[r, c] = COLOR_GOAL
            elif val == 4:
                img[r, c] = COLOR_WEIGHTED
            else:
                img[r, c] = COLOR_EMPTY
                
    if explored:
        for r, c in explored:
            if grid[r, c] not in [2, 3]:
                img[r, c] = COLOR_EXPLORED
                
    if path:
        for r, c in path:
            if grid[r, c] not in [2, 3]:
                img[r, c] = COLOR_PATH
                
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    
    return fig
