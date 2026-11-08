import streamlit as st
import numpy as np
import time
import random
import heapq
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# ==================== 1. MAZE GENERATOR & UTILS ====================
class MazeGenerator:
    def __init__(self, width=15, height=15):
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.goal = (width - 1, height - 1)

    def generate_random_maze(self, wall_probability=0.25):
        grid = np.zeros((self.height, self.width), dtype=int)
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) == self.start or (r, c) == self.goal:
                    continue
                if random.random() < wall_probability:
                    grid[r][c] = 1  # Wall
        grid[self.start[0]][self.start[1]] = 0
        grid[self.goal[0]][self.goal[1]] = 0
        return grid

def manhattan_distance(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

# ==================== 2. SEARCH ALGORITHMS ====================
def get_neighbors(grid, node):
    r, c = node
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if grid[nr][nc] != 1:  # Not a wall
                neighbors.append((nr, nc))
    return neighbors

def bfs(grid, start, goal):
    queue = [[start]]
    explored = set([start])
    while queue:
        path = queue.pop(0)
        node = path[-1]
        if node == goal:
            return {"success": True, "path": path, "explored_nodes": list(explored), "path_cost": len(path) - 1}
        for neighbor in get_neighbors(grid, node):
            if neighbor not in explored:
                explored.add(neighbor)
                queue.append(path + [neighbor])
    return {"success": False, "path": [], "explored_nodes": list(explored), "path_cost": float('inf')}

def ucs(grid, start, goal):
    queue = [(0, [start])]
    explored = set()
    while queue:
        cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return {"success": True, "path": path, "explored_nodes": list(explored), "path_cost": cost}
        if node not in explored:
            explored.add(node)
            for neighbor in get_neighbors(grid, node):
                if neighbor not in explored:
                    heapq.heappush(queue, (cost + 1, path + [neighbor]))
    return {"success": False, "path": [], "explored_nodes": list(explored), "path_cost": float('inf')}

def astar(grid, start, goal):
    queue = [(manhattan_distance(start, goal), 0, [start])]
    explored = set()
    while queue:
        f, cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return {"success": True, "path": path, "explored_nodes": list(explored), "path_cost": cost}
        if node not in explored:
            explored.add(node)
            for neighbor in get_neighbors(grid, node):
                if neighbor not in explored:
                    g = cost + 1
                    h = manhattan_distance(neighbor, goal)
                    heapq.heappush(queue, (g + h, g, path + [neighbor]))
    return {"success": False, "path": [], "explored_nodes": list(explored), "path_cost": float('inf')}

def weighted_astar(grid, start, goal, weight=1.5):
    queue = [(weight * manhattan_distance(start, goal), 0, [start])]
    explored = set()
    while queue:
        f, cost, path = heapq.heappop(queue)
        node = path[-1]
        if node == goal:
            return {"success": True, "path": path, "explored_nodes": list(explored), "path_cost": cost}
        if node not in explored:
            explored.add(node)
            for neighbor in get_neighbors(grid, node):
                if neighbor not in explored:
                    g = cost + 1
                    h = manhattan_distance(neighbor, goal)
                    heapq.heappush(queue, (g + weight * h, g, path + [neighbor]))
    return {"success": False, "path": [], "explored_nodes": list(explored), "path_cost": float('inf')}

# ==================== 3. VISUALIZATION ====================
def render_maze_grid(grid, path=None, explored=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap_grid = np.array(grid, dtype=float)
    
    if explored:
        for r, c in explored:
            if grid[r][c] == 0:
                cmap_grid[r][c] = 0.5  # Explored color
    if path:
        for r, c in path:
            cmap_grid[r][c] = 0.3  # Path color
            
    # Start and Goal markers
    start_pos = (0, 0)
    goal_pos = (grid.shape[1] - 1, grid.shape[0] - 1)
    cmap_grid[start_pos[1]][start_pos[0]] = 0.2
    cmap_grid[goal_pos[1]][goal_pos[0]] = 0.8

    ax.imshow(cmap_grid, cmap="Accent")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

# ==================== 4. MACHINE LEARNING MODEL ====================
class MazeHeuristicModel:
    def __init__(self):
        self.model = DecisionTreeRegressor(random_state=42)
        self.mae = 0.0

    def train(self, num_mazes=5, width=10, height=10):
        X, y = [], []
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
                X.append([r, c, gen.goal[0], gen.goal[1], manh, width, height])
                y.append(remaining_cost)
        if len(X) > 0:
            X_arr, y_arr = np.array(X), np.array(y)
            X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42)
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)
            self.mae = mean_absolute_error(y_test, preds)
        return self.mae

# ==================== 5. STREAMLIT UI APP ====================
st.set_page_config(page_title="AI Maze Solver", page_icon="🤖", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0d1117; color: #c9d1d9; }
    .stSidebar { background-color: #161b22; }
    h1, h2, h3 { color: #58a6ff; }
    .metric-card { background-color: #21262d; border: 1px solid #30363d; padding: 15px; border-radius: 8px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

if "generator" not in st.session_state:
    st.session_state.width = 15
    st.session_state.height = 15
    st.session_state.generator = MazeGenerator(width=15, height=15)
    st.session_state.grid = st.session_state.generator.generate_random_maze(wall_probability=0.25)
    st.session_state.ml_model = None

st.sidebar.title("🤖 AI Maze Solver")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Maze Generator", "Solve Maze", "Algorithm Comparison", "Machine Learning"])

if page == "Home":
    st.title("🧠 AI Maze Solver & Pathfinding Game")
    st.markdown("### University Artificial Intelligence Project")
    st.write("Welcome! Use the sidebar navigation to generate mazes, test AI search algorithms (BFS, UCS, A*, Weighted A*), and evaluate the machine learning heuristic model.")

elif page == "Maze Generator":
    st.title("🗺️ Maze Generator & Configuration")
    w = st.slider("Width", 5, 25, st.session_state.width)
    h = st.slider("Height", 5, 25, st.session_state.height)
    wall_p = st.slider("Wall Density", 0.1, 0.4, 0.25)
    
    if st.button("Generate New Maze", type="primary"):
        st.session_state.width = w
        st.session_state.height = h
        st.session_state.generator = MazeGenerator(width=w, height=h)
        st.session_state.grid = st.session_state.generator.generate_random_maze(wall_probability=wall_p)
        st.success("New maze generated!")
        
    fig = render_maze_grid(st.session_state.grid)
    st.pyplot(fig)

elif page == "Solve Maze":
    st.title("⚡ Solve Maze with AI Algorithms")
    algo = st.selectbox("Choose Algorithm", ["BFS", "UCS", "A*", "Weighted A*"])
    weight = 1.5
    if algo == "Weighted A*":
        weight = st.slider("Weight (w)", 1.0, 5.0, 1.5)
        
    if st.button("Run Algorithm", type="primary"):
        grid = st.session_state.grid
        start = st.session_state.generator.start
        goal = st.session_state.generator.goal
        
        t0 = time.time()
        if algo == "BFS": res = bfs(grid, start, goal)
        elif algo == "UCS": res = ucs(grid, start, goal)
        elif algo == "A*": res = astar(grid, start, goal)
        else: res = weighted_astar(grid, start, goal, weight=weight)
        t_exec = (time.time() - t0) * 1000
        
        if res["success"]:
            st.success(f"Path found in {t_exec:.2f} ms!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Path Cost", res["path_cost"])
            c2.metric("Explored Nodes", len(res["explored_nodes"]))
            c3.metric("Time", f"{t_exec:.1f} ms")
            fig = render_maze_grid(grid, path=res["path"], explored=res["explored_nodes"])
            st.pyplot(fig)
        else:
            st.error("No path found!")
    else:
        fig = render_maze_grid(st.session_state.grid)
        st.pyplot(fig)

elif page == "Algorithm Comparison":
    st.title("📊 Algorithm Comparison")
    if st.button("Run Comparison", type="primary"):
        grid = st.session_state.grid
        start = st.session_state.generator.start
        goal = st.session_state.generator.goal
        algos = {
            "BFS": bfs(grid, start, goal),
            "UCS": ucs(grid, start, goal),
            "A*": astar(grid, start, goal),
            "Weighted A*": weighted_astar(grid, start, goal, weight=2.0)
        }
        import pandas as pd
        data = [{"Algorithm": k, "Success": v["success"], "Path Cost": v["path_cost"], "Explored Nodes": len(v["explored_nodes"])} for k, v in algos.items()]
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)

elif page == "Machine Learning":
    st.title("🤖 Machine Learning Heuristic Component")
    if st.button("Train ML Model Now", type="primary"):
        with st.spinner("Training Decision Tree model..."):
            ml = MazeHeuristicModel()
            mae = ml.train()
            st.session_state.ml_model = ml
            st.success(f"Trained successfully! MAE: {mae:.4f}")
    if st.session_state.ml_model:
        st.metric("Model MAE", f"{st.session_state.ml_model.mae:.4f}")
    else:
        st.warning("Click above to train the model.")
 
