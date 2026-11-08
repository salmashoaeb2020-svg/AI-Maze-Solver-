import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import streamlit as st
import numpy as np
import time
import sys
import os

# ضبط مسار المشروع الأساسي لضمان عمل الاستيرادات على Streamlit Cloud
sys.path.append(os.path.abspath('.'))

from maze.maze_generator import MazeGenerator
from algorithms.bfs import bfs
from algorithms.ucs import ucs
from algorithms.astar import astar
from algorithms.weighted_astar import weighted_astar
from visualization.visualization import render_maze_grid
from machine_learning.ml_model import MazeHeuristicModel

# Page Configuration
st.set_page_config(
    page_title="AI Maze Solver - University Project",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark blue & purple theme)
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stSidebar {
        background-color: #161b22;
    }
    h1, h2, h3 {
        color: #58a6ff;
    }
    .metric-card {
        background-color: #21262d;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize Session State for Maze persistence across tabs
if "maze_gen" not in st.session_state:
    st.session_state.width = 15
    st.session_state.height = 15
    st.session_state.generator = MazeGenerator(width=15, height=15)
    st.session_state.grid = st.session_state.generator.generate_random_maze(wall_probability=0.25)
    st.session_state.ml_model = None

# Sidebar Navigation
st.sidebar.title("🤖 AI Maze Solver")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Maze Generator", "Solve Maze", "Algorithm Comparison", "Machine Learning"]
)

if page == "Home":
    st.title("🧠 AI Maze Solver & Pathfinding Game")
    st.markdown("### University Artificial Intelligence Project (9-Student Team)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Welcome to our interactive AI pathfinding application! This tool demonstrates how classical search agents and machine learning heuristics navigate complex environments.
        
        ### 🚀 Key Features:
        - **Classical Search**: BFS, UCS, A* Search (Manhattan), and Weighted A*.
        - **Machine Learning Integration**: Decision Tree Regressor trained to predict heuristic distances.
        - **Interactive Grid**: Custom start, goal, walls, and weighted terrain.
        - **Visual Analytics**: Compare explored nodes, path costs, and execution times instantly.
        """)
        if st.button("🚀 Generate Maze & Start", type="primary"):
            st.success("Switch to the 'Maze Generator' or 'Solve Maze' tabs using the sidebar!")
            
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Project Status</h3>
            <p style="color: #3fb950; font-weight: bold;">Fully Operational</p>
            <p>All core modules and UI components linked successfully.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Maze Generator":
    st.title("🗺️ Maze Generator & Configuration")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        width = st.slider("Width", 5, 25, st.session_state.width)
        height = st.slider("Height", 5, 25, st.session_state.height)
        wall_prob = st.slider("Wall Density", 0.1, 0.4, 0.25)
        
        if st.button("Generate New Maze", type="primary"):
            st.session_state.width = width
            st.session_state.height = height
            st.session_state.generator = MazeGenerator(width=width, height=height)
            st.session_state.grid = st.session_state.generator.generate_random_maze(wall_probability=wall_prob)
            st.success("New random maze generated!")
            
    with col2:
        st.subheader("Current Maze Grid Preview")
        fig = render_maze_grid(st.session_state.grid)
        st.pyplot(fig)

elif page == "Solve Maze":
    st.title("⚡ Solve Maze with AI Algorithms")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        algorithm = st.selectbox(
            "Choose Search Algorithm",
            ["Breadth-First Search (BFS)", "Uniform Cost Search (UCS)", "A* Search", "Weighted A*"]
        )
        weight = 1.5
        if algorithm == "Weighted A*":
            weight = st.slider("Heuristic Weight (w)", 1.0, 5.0, 1.5)
            
        run_btn = st.button("Run Algorithm", type="primary")
        
    with col2:
        grid = st.session_state.grid
        start = st.session_state.generator.start
        goal = st.session_state.generator.goal
        
        if run_btn:
            start_time = time.time()
            if algorithm == "Breadth-First Search (BFS)":
                res = bfs(grid, start, goal)
            elif algorithm == "Uniform Cost Search (UCS)":
                res = ucs(grid, start, goal)
            elif algorithm == "A* Search":
                res = astar(grid, start, goal)
            else:
                res = weighted_astar(grid, start, goal, weight=weight)
            exec_time = (time.time() - start_time) * 1000
            
            if res["success"]:
                st.success(f"Solution found in {exec_time:.2f} ms!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Path Cost", res["path_cost"])
                c2.metric("Explored Nodes", len(res["explored_nodes"]))
                c3.metric("Execution Time", f"{exec_time:.1f} ms")
                
                fig = render_maze_grid(grid, path=res["path"], explored=res["explored_nodes"])
                st.pyplot(fig)
            else:
                st.error("No path found to the goal!")
                fig = render_maze_grid(grid, explored=res["explored_nodes"])
                st.pyplot(fig)
        else:
            fig = render_maze_grid(grid)
            st.pyplot(fig)

elif page == "Algorithm Comparison":
    st.title("📊 Algorithm Performance Comparison")
    st.write("Compare all implemented search algorithms side-by-side on the current maze.")
    
    if st.button("Run Full Comparison", type="primary"):
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
        data = []
        for name, res in algos.items():
            data.append({
                "Algorithm": name,
                "Success": res["success"],
                "Path Cost": res["path_cost"],
                "Explored Nodes": len(res["explored_nodes"]),
            })
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("Algorithm")[["Explored Nodes", "Path Cost"]])

elif page == "Machine Learning":
    st.title("🤖 Machine Learning Heuristic Component")
    st.write("Train a Scikit-learn Decision Tree Regressor to predict remaining path distances.")
    
    if st.button("Train ML Model Now", type="primary"):
        with st.spinner("Training Decision Tree on generated maze datasets..."):
            model = MazeHeuristicModel()
            mae = model.train()
            st.session_state.ml_model = model
            st.success(f"Model trained successfully! Test Mean Absolute Error (MAE): {mae:.4f}")
            
    if st.session_state.ml_model is not None:
        st.info("The ML model is active and ready to guide heuristic estimations.")
        st.metric("Model MAE", f"{st.session_state.ml_model.mae:.4f}")
    else:
        st.warning("Click the button above to train and initialize the model.")
