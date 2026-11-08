import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import streamlit as st
import os
from maze.maze_generator import MazeGenerator

# Page Configuration
st.set_page_config(
    page_title="AI Maze Solver",
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

# Sidebar Navigation
st.sidebar.title("🤖 AI Maze Solver")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Maze Generator", "Solve Maze", "Algorithm Comparison", "Machine Learning", "Optimization", "Performance Dashboard"]
)

if page == "Home":
    st.title("🧠 AI Maze Solver: Intelligent Pathfinding Game")
    st.markdown("### Developed by University AI Team (9 Students)")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("""
        Welcome to the **AI Maze Solver**, a modern educational application designed to demonstrate, analyze, and compare classical and advanced Artificial Intelligence search algorithms.
        
        ### 🚀 What You Can Do:
        - **Generate & Customize**: Create random mazes, set start/goal nodes, and paint obstacles or weighted terrain.
        - **Explore Algorithms**: Watch **BFS**, **UCS**, **A* Search**, and **Weighted A*** explore step by step.
        - **Machine Learning Integration**: Observe how a **Decision Tree Regressor** guides search heuristics.
        - **Optimization**: Use **Hill Climbing** to optimize heuristic weights automatically.
        """)
        if st.button("Start Exploring -> Go to Maze Generator", type="primary"):
            st.info("Use the sidebar navigation to switch to 'Maze Generator'.")
            
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Project Status</h3>
            <p style="color: #3fb950; font-weight: bold;">Stage 1: Architecture & Generator Ready</p>
            <p>Modular Python backend configured successfully.</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Maze Generator":
    st.title("🗺️ Maze Generator & Configuration")
    st.write("Configure your maze dimensions and generate a random layout.")
    
    width = st.slider("Maze Width", 5, 30, 15)
    height = st.slider("Maze Height", 5, 30, 15)
    wall_prob = st.slider("Wall Density", 0.0, 0.5, 0.25)
    
    if st.button("Generate New Maze"):
        generator = MazeGenerator(width, height)
        grid = generator.generate_random_maze(wall_probability=wall_prob)
        st.success("Maze generated successfully!")
        st.write(grid)

else:
    st.title(f"📍 {page}")
    st.info(f"The module for **{page}** will be integrated in subsequent stages.")
