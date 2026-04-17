# Autonomous Navigation in a Dynamic Environment (Safe RL / CMDP)

This project demonstrates an intelligent agent that safely navigates a dynamic **7x7 grid environment** using the **Learning Real-Time A* (LRTA*)** algorithm. 

The environment is modeled as a Constrained Markov Decision Process (CMDP) where the agent must maximize its reward (reaching the goal) while ensuring its safety cost (avoiding dynamic obstacles) remains within a budget. Because the environment features moving obstacles, the agent relies on the online planning capabilities of LRTA* to recalculate safe routes in real-time.

## Features
- **Learning Real-Time A* (LRTA*)**: Handles online path recalculation to avoid moving hazards seamlessly.
- **Dynamic Hazards**: Obstacles randomize and drift systematically across the grid during an episode, testing the agent's adaptability.
- **Customizable Starting States**: Launch the agent from any valid `(X, Y)` coordinate on the grid.
- **Trapped Corner Demo**: A pre-configured scenario demonstrating heuristic inflation, where the agent correctly learns to briefly violate safety constraints to escape an impossible geometric entrapment.
- **Pygame Dashboard**: Fully visualizes the simulated grid, real-time metrics (rewards, costs, step counts), and hazard movements.

## Installation & Requirements

Ensure you have Python 3 installed. You will need the following libraries:
```bash
pip install numpy gymnasium pygame
```

## How to Run

Navigate to the project directory in your terminal and execute the main entry file:
```bash
python main.py
```

### Simulation Prompts
Once the script runs, you will be prompted via the terminal to configure the simulation:
1. **Trapped Corner Demo**: Enter `y` to run the demonstration where the agent must break through a wall. Enter `n` to configure a standard run.
2. **Start State**: Enter the `X` and `Y` coordinate (0-6) of where you want the agent to spawn. *(Example: `0` and `0` for the top-left corner)*.
3. **Dynamic Hazards**: Enter `y` to enable dynamic movement where hazards drift to adjacent cells every few steps. 

### Visual Controls
The Pygame graphical interface will launch automatically showing the agent's navigation. Terminal logs will also stream "telemetry" tracing decisions in real-time. Please note that the agent's movement is slightly slowed down deliberately to allow for easy observation of decision-making.
