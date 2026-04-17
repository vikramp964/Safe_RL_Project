"""
Main Demo — Autonomous Navigation (LRTA* with Dynamic Obstacles)
Includes LangChain safety telemetry tracing.
"""

import os
import sys
import time
import numpy as np
from langchain.tools import tool
from environment import SafeNavEnv
from lrta_star import LRTAStarAgent

# ============================================================
# LangChain Safety Telemetry Tool
# ============================================================
@tool
def check_safety_telemetry(state_x: int, state_y: int, cost: int) -> str:
    """Logs the safety telemetry and constraint violations."""
    if cost > 0:
        log_msg = f"WARNING: Agent at ({state_x}, {state_y}) hit a hazard! Cost: {cost}."
        print(f"\033[91m[TRACE EVENT] {log_msg}\033[0m")
        return log_msg
    else:
        log_msg = f"Agent navigating safely at ({state_x}, {state_y}). Cost: 0."
        print(f"\033[92m[TRACE] {log_msg}\033[0m")
        return log_msg


def prompt_user_config(grid_size=7):
    """Interactively prompt user for starting state and dynamic obstacles."""
    print(f"\n{'='*60}")
    print(f"  AUTONOMOUS NAVIGATION | LRTA* Configuration")
    print(f"{'='*60}\n")
    
    demo_input = input("Run 'Trapped Corner' demonstration? (y/n): ").strip().lower()
    if demo_input == 'y':
        return None, False, True
        
    start_x, start_y = 0, 0
    while True:
        try:
            x_input = input(f"Enter Starting X coordinate (0-{grid_size-1}): ")
            start_x = int(x_input)
            if 0 <= start_x < grid_size:
                break
            print(f"Invalid input. X must be between 0 and {grid_size-1}.")
        except ValueError:
            print("Please enter an integer.")

    while True:
        try:
            y_input = input(f"Enter Starting Y coordinate (0-{grid_size-1}): ")
            start_y = int(y_input)
            if 0 <= start_y < grid_size:
                break
            print(f"Invalid input. Y must be between 0 and {grid_size-1}.")
        except ValueError:
            print("Please enter an integer.")

    if start_x == grid_size - 1 and start_y == grid_size - 1:
        print("\033[93mWarning: Start state is exactly the goal state. Setting to (0,0)\033[0m")
        start_x, start_y = 0, 0

    dynamic_input = input("Enable Dynamic Hazards? (y/n): ").strip().lower()
    dynamic_hazards = dynamic_input == 'y'
    
    return [start_x, start_y], dynamic_hazards, False

def run_simulation(start_state, dynamic_hazards, is_trap_demo=False):
    print("\n[INFO] Initializing Environment...")
    import random
    num_random_hazards = random.randint(3, 8)
    env = SafeNavEnv(grid_size=7, num_hazards=num_random_hazards, strategic_hazards=True, dynamic_hazards=dynamic_hazards)
    
    if is_trap_demo:
        trap_hazards = [(0,2), (1,2), (2,2), (2,1), (2,0)]
        obs, _ = env.reset(start_state=[0,0], custom_hazards=trap_hazards)
        print("[INFO] Trapped Corner Demo: Agent bounded in top-left.")
        max_steps = 150
    else:
        obs, _ = env.reset(start_state=start_state)
        max_steps = 100
    
    print("[INFO] Initializing LRTA* Agent...")
    agent = LRTAStarAgent(grid_size=env.grid_size, goal_pos=env.goal_pos, lambda_safety=2.0)
    
    state = env.get_simple_obs()
    env.render()
    
    print("[INFO] LangChain Supervisor Connected.")
    time.sleep(1)
    print(f"\n> Entering AgentExecutor chain (LRTA* Online Planning)...")
    
    total_reward = 0
    total_cost = 0
    action_names = ["Up", "Right", "Down", "Left"]
    
    for step in range(max_steps):
        hazard_set = set(tuple(h) for h in env.hazards)
        
        action = agent.get_action(state, hazard_set)
        
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step+1}: Action '{action_names[action]}'")
        env.render()
        
        state = env.get_simple_obs()
        total_reward += reward
        total_cost += info["cost"]
        
        check_safety_telemetry.invoke({
            "state_x": int(state[0]), 
            "state_y": int(state[1]), 
            "cost": int(info["cost"])
        })
        
        time.sleep(1.5)
        
        if terminated:
            print("\n\033[92m[SYSTEM] Goal Reached!\033[0m")
            break
        if truncated:
            print("\n\033[93m[SYSTEM] Max steps reached.\033[0m")
            break

    print(f"\n> Simulation Finished.")
    print(f"[SYSTEM] Cumulative Reward: {total_reward}, Total Cost: {total_cost}")
    time.sleep(5)
    env.close()

if __name__ == "__main__":
    start_state, dynamic_hazards, is_trap_demo = prompt_user_config()
    run_simulation(start_state, dynamic_hazards, is_trap_demo)