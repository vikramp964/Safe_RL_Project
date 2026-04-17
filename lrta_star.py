"""
Learning Real-Time A* (LRTA*) for Safe Navigation
Based on: MAS Book (Shoham & Leyton-Brown) Section 2.1.2, Figure 2.3

Algorithm: LRTA*
- Agent starts at node s, moves toward goal
- At each step:
    1. Compute f(j) = w(i,j) + h(j) for all neighbors j
    2. Move to i' = argmin_j f(j) (breaking ties randomly)
    3. Update: h(i) = max(h(i), f(i'))
- h-values persist across trials and converge to optimal distances

Key Properties (from textbook):
- h-values never decrease and remain admissible
- LRTA* terminates (each trial reaches the goal)
- With repeated trials, discovers the shortest path
- Same path found twice in a row = optimal path

CMDP Extension: Edge weight includes safety penalty
  w(i, j) = step_cost + lambda * hazard_cost(j)
"""

import numpy as np
import random

class LRTAStarAgent:
    """Learning Real-Time A* agent for grid navigation with safety constraints"""
    
    def __init__(self, grid_size, goal_pos, lambda_safety=2.0):
        self.grid_size = grid_size
        self.goal_pos = (int(goal_pos[0]), int(goal_pos[1]))
        self.lambda_safety = lambda_safety
        
        self.h = np.zeros((grid_size, grid_size))
        for x in range(grid_size):
            for y in range(grid_size):
                self.h[x, y] = abs(x - self.goal_pos[0]) + abs(y - self.goal_pos[1])
        
        # Action definitions: 0=Up, 1=Right, 2=Down, 3=Left
        self.action_deltas = {
            0: (0, -1),
            1: (1, 0),
            2: (0, 1),
            3: (-1, 0),
        }
        
        self.last_path = None
        self.converged = False
    
    def _get_neighbors(self, x, y):
        """Get valid neighboring cells and the action to reach them"""
        neighbors = []
        for action, (dx, dy) in self.action_deltas.items():
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                neighbors.append((nx, ny, action))
        return neighbors
    
    def _edge_weight(self, target_x, target_y, hazard_set):
        """
        Compute edge weight w(i, j) with CMDP safety penalty
        w(i, j) = 1 (step cost) + lambda * hazard_cost(j)
        """
        step_cost = 1.0
        hazard_cost = 10.0 if (target_x, target_y) in hazard_set else 0.0
        return step_cost + self.lambda_safety * hazard_cost
    
    def get_action(self, state, hazard_set):
        """
        One step of LRTA* — from MAS Book Figure 2.3:
        
            foreach neighbor j do
                f(j) <- w(i,j) + h(j)
            i' <- argmin_j f(j)         // breaking ties at random
            h(i) <- max(h(i), f(i'))    // key LRTA* update: never decrease
            move to i'
        """
        x, y = int(state[0]), int(state[1])
        
        if (x, y) == self.goal_pos:
            return 0
        
        neighbors = self._get_neighbors(x, y)
        if not neighbors:
            return random.randint(0, 3)
        
        f_values = []
        for nx, ny, action in neighbors:
            f_val = self._edge_weight(nx, ny, hazard_set) + self.h[nx, ny]
            f_values.append((f_val, action, nx, ny))
        
        min_f = min(fv[0] for fv in f_values)
        best_options = [(fv[1], fv[2], fv[3]) for fv in f_values if abs(fv[0] - min_f) < 1e-10]
        chosen = random.choice(best_options)
        best_action, best_nx, best_ny = chosen
        
        f_best = self._edge_weight(best_nx, best_ny, hazard_set) + self.h[best_nx, best_ny]
        self.h[x, y] = max(self.h[x, y], f_best)
        
        return best_action
    
    def check_convergence(self, current_path):
        """
        From textbook: If LRTA* finds the same path on two sequential trials,
        this is the shortest path.
        """
        if self.last_path is not None and current_path == self.last_path:
            self.converged = True
        self.last_path = current_path
        return self.converged


def train_lrta_star(env, episodes=500, lambda_safety=2.0):
    """
    Train the LRTA* agent over multiple trials.
    h-values persist across trials and converge to optimal.
    """
    agent = LRTAStarAgent(
        grid_size=env.grid_size,
        goal_pos=env.goal_pos,
        lambda_safety=lambda_safety
    )
    
    history_rewards = []
    history_costs = []
    history_path_lengths = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        state = env.get_simple_obs()
        hazard_set = set(tuple(h) for h in env.hazards)
        
        episode_reward = 0
        episode_cost = 0
        steps = 0
        path = [tuple(state)]
        
        for step in range(env.max_steps):
            action = agent.get_action(state, hazard_set)
            obs, reward, terminated, truncated, info = env.step(action)
            state = env.get_simple_obs()
            
            path.append(tuple(state))
            episode_reward += reward
            episode_cost += info["cost"]
            steps += 1
            
            if terminated or truncated:
                break
        
        agent.check_convergence(path)
        
        history_rewards.append(episode_reward)
        history_costs.append(episode_cost)
        history_path_lengths.append(steps)
        
        if (episode + 1) % 100 == 0:
            avg_r = np.mean(history_rewards[-100:])
            avg_c = np.mean(history_costs[-100:])
            conv = "YES" if agent.converged else "no"
            print(f"LRTA* Episode {episode+1:04d} | Avg Reward: {avg_r:6.1f} | Avg Cost: {avg_c:4.1f} | Converged: {conv}")
    
    return agent, history_rewards, history_costs, history_path_lengths
