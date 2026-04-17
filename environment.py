import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math

class SafeNavEnv(gym.Env):
    """A Dynamic Grid Environment for Safe Reinforcement Learning (CMDP)"""
    
    def __init__(self, grid_size=7, num_hazards=5, max_steps=100, strategic_hazards=True, dynamic_hazards=False):
        super(SafeNavEnv, self).__init__()
        
        self.grid_size = grid_size
        self.num_hazards = num_hazards
        self.max_steps = max_steps
        self.strategic_hazards = strategic_hazards
        self.dynamic_hazards = dynamic_hazards
        
        self.grid_window_size = 600
        self.sidebar_width = 300
        self.window_width = self.grid_window_size + self.sidebar_width
        self.window_height = self.grid_window_size
        self.cell_size = self.grid_window_size // self.grid_size
        
        self.action_space = spaces.Discrete(4)
        
        obs_size = 2 + (grid_size * grid_size)
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(obs_size,), dtype=np.float32)
        
        self.start_pos = np.array([0, 0])
        self.agent_pos = self.start_pos.copy()
        self.goal_pos = np.array([self.grid_size-1, self.grid_size-1]) 
        
        self.hazards = []
        self.window = None
        self.clock = None
        self.font = None
        self.title_font = None
        self.current_step = 0
        self.total_cost = 0

    def _randomize_hazards(self):
        """Place hazards — strategically along the diagonal or randomly"""
        self.hazards = []
        
        if self.strategic_hazards:
            diagonal_cells = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if abs(i - j) <= 2 and not np.array_equal(np.array([i,j]), self.agent_pos) and not np.array_equal(np.array([i,j]), self.goal_pos):
                        diagonal_cells.append(np.array([i, j]))
            
            random.shuffle(diagonal_cells)
            self.hazards = diagonal_cells[:min(self.num_hazards, len(diagonal_cells))]
        else:
            while len(self.hazards) < self.num_hazards:
                hx = random.randint(0, self.grid_size - 1)
                hy = random.randint(0, self.grid_size - 1)
                new_hazard = np.array([hx, hy])
                
                if not np.array_equal(new_hazard, self.agent_pos) and not np.array_equal(new_hazard, self.goal_pos):
                    if not any(np.array_equal(new_hazard, existing) for existing in self.hazards):
                        self.hazards.append(new_hazard)

    def _move_hazardous_obstacles(self):
        """Pick one hazard and drift it 1 cell in a valid random direction"""
        if not self.hazards:
            return
            
        idx = random.randint(0, len(self.hazards) - 1)
        hazard = self.hazards[idx]
        
        dirs = [(0,1), (0,-1), (1,0), (-1,0)]
        random.shuffle(dirs)
        
        for dx, dy in dirs:
            nx, ny = hazard[0] + dx, hazard[1] + dy
            new_pos = np.array([nx, ny])
            
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                if not np.array_equal(new_pos, self.agent_pos) and \
                   not np.array_equal(new_pos, self.goal_pos) and \
                   not np.array_equal(new_pos, self.start_pos) and \
                   not any(np.array_equal(new_pos, existing) for existing in self.hazards):
                    self.hazards[idx] = new_pos
                    break

    def get_grid(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for hazard in self.hazards:
            grid[hazard[0], hazard[1]] = 1
        grid[self.goal_pos[0], self.goal_pos[1]] = 2
        return grid

    def _get_obs(self):
        grid = self.get_grid()
        obs = np.concatenate([self.agent_pos.astype(np.float32), grid.flatten().astype(np.float32)])
        return obs

    def get_simple_obs(self):
        return self.agent_pos.copy()

    def reset(self, seed=None, start_state=None, custom_hazards=None):
        super().reset(seed=seed)
        if start_state is not None:
            self.start_pos = np.array(start_state)
        self.agent_pos = self.start_pos.copy()
        
        self.current_step = 0
        self.total_cost = 0
        self.steps_until_hazard_move = random.choice([1, 2, 3])  # Added random hazard timer
        
        if custom_hazards is not None:
            self.hazards = [np.array(h) for h in custom_hazards]
            self.num_hazards = len(custom_hazards)
        else:
            self._randomize_hazards()
            
        return self._get_obs(), {}

    def step(self, action):
        if action == 0 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 1 and self.agent_pos[0] < self.grid_size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] < self.grid_size - 1:
            self.agent_pos[1] += 1
        elif action == 3 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1

        self.current_step += 1
        
        if self.dynamic_hazards:
            self.steps_until_hazard_move -= 1
            if self.steps_until_hazard_move <= 0:
                self._move_hazardous_obstacles()
                self.steps_until_hazard_move = random.choice([1, 2, 3])

        reward = -1
        cost = 0
        terminated = False
        truncated = self.current_step >= self.max_steps

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = 100
            terminated = True
            
        for hazard in self.hazards:
            if np.array_equal(self.agent_pos, hazard):
                cost = 10 
                reward = -10  
                break 
                
        self.total_cost += cost
            
        info = {"cost": cost}
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption(f"CMDP Safe Navigation | LRTA*")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            
            import platform
            if platform.system() == 'Windows':
                import ctypes
                hwnd = pygame.display.get_wm_info()["window"]
                ctypes.windll.user32.SetForegroundWindow(hwnd)
                
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Segoe UI", 24)
            self.title_font = pygame.font.SysFont("Segoe UI", 32, bold=True)
            self.small_font = pygame.font.SysFont("Segoe UI", 18)
            
        # Colors - Modern Dashboard Aesthetics
        BG_COLOR = (240, 242, 245)
        GRID_COLOR = (255, 255, 255)
        LINE_COLOR = (220, 222, 225)
        SIDEBAR_BG = (30, 41, 59)     # Slate 800
        TEXT_COLOR = (248, 250, 252)  # Slate 50
        ACCENT_COLOR = (56, 189, 248) # Light Blue Accent
        
        self.window.fill(BG_COLOR)
        
        # 1. Draw Grid Area
        grid_rect = pygame.Rect(0, 0, self.grid_window_size, self.grid_window_size)
        pygame.draw.rect(self.window, GRID_COLOR, grid_rect)
        
        for x in range(0, self.grid_window_size + 1, self.cell_size):
            pygame.draw.line(self.window, LINE_COLOR, (x, 0), (x, self.grid_window_size))
            pygame.draw.line(self.window, LINE_COLOR, (0, x), (self.grid_window_size, x))
            
        # 2. Draw Entities
        def get_center(pos):
            return pos[0] * self.cell_size + self.cell_size//2, pos[1] * self.cell_size + self.cell_size//2
            
        # Goal (Golden Trophy/Star)
        gx, gy = get_center(self.goal_pos)
        pygame.draw.circle(self.window, (250, 204, 21), (gx, gy), self.cell_size//2 - 10)
        pygame.draw.circle(self.window, (253, 224, 71), (gx, gy), self.cell_size//2 - 18)
        
        # Start State Highlight
        sx, sy = get_center(self.start_pos)
        pygame.draw.rect(self.window, (226, 232, 240), (self.start_pos[0]*self.cell_size, self.start_pos[1]*self.cell_size, self.cell_size, self.cell_size), 4)
        
        # Hazards (Red Spikes/Fire)
        for hazard in self.hazards:
            hx, hy = get_center(hazard)
            points = []
            num_spikes = 8
            outer_r = self.cell_size // 2 - 8
            inner_r = outer_r - 10
            for i in range(num_spikes * 2):
                angle = i * (math.pi / num_spikes)
                r = outer_r if i % 2 == 0 else inner_r
                px = hx + int(math.cos(angle) * r)
                py = hy + int(math.sin(angle) * r)
                points.append((px, py))
            pygame.draw.polygon(self.window, (239, 68, 68), points) # Red-500
            pygame.draw.circle(self.window, (248, 113, 113), (hx, hy), inner_r - 2) # Red-400
        
        # Agent (Robot Head)
        ax, ay = get_center(self.agent_pos)
        # Body
        rob_rect = pygame.Rect(0, 0, self.cell_size - 24, self.cell_size - 24)
        rob_rect.center = (ax, ay)
        pygame.draw.rect(self.window, (59, 130, 246), rob_rect, border_radius=8) # Blue-500
        # Visor
        visor_rect = pygame.Rect(0, 0, self.cell_size - 36, 12)
        visor_rect.center = (ax, ay - 4)
        pygame.draw.rect(self.window, (15, 23, 42), visor_rect, border_radius=4)
        # Eye (glowing dot)
        pygame.draw.circle(self.window, (56, 189, 248), (ax, ay - 4), 3)
            
        # 3. Draw Sidebar Dashboard
        sidebar_rect = pygame.Rect(self.grid_window_size, 0, self.sidebar_width, self.grid_window_size)
        pygame.draw.rect(self.window, SIDEBAR_BG, sidebar_rect)
        pygame.draw.line(self.window, ACCENT_COLOR, (self.grid_window_size, 0), (self.grid_window_size, self.grid_window_size), 4)

        # Dashboard Content
        content_x = self.grid_window_size + 20
        y_offset = 30
        
        title = self.title_font.render("CMDP Dashboard", True, TEXT_COLOR)
        self.window.blit(title, (content_x, y_offset))
        y_offset += 60
        
        mode_text = "DYNAMIC" if self.dynamic_hazards else "STATIC"
        mode_color = (74, 222, 128) if self.dynamic_hazards else (148, 163, 184)
        mode_lbl = self.font.render(f"Hazards: ", True, TEXT_COLOR)
        mode_val = self.font.render(mode_text, True, mode_color)
        self.window.blit(mode_lbl, (content_x, y_offset))
        self.window.blit(mode_val, (content_x + mode_lbl.get_width(), y_offset))
        y_offset += 50
        
        step_lbl = self.font.render(f"Step Count: ", True, TEXT_COLOR)
        step_val = self.font.render(str(self.current_step), True, ACCENT_COLOR)
        self.window.blit(step_lbl, (content_x, y_offset))
        self.window.blit(step_val, (content_x + step_lbl.get_width(), y_offset))
        y_offset += 50
        
        cost_lbl = self.font.render(f"Accum. Cost: ", True, TEXT_COLOR)
        cost_color = (248, 113, 113) if self.total_cost > 0 else (74, 222, 128)
        cost_val = self.font.render(str(self.total_cost), True, cost_color)
        self.window.blit(cost_lbl, (content_x, y_offset))
        self.window.blit(cost_val, (content_x + cost_lbl.get_width(), y_offset))
        y_offset += 70
        
        # Legend
        self.window.blit(self.font.render("Legend:", True, (148, 163, 184)), (content_x, y_offset))
        y_offset += 30
        pygame.draw.rect(self.window, (59, 130, 246), (content_x, y_offset, 20, 20), border_radius=4)
        self.window.blit(self.small_font.render("Agent (LRTA*)", True, TEXT_COLOR), (content_x + 30, y_offset))
        y_offset += 35
        pygame.draw.circle(self.window, (239, 68, 68), (content_x + 10, y_offset + 10), 10)
        self.window.blit(self.small_font.render("Lethal Hazard", True, TEXT_COLOR), (content_x + 30, y_offset))
        y_offset += 35
        pygame.draw.circle(self.window, (250, 204, 21), (content_x + 10, y_offset + 10), 10)
        self.window.blit(self.small_font.render("Goal State", True, TEXT_COLOR), (content_x + 30, y_offset))
        
        pygame.display.flip()
        self.clock.tick(10) # 10 FPS for smoother viewing
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None