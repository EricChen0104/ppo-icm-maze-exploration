import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

_EMPTY = 0
_OBSTACLE = 1
_AGENT = 2
_VICTIM = 3
_FOUND_VICTIM = 4
_OUT_OF_BOUNDS = 5 

COLORS = {
    _EMPTY: (255, 255, 255),  # White
    _OBSTACLE: (50, 50, 50),     # Dark Grey 
    _AGENT: (0, 0, 255),       # Blue
    _VICTIM: (255, 0, 0),      # Red
    _FOUND_VICTIM: (0, 255, 0), # Green
}
FOG_COLOR = (128, 128, 128) # Grey

class RescueGridEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, map_size=(25, 25), num_victims=5, obstacle_ratio=0.3, view_radius=3, render_mode=None):
        super().__init__()
        
        self.width, self.height = map_size
        self.num_victims = num_victims
        self.obstacle_ratio = obstacle_ratio
        self.view_radius = view_radius
        self.view_size = 2 * self.view_radius + 1
        
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([-1, 0]), 1: np.array([0, 1]), # up, right 
            2: np.array([1, 0]),  3: np.array([0, -1]), # down, left
        }
        
        self.observation_space = spaces.Box(
            low=0, high=max(COLORS.keys()), 
            shape=(self.view_size, self.view_size), 
            dtype=np.uint8
        )
        
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._visited_map = None

        self.max_steps = 2 * self.width * self.height
        self._current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._current_step = 0
        
        self._grid = np.full((self.width, self.height), _EMPTY, dtype=np.uint8)
        num_obstacles = int(self.width * self.height * self.obstacle_ratio)
        possible_coords = [(r, c) for r in range(self.width) for c in range(self.height)]
        obstacle_indices = self.np_random.choice(len(possible_coords), num_obstacles, replace=False)
        for idx in obstacle_indices:
            r, c = possible_coords[idx]
            self._grid[r, c] = _OBSTACLE
        
        self._free_spaces = np.argwhere(self._grid == _EMPTY)
        if len(self._free_spaces) < self.num_victims + 1:
            raise ValueError("地圖空間不足，請降低障礙物比例或增加地圖尺寸。")
        chosen_indices = self.np_random.choice(len(self._free_spaces), self.num_victims + 1, replace=False)
        
        self._agent_location = self._free_spaces[chosen_indices[0]]
        self._victim_locations = [loc for loc in self._free_spaces[chosen_indices[1:]]]
        self._found_victims = [False] * self.num_victims
        self._update_true_grid()

        self._explored_map = np.full_like(self._grid, False, dtype=bool)
        self._update_explored_map()

        self._visited_map = np.full_like(self._grid, False, dtype=bool)
        self._visited_map[self._agent_location[0], self._agent_location[1]] = True
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), self._get_info()

    def step(self, action):

        self._current_step += 1

        old_location = self._agent_location.copy()
    
        direction = self._action_to_direction[action]
        new_location_attempt = self._agent_location + direction
        
        row, col = new_location_attempt
        collided = False
        
        if (0 <= row < self.width and 
            0 <= col < self.height and 
            self._grid[row, col] != _OBSTACLE):
            self._agent_location = new_location_attempt
        else:
            collided = True
            self._agent_location = old_location 
            

        is_new_visit = not self._visited_map[self._agent_location[0], self._agent_location[1]]
        found_new_victim = False
        for i, vic_loc in enumerate(self._victim_locations):
            if np.array_equal(self._agent_location, vic_loc) and not self._found_victims[i]:
                found_new_victim = True
                self._found_victims[i] = True
                break
                
        newly_explored_count = self._calculate_newly_explored_cells()
        self._visited_map[self._agent_location[0], self._agent_location[1]] = True
            
        self._update_explored_map()
        self._update_true_grid()

        reward = self._calculate_reward(collided, found_new_victim, is_new_visit, newly_explored_count)
        
        terminated = all(self._found_victims)
        truncated = self._current_step >= self.max_steps

        if self.render_mode == 'human' and truncated and not terminated:
            print(f"Time is up! Reached {self.max_steps} steps limit")

        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def _calculate_reward(self, collided, found_new_victim, is_new_visit, newly_explored_count):
        
        if found_new_victim:
            return 10.0 
        if collided:
            return -0.5 

        return -0.02
    
    def _calculate_newly_explored_cells(self):
        old_explored_map = self._explored_map.copy()
        
        r, c = self._agent_location
        top = max(0, r - self.view_radius)
        bottom = min(self.height, r + self.view_radius + 1)
        left = max(0, c - self.view_radius)
        right = min(self.width, c + self.view_radius + 1)
        
        current_view_mask = np.full_like(self._explored_map, False, dtype=bool)
        current_view_mask[top:bottom, left:right] = True
        
        newly_explored_mask = current_view_mask & ~old_explored_map
        
        # 返回新探索格子的數量
        return np.sum(newly_explored_mask)
    
    def _update_true_grid(self):
        grid_copy = self._grid.copy()
        grid_copy[grid_copy == _AGENT] = _EMPTY
        grid_copy[grid_copy == _FOUND_VICTIM] = _EMPTY
        
        for i, vic_loc in enumerate(self._victim_locations):
            if self._found_victims[i]:
                grid_copy[vic_loc[0], vic_loc[1]] = _FOUND_VICTIM
            else:
                grid_copy[vic_loc[0], vic_loc[1]] = _VICTIM
                
        grid_copy[self._agent_location[0], self._agent_location[1]] = _AGENT
        self._full_state_grid = grid_copy

    def _update_explored_map(self):
        r, c = self._agent_location
        center = (self.view_radius, self.view_radius)
        
        for i in range(self.view_size):
            for j in range(self.view_size):
                if np.hypot(i-center[0], j-center[1]) > self.view_radius:
                    continue
                for x, y in self.bresenham_line(center[0], center[1], i, j):
                    gx = r - self.view_radius + x
                    gy = c - self.view_radius + y
                    if not (0 <= gx < self.height and 0 <= gy < self.width):
                        break
                    self._explored_map[gx, gy] = True
                    if self._grid[gx, gy] == _OBSTACLE:
                        break

    def bresenham_line(self, x0, y0, x1, y1):
        points = []
        dx, dy = abs(x1-x0), abs(y1-y0)
        x, y = x0, y0
        sx = 1 if x1>x0 else -1
        sy = 1 if y1>y0 else -1
        if dx > dy:
            err = dx/2
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy/2
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x1, y1))
        return points

    def _get_obs(self):
        obs = np.full((self.view_size, self.view_size), _OUT_OF_BOUNDS, dtype=np.uint8)
        r, c = self._agent_location

        top, bottom = r-self.view_radius, r+self.view_radius+1
        left, right  = c-self.view_radius, c+self.view_radius+1
        g_top, g_bottom = max(0, top), min(self.height, bottom)
        g_left, g_right = max(0, left),  min(self.width, right)
        o_top, o_left  = g_top-top, g_left-left

        full = np.full_like(obs, _OBSTACLE, dtype=np.uint8)
        full[o_top:o_top+(g_bottom-g_top), o_left:o_left+(g_right-g_left)] = \
            self._full_state_grid[g_top:g_bottom, g_left:g_right]

        center = (self.view_radius, self.view_radius)
        for i in range(self.view_size):
            for j in range(self.view_size):
                if np.hypot(i-center[0], j-center[1]) > self.view_radius:
                    continue
                for x, y in self.bresenham_line(center[0], center[1], i, j):
                    gx = r - self.view_radius + x
                    gy = c - self.view_radius + y
                    if not (0 <= gx < self.height and 0 <= gy < self.width):
                        break

                    if self._grid[gx, gy] == _OBSTACLE:
                        obs[x, y] = _OBSTACLE
                        break

                    obs[x, y] = full[x, y]
        return obs


    def _get_info(self):
        return {"agent_location": self._agent_location, "victims_found": sum(self._found_victims)}

    def render(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.cell_size = min(800 // self.width, 800 // self.height)
            self.window = pygame.display.set_mode((self.height * self.cell_size, self.width * self.cell_size))
            pygame.display.set_caption("Rescue Grid World (Exploration View)")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.height * self.cell_size, self.width * self.cell_size))
        canvas.fill(FOG_COLOR) 

        for r in range(self.width):
            for c in range(self.height):
                if self._explored_map[r, c]: 
                    cell_type = self._full_state_grid[r, c]
                    pygame.draw.rect(
                        canvas, COLORS[cell_type],
                        pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size),
                    )
        
        for r in range(self.width):
            for c in range(self.height):
                 pygame.draw.rect(
                    canvas, (200, 200, 200),
                    pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size), 1
                )
        
        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
