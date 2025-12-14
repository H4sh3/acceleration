"""Zombie Survival Environment with CNN-based egocentric observation.

The agent receives a small 2D grid centered on itself, rotated so the agent
always faces "up". This enables translation and rotation invariant learning.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from .worms_3d_env import (
    MAX_HEALTH, MAX_AMMO,
    DASH_COOLDOWN, DASH_DISTANCE,
    point_in_obstacles
)

# Zombie constants
ZOMBIE_SPEED = 0.15
ZOMBIE_HEALTH = 50
ZOMBIE_DAMAGE = 10
ZOMBIE_ATTACK_RANGE = 1.5

# Movement constants
ACCELERATION = 0.08
MAX_VELOCITY = 0.5
FRICTION = 0.92

# Projectile constants
PROJECTILE_SPEED = 0.8
PROJECTILE_DAMAGE = 25
PROJECTILE_RADIUS = 0.5
PROJECTILE_MIN_HIT_DIST = 2.0  # Minimum distance from agent for projectile to hit
MAX_PROJECTILES = 5  # Limited ammo forces agent to aim

# CNN observation constants
GRID_SIZE = 21  # 21x21 grid centered on agent
GRID_RESOLUTION = 0.5  # 0.5 world units per cell (doubled resolution)
N_CHANNELS = 5  # walls, zombies, projectiles, aim direction, enemy radar


def compute_egocentric_grid(agent_pos, agent_angle, arena_size, obstacles, 
                            zombies, projectiles, grid_size=GRID_SIZE, 
                            resolution=GRID_RESOLUTION):
    """Compute an egocentric grid observation.
    
    The grid is centered on the agent and rotated so the agent faces "up" (+Y in grid).
    
    Channels:
        0: Walls/obstacles/boundaries (1.0 = blocked)
        1: Zombies (1.0 = zombie present)
        2: Projectiles (1.0 = projectile present)
        3: Aim indicator (line in front of agent)
        4: Enemy radar (border cells lit in direction of out-of-view zombies)
    
    Returns:
        np.ndarray of shape (N_CHANNELS, grid_size, grid_size)
    """
    grid = np.zeros((N_CHANNELS, grid_size, grid_size), dtype=np.float32)
    
    center = grid_size // 2
    
    # To make agent face "up" in grid, we rotate by -agent_angle
    # (if agent faces right (0), we rotate -0 = 0, so right in world = up in grid... wait)
    # Actually: agent_angle=0 means facing right (+X in world)
    # We want that to appear as "up" in grid (+Y in grid, which is -gy direction)
    # So we rotate world coords by (pi/2 - agent_angle) to align agent's facing with grid up
    rotation = math.pi/2 - agent_angle
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)
    
    ax, ay = agent_pos[0], agent_pos[1]
    
    # For each cell in the grid
    for gy in range(grid_size):
        for gx in range(grid_size):
            # Grid offset from center (agent position)
            # In grid: +X is right, +Y is down (standard image coords)
            # We want: grid up (gy=0) to be agent's forward direction
            dx_grid = (gx - center) * resolution
            dy_grid = (center - gy) * resolution  # Flip so gy=0 is "up" (+Y in local)
            
            # Rotate from grid (agent-local) to world coordinates
            # Inverse rotation: rotate by -rotation = agent_angle - pi/2
            cos_inv = math.cos(-rotation)
            sin_inv = math.sin(-rotation)
            dx_world = dx_grid * cos_inv - dy_grid * sin_inv
            dy_world = dx_grid * sin_inv + dy_grid * cos_inv
            
            # World position
            wx = ax + dx_world
            wy = ay + dy_world
            
            # Channel 0: Walls/boundaries
            if wx < 0 or wx >= arena_size or wy < 0 or wy >= arena_size:
                grid[0, gy, gx] = 1.0
            elif point_in_obstacles(wx, wy, obstacles):
                grid[0, gy, gx] = 1.0
    
    # Channel 1: Zombies
    for zombie in zombies:
        if not zombie.get("alive", False):
            continue
        zx, zy = zombie["pos"]
        
        # Transform to agent-relative coordinates
        dx = zx - ax
        dy = zy - ay
        
        # Rotate world-to-grid (inverse of grid-to-world)
        # Grid-to-world uses cos_inv, sin_inv where cos_inv=cos(-rotation), sin_inv=sin(-rotation)
        # World-to-grid is the transpose: use cos_inv, -sin_inv for the matrix
        # Since cos_inv = cos_r and sin_inv = -sin_r:
        local_x = dx * cos_r - dy * sin_r
        local_y = dx * sin_r + dy * cos_r
        
        # Convert to grid indices
        gx = int(center + local_x / resolution)
        gy = int(center - local_y / resolution)
        
        # Mark zombie and surrounding cells (zombie has radius)
        for dgy in range(-1, 2):
            for dgx in range(-1, 2):
                ngx, ngy = gx + dgx, gy + dgy
                if 0 <= ngx < grid_size and 0 <= ngy < grid_size:
                    grid[1, ngy, ngx] = 1.0
    
    # Channel 2: Projectiles
    for proj in projectiles:
        if not proj.get("active", True):
            continue
        px, py = proj["pos"]
        
        dx = px - ax
        dy = py - ay
        
        # Same rotation as zombies
        local_x = dx * cos_r - dy * sin_r
        local_y = dx * sin_r + dy * cos_r
        
        gx = int(center + local_x / resolution)
        gy = int(center - local_y / resolution)
        
        if 0 <= gx < grid_size and 0 <= gy < grid_size:
            grid[2, gy, gx] = 1.0
    
    # Channel 3: Aim direction (line in front of agent)
    # Since agent faces up in grid, draw a line upward from center
    for i in range(1, center + 1):
        gy = center - i  # Go up from center
        if 0 <= gy < grid_size:
            grid[3, gy, center] = 1.0
    
    # Channel 4: Enemy radar - light up border cells in direction of out-of-view zombies
    # View range is grid_size * resolution / 2
    view_range = grid_size * resolution / 2
    
    for zombie in zombies:
        if not zombie.get("alive", True):
            continue
        
        zx, zy = zombie["pos"]
        dx = zx - ax
        dy = zy - ay
        
        # Transform to agent-local coordinates (same as zombie channel)
        local_x = dx * cos_r - dy * sin_r
        local_y = dx * sin_r + dy * cos_r
        
        # Check if zombie is outside the view
        if abs(local_x) <= view_range and abs(local_y) <= view_range:
            continue  # Zombie is in view, skip
        
        # Calculate which border to light up based on direction
        # Normalize to get direction
        dist = math.sqrt(local_x * local_x + local_y * local_y)
        if dist < 0.1:
            continue
        
        dir_x = local_x / dist
        dir_y = local_y / dist
        
        # Intensity based on distance (closer = brighter)
        intensity = max(0.3, 1.0 - dist / 30.0)
        
        # Light up border cells in that direction
        # Determine which edge(s) to light based on direction
        if abs(dir_x) > abs(dir_y):
            # Primarily left or right
            if dir_x > 0:
                # Right edge
                gx = grid_size - 1
                # Light multiple cells on right edge based on vertical position
                center_gy = int(center - dir_y * center * 0.8)
                for offset in range(-2, 3):
                    gy = center_gy + offset
                    if 0 <= gy < grid_size:
                        grid[4, gy, gx] = max(grid[4, gy, gx], intensity)
            else:
                # Left edge
                gx = 0
                center_gy = int(center - dir_y * center * 0.8)
                for offset in range(-2, 3):
                    gy = center_gy + offset
                    if 0 <= gy < grid_size:
                        grid[4, gy, gx] = max(grid[4, gy, gx], intensity)
        else:
            # Primarily up or down
            if dir_y > 0:
                # Top edge (forward)
                gy = 0
                center_gx = int(center + dir_x * center * 0.8)
                for offset in range(-2, 3):
                    gx = center_gx + offset
                    if 0 <= gx < grid_size:
                        grid[4, gy, gx] = max(grid[4, gy, gx], intensity)
            else:
                # Bottom edge (behind)
                gy = grid_size - 1
                center_gx = int(center + dir_x * center * 0.8)
                for offset in range(-2, 3):
                    gx = center_gx + offset
                    if 0 <= gx < grid_size:
                        grid[4, gy, gx] = max(grid[4, gy, gx], intensity)
    
    return grid


class ZombieSurvivalCNNEnv(gym.Env):
    """Zombie survival environment with CNN-based egocentric observation."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    SIZE = 30
    
    OBSTACLES = [
        [12.5, 12.5, 17.5, 17.5],
    ]
    
    # Action space: 0=noop, 1-4=move, 5-6=fine rotate, 7-8=coarse rotate, 9=shoot, 10=dash
    N_ACTIONS = 11
    
    def __init__(self, render_mode=None, grid_size=GRID_SIZE):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = grid_size
        
        self.n_agents = 1
        
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        
        # Dict observation: image (CNN) + vector (MLP)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0, 
                shape=(N_CHANNELS, grid_size, grid_size), 
                dtype=np.float32
            ),
            "vector": spaces.Box(
                low=0.0, high=1.0,
                shape=(3,),  # health, shots_remaining, aim_on_target
                dtype=np.float32
            )
        })
        
        self.zombies = []
        self.kills = 0
    
    def _random_spawn_pos(self):
        """Get a random spawn position not inside obstacles."""
        margin = 3
        for _ in range(100):
            x = self.np_random.uniform(margin, self.SIZE - margin)
            y = self.np_random.uniform(margin, self.SIZE - margin)
            if not point_in_obstacles(x, y, self.OBSTACLES):
                return np.array([x, y])
        return np.array([self.SIZE / 2, self.SIZE / 2])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        spawn_pos = self._random_spawn_pos()
        
        self.zombies = []
        self.kills = 0
        self.current_step = 0
        
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        self.last_shots = []
        self.projectiles = []
        
        # Spawn initial zombies
        for _ in range(2):
            self._spawn_zombie()
        
        # 50% chance: aim at closest zombie (curriculum learning!)
        if self.np_random.random() < 0.5 and len(self.zombies) > 0:
            closest = min(self.zombies, key=lambda z: np.linalg.norm(z["pos"] - spawn_pos))
            direction = closest["pos"] - spawn_pos
            aimed_angle = math.atan2(direction[1], direction[0])
        else:
            aimed_angle = self.np_random.uniform(-math.pi, math.pi)
        
        self.agent = {
            "id": 0, "team": 0,
            "pos": spawn_pos,
            "health": MAX_HEALTH,
            "angle": aimed_angle,  # Sometimes pre-aimed at zombie!
            "alive": True,
            "velocity": np.array([0.0, 0.0]),
            "ammo": MAX_AMMO,
            "cooldown": 0,
            "dash_cooldown": 0
        }
        
        self.agents = [self.agent]
        
        return self._get_obs(), {}
    
    def _spawn_zombie(self):
        edge = self.np_random.integers(0, 4)
        margin = 2
        
        if edge == 0:  # Top
            x = self.np_random.uniform(margin, self.SIZE - margin)
            y = self.SIZE - margin
        elif edge == 1:  # Bottom
            x = self.np_random.uniform(margin, self.SIZE - margin)
            y = margin
        elif edge == 2:  # Left
            x = margin
            y = self.np_random.uniform(margin, self.SIZE - margin)
        else:  # Right
            x = self.SIZE - margin
            y = self.np_random.uniform(margin, self.SIZE - margin)
        
        if point_in_obstacles(x, y, self.OBSTACLES):
            return
        
        zombie = {
            "id": len(self.zombies),
            "team": 1,
            "pos": np.array([x, y]),
            "health": ZOMBIE_HEALTH,
            "angle": 0.0,
            "alive": True
        }
        self.zombies.append(zombie)
    
    def _get_closest_zombie(self):
        """Get the closest alive zombie to the player."""
        closest = None
        closest_dist = float('inf')
        for z in self.zombies:
            if not z["alive"]:
                continue
            dist = np.linalg.norm(z["pos"] - self.agent["pos"])
            if dist < closest_dist:
                closest_dist = dist
                closest = z
        return closest
    
    def _check_aim_on_target(self):
        """Check if agent is aiming at a zombie."""
        agent = self.agent
        cos_a = math.cos(agent["angle"])
        sin_a = math.sin(agent["angle"])
        
        for zombie in self.zombies:
            if not zombie.get("alive", True):
                continue
            
            dx = zombie["pos"][0] - agent["pos"][0]
            dy = zombie["pos"][1] - agent["pos"][1]
            
            # Project onto aim direction
            proj = dx * cos_a + dy * sin_a
            if proj <= 0:
                continue
            
            # Perpendicular distance
            perp_x = cos_a * proj
            perp_y = sin_a * proj
            perp_dist = math.sqrt((dx - perp_x)**2 + (dy - perp_y)**2)
            
            if perp_dist < 1.0:  # Hit radius
                return 1.0
        
        return 0.0
    
    def step(self, action):
        reward = 0.0
        self.last_shots = []
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        agent = self.agent
        
        if not agent["alive"]:
            return self._get_obs(), 0.0, True, False, {"kills": self.kills}
        
        max_vel = MAX_VELOCITY
        accel = ACCELERATION
        ROTATE_FINE = 0.05
        ROTATE_COARSE = 0.3
        
        moved = False
        
        # Process action
        if action == 1:  # Up
            agent["velocity"][1] += accel
            moved = True
        elif action == 2:  # Down
            agent["velocity"][1] -= accel
            moved = True
        elif action == 3:  # Left
            agent["velocity"][0] -= accel
            moved = True
        elif action == 4:  # Right
            agent["velocity"][0] += accel
            moved = True
        elif action == 5:  # Fine rotate left
            agent["angle"] += ROTATE_FINE
        elif action == 6:  # Fine rotate right
            agent["angle"] -= ROTATE_FINE
        elif action == 7:  # Coarse rotate left
            agent["angle"] += ROTATE_COARSE
        elif action == 8:  # Coarse rotate right
            agent["angle"] -= ROTATE_COARSE
        elif action == 9:  # Shoot
            if len(self.projectiles) < MAX_PROJECTILES:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                self.projectiles.append({
                    "pos": agent["pos"].copy(),
                    "vel": np.array([dx * PROJECTILE_SPEED, dy * PROJECTILE_SPEED]),
                    "active": True
                })
        elif action == 10:  # Dash
            if agent["dash_cooldown"] <= 0:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                agent["pos"][0] += dx * DASH_DISTANCE
                agent["pos"][1] += dy * DASH_DISTANCE
                agent["dash_cooldown"] = DASH_COOLDOWN
                moved = True
        
        if moved:
            reward += 0.1  # Small reward for moving - standing still = death
        
        # Aiming reward - moderate signal to guide learning
        aim_score = self._check_aim_on_target()
        if aim_score > 0:
            reward += 0.5  # Reward for aiming (reduced from 2.0)
        
        # Clamp velocity
        vel_mag = np.linalg.norm(agent["velocity"])
        if vel_mag > max_vel:
            agent["velocity"] = agent["velocity"] / vel_mag * max_vel
        
        # Apply velocity
        agent["pos"] += agent["velocity"]
        agent["velocity"] *= FRICTION
        
        # Decrement cooldowns
        if agent["dash_cooldown"] > 0:
            agent["dash_cooldown"] -= 1
        
        # Handle collision
        if self._handle_collision(agent):
            agent["velocity"] *= 0.5
        
        # Update projectiles
        reward += self._update_projectiles()
        
        # Move zombies
        for zombie in self.zombies:
            if not zombie["alive"]:
                continue
            
            to_player = agent["pos"] - zombie["pos"]
            dist = np.linalg.norm(to_player)
            
            if dist > 0.1:
                direction = to_player / dist
                zombie["pos"] += direction * ZOMBIE_SPEED
                zombie["angle"] = math.atan2(direction[1], direction[0])
            
            if dist < ZOMBIE_ATTACK_RANGE:
                agent["health"] -= ZOMBIE_DAMAGE * 0.1
                self.was_hit = 1.0
                reward -= 1.0  # Penalty for getting hit
            
            self._handle_collision(zombie)
        
        # Maintain 2 zombies
        while sum(1 for z in self.zombies if z["alive"]) < 2:
            self._spawn_zombie()
        
        # No distance shaping - learn from outcomes only
        
        # Check death
        if agent["health"] <= 0:
            agent["alive"] = False
            reward -= 100  # Death penalty
        
        # Survival bonus
        reward += 0.1
        
        self.current_step += 1
        
        terminated = not agent["alive"]
        truncated = self.current_step >= 1000
        
        self.agents = [self.agent] + [z for z in self.zombies if z["alive"]]
        
        return self._get_obs(), reward, terminated, truncated, {"kills": self.kills}
    
    def _update_projectiles(self):
        """Update projectiles and return reward."""
        reward = 0.0
        self.last_shots = []
        
        for proj in self.projectiles:
            if not proj["active"]:
                continue
            
            proj["pos"] += proj["vel"]
            
            # Bounds check
            if (proj["pos"][0] < 0 or proj["pos"][0] > self.SIZE or
                proj["pos"][1] < 0 or proj["pos"][1] > self.SIZE):
                proj["active"] = False
                continue
            
            # Obstacle check
            if point_in_obstacles(proj["pos"][0], proj["pos"][1], self.OBSTACLES):
                proj["active"] = False
                continue
            
            # Zombie hit check
            for zombie in self.zombies:
                if not zombie["alive"]:
                    continue
                
                # Check if zombie is too close to agent (can't hit point-blank)
                zombie_agent_dist = np.linalg.norm(zombie["pos"] - self.agent["pos"])
                if zombie_agent_dist < PROJECTILE_MIN_HIT_DIST:
                    continue
                
                dist = np.linalg.norm(proj["pos"] - zombie["pos"])
                if dist < PROJECTILE_RADIUS + 0.5:
                    zombie["health"] -= PROJECTILE_DAMAGE
                    self.hit_enemy = 1.0
                    reward += PROJECTILE_DAMAGE
                    proj["active"] = False
                    
                    if zombie["health"] <= 0:
                        zombie["alive"] = False
                        self.kills += 1
                        reward += 500  # Big reward for kills
                    break
            
            if proj["active"]:
                self.last_shots.append({
                    "pos": proj["pos"].copy(),
                    "vel": proj["vel"].copy(),
                    "active": True
                })
        
        self.projectiles = [p for p in self.projectiles if p["active"]]
        return reward
    
    def _handle_collision(self, entity):
        """Handle boundary and obstacle collision."""
        collided = False
        
        old_x, old_y = entity["pos"][0], entity["pos"][1]
        entity["pos"][0] = np.clip(entity["pos"][0], 0, self.SIZE - 1)
        entity["pos"][1] = np.clip(entity["pos"][1], 0, self.SIZE - 1)
        if entity["pos"][0] != old_x or entity["pos"][1] != old_y:
            collided = True
        
        for obs in self.OBSTACLES:
            ox_min, oy_min, ox_max, oy_max = obs
            if ox_min <= entity["pos"][0] <= ox_max and oy_min <= entity["pos"][1] <= oy_max:
                cx, cy = (ox_min + ox_max) / 2, (oy_min + oy_max) / 2
                dx = entity["pos"][0] - cx
                dy = entity["pos"][1] - cy
                if abs(dx) > abs(dy):
                    entity["pos"][0] = ox_max + 0.1 if dx > 0 else ox_min - 0.1
                else:
                    entity["pos"][1] = oy_max + 0.1 if dy > 0 else oy_min - 0.1
                collided = True
        
        return collided
    
    def _get_obs(self):
        """Get Dict observation with image and vector components."""
        # Image: egocentric grid
        image = compute_egocentric_grid(
            self.agent["pos"], self.agent["angle"],
            self.SIZE, self.OBSTACLES,
            self.zombies, self.projectiles,
            grid_size=self.grid_size
        )
        
        # Vector: health, shots remaining, aim on target
        shots_remaining = (MAX_PROJECTILES - len(self.projectiles)) / MAX_PROJECTILES
        aim_on_target = self._check_aim_on_target()
        
        vector = np.array([
            self.agent["health"] / MAX_HEALTH,
            shots_remaining,
            aim_on_target
        ], dtype=np.float32)
        
        return {"image": image, "vector": vector}
    
    def render(self):
        pass
    
    def close(self):
        pass
