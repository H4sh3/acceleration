"""Zombie Survival Environment.

Single agent fights waves of zombies. Zombies spawn and chase the player.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


# =============================================================================
# Constants
# =============================================================================
MAX_HEALTH = 100.0
MAX_AMMO = 5
DASH_COOLDOWN = 50
DASH_DISTANCE = 5.0
N_RAYS = 8
RAY_MAX_RANGE = 10.0

ZOMBIE_SPEED = 0.15
ZOMBIE_HEALTH = 50
ZOMBIE_DAMAGE = 10
ZOMBIE_ATTACK_RANGE = 1.5

MOVE_STEP = 0.3
PROJECTILE_SPEED = 0.8
PROJECTILE_DAMAGE = 25
PROJECTILE_RADIUS = 0.5
PROJECTILE_MIN_HIT_DIST = 0.5
PROJECTILE_MAX_RANGE = 7.0
MAX_PROJECTILES = 10

# Observation: 1 health + 2 zombies x 4 + 8 wall rays + 1 shots + 9 quadrants = 27
OBS_DIM = 27


# =============================================================================
# Geometry Helpers
# =============================================================================
def point_in_obstacles(x, y, obstacles):
    """Check if point is inside any obstacle."""
    if obstacles is None:
        return False
    for obs in obstacles:
        ox_min, oy_min, ox_max, oy_max = obs
        if ox_min <= x <= ox_max and oy_min <= y <= oy_max:
            return True
    return False


def ray_cast(x, y, angle, max_range, arena_size, obstacles):
    """Cast a ray and return distance to nearest hit."""
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    if obstacles is None:
        obstacles = []
    
    step_size = 0.2
    dist = 0.0
    
    while dist < max_range:
        px = x + dx * dist
        py = y + dy * dist
        
        if px < 0 or px >= arena_size or py < 0 or py >= arena_size:
            return dist
        
        for obs in obstacles:
            ox_min, oy_min, ox_max, oy_max = obs
            if ox_min <= px <= ox_max and oy_min <= py <= oy_max:
                return dist
        
        dist += step_size
    
    return max_range


# =============================================================================
# Observation
# =============================================================================
def compute_observation(agent_pos, agent_angle, agent_health,
                        zombie_positions, obstacles, arena_size,
                        num_active_projectiles):
    """Compute 27-dim observation.
    
    Indices:
        0: Health (normalized 0-1)
        1-4: Zombie 1 (closest): cos_delta, sin_delta, dist, in_range
        5-8: Zombie 2 (next closest): cos_delta, sin_delta, dist, in_range
        9-16: Wall ray sensors (8 rays)
        17: Shots remaining (normalized 0-1)
        18-26: Quadrant position (3x3 grid, one-hot)
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    
    x_self, y_self = agent_pos[0], agent_pos[1]
    theta_self = agent_angle
    max_dist = arena_size * math.sqrt(2)
    
    # Health
    obs[0] = np.clip(agent_health / MAX_HEALTH, 0.0, 1.0)
    
    # 2 closest zombies
    for i, zombie_pos in enumerate(zombie_positions[:2]):
        base_idx = 1 + i * 4
        if zombie_pos is not None:
            dx = zombie_pos[0] - x_self
            dy = zombie_pos[1] - y_self
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist > 0.001:
                angle_to = math.atan2(dy, dx)
                delta = angle_to - theta_self
                obs[base_idx] = math.cos(delta)
                obs[base_idx + 1] = math.sin(delta)
            else:
                obs[base_idx] = 1.0
                obs[base_idx + 1] = 0.0
            obs[base_idx + 2] = np.clip(dist / max_dist, 0.0, 1.0)
            obs[base_idx + 3] = 1.0 if dist <= PROJECTILE_MAX_RANGE else 0.0
        else:
            obs[base_idx] = 0.0
            obs[base_idx + 1] = 0.0
            obs[base_idx + 2] = 1.0
            obs[base_idx + 3] = 0.0
    
    # Wall rays
    for i in range(N_RAYS):
        t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
        local_angle = -math.pi / 2 + t * math.pi
        global_angle = theta_self + local_angle
        dist = ray_cast(x_self, y_self, global_angle, RAY_MAX_RANGE, arena_size, obstacles)
        obs[9 + i] = dist / RAY_MAX_RANGE
    
    # Shots remaining
    obs[17] = (MAX_PROJECTILES - num_active_projectiles) / MAX_PROJECTILES
    
    # Quadrant (3x3 grid one-hot)
    qx = int(np.clip(x_self / arena_size * 3, 0, 2))
    qy = int(np.clip(y_self / arena_size * 3, 0, 2))
    obs[18 + qy * 3 + qx] = 1.0
    
    return obs


# =============================================================================
# Environment
# =============================================================================
class ZombieSurvivalEnv(gym.Env):
    """Zombie survival environment - fight waves of zombies."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    SIZE = 30
    OBSTACLES = [[12.5, 12.5, 17.5, 17.5]]
    N_ACTIONS = 10  # 0=noop, 1-4=move, 5-6=fine rotate, 7-8=coarse rotate, 9=shoot
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.n_agents = 1
        
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM * 2,), dtype=np.float32
        )
        
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
        
        self.agent = {
            "id": 0, "team": 0,
            "pos": self._random_spawn_pos(),
            "health": MAX_HEALTH,
            "angle": self.np_random.uniform(-math.pi, math.pi),
            "alive": True,
            "ammo": MAX_AMMO,
            "cooldown": 0,
            "dash_cooldown": 0
        }
        
        self.agents = [self.agent]  # For renderer compatibility
        self.zombies = []
        self.kills = 0
        self.current_step = 0
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        self.last_shots = []
        self.projectiles = []
        self.spawn_edge_index = 0
        
        for _ in range(4):
            self._spawn_zombie()
        
        return self._get_obs(), {}
    
    def _spawn_zombie(self):
        edge = self.spawn_edge_index
        self.spawn_edge_index = (self.spawn_edge_index + 1) % 4
        margin = 2
        
        if edge == 0:    # Top
            x, y = self.SIZE // 2, self.SIZE - margin
        elif edge == 1:  # Left
            x, y = margin, self.SIZE // 2
        elif edge == 2:  # Right
            x, y = self.SIZE - margin, self.SIZE // 2
        else:            # Bottom
            x, y = self.SIZE // 2, margin
        
        if point_in_obstacles(x, y, self.OBSTACLES):
            return
        
        self.zombies.append({
            "id": len(self.zombies),
            "team": 1,
            "pos": np.array([float(x), float(y)]),
            "health": ZOMBIE_HEALTH,
            "angle": 0.0,
            "alive": True
        })
    
    def _get_zombies_sorted_by_distance(self):
        alive = [z for z in self.zombies if z["alive"]]
        alive.sort(key=lambda z: np.linalg.norm(z["pos"] - self.agent["pos"]))
        return alive
    
    def step(self, action):
        reward = 0.0
        self.last_shots = []
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        agent = self.agent
        if not agent["alive"]:
            return self._get_obs(), 0.0, True, False, {"kills": self.kills}
        
        ROTATE_FINE = 0.05
        ROTATE_COARSE = 0.3
        
        # Process action
        if action == 1:
            agent["pos"][1] += MOVE_STEP
        elif action == 2:
            agent["pos"][1] -= MOVE_STEP
        elif action == 3:
            agent["pos"][0] -= MOVE_STEP
        elif action == 4:
            agent["pos"][0] += MOVE_STEP
        elif action == 5:
            agent["angle"] += ROTATE_FINE
        elif action == 6:
            agent["angle"] -= ROTATE_FINE
        elif action == 7:
            agent["angle"] += ROTATE_COARSE
        elif action == 8:
            agent["angle"] -= ROTATE_COARSE
        elif action == 9:
            if len(self.projectiles) < MAX_PROJECTILES:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                self.projectiles.append({
                    "pos": agent["pos"].copy(),
                    "vel": np.array([dx * PROJECTILE_SPEED, dy * PROJECTILE_SPEED]),
                    "active": True,
                    "distance_traveled": 0.0
                })
        
        if agent["dash_cooldown"] > 0:
            agent["dash_cooldown"] -= 1
        
        self._handle_collision(agent)
        reward += self._update_projectiles()
        
        # Move zombies
        for zombie in self.zombies:
            if not zombie["alive"]:
                continue
            
            to_player = agent["pos"] - zombie["pos"]
            dist = np.linalg.norm(to_player)
            
            if dist > 0.1:
                direction = to_player / dist
                zombie["pos"] = zombie["pos"] + direction * ZOMBIE_SPEED
                zombie["angle"] = math.atan2(direction[1], direction[0])
            
            if dist < ZOMBIE_ATTACK_RANGE:
                agent["health"] -= ZOMBIE_DAMAGE * 0.1
                self.was_hit = 1.0
                reward -= 1.0
            
            self._handle_collision(zombie)
        
        # Maintain 4 zombies
        while sum(1 for z in self.zombies if z["alive"]) < 4:
            self._spawn_zombie()
        
        # Hunting rewards
        alive_zombies = [z for z in self.zombies if z["alive"]]
        if alive_zombies:
            closest = min(alive_zombies, key=lambda z: np.linalg.norm(z["pos"] - agent["pos"]))
            dist = np.linalg.norm(closest["pos"] - agent["pos"])
            
            if dist < 3.0:
                reward -= 0.05
            elif dist <= 8.0:
                reward += 0.1
            else:
                reward -= 0.02 * (dist - 8.0)
        
        if agent["health"] <= 0:
            agent["alive"] = False
            reward -= 100
        
        self.current_step += 1
        terminated = not agent["alive"]
        truncated = self.current_step >= 1000
        
        self.agents = [self.agent] + [z for z in self.zombies if z["alive"]]
        
        return self._get_obs(), reward, terminated, truncated, {"kills": self.kills}
    
    def _update_projectiles(self):
        reward = 0.0
        self.last_shots = []
        
        for proj in self.projectiles:
            if not proj["active"]:
                continue
            
            proj["pos"] += proj["vel"]
            proj["distance_traveled"] += PROJECTILE_SPEED
            
            if proj["distance_traveled"] >= PROJECTILE_MAX_RANGE:
                proj["active"] = False
                continue
            
            if (proj["pos"][0] < 0 or proj["pos"][0] > self.SIZE or
                proj["pos"][1] < 0 or proj["pos"][1] > self.SIZE):
                proj["active"] = False
                continue
            
            if point_in_obstacles(proj["pos"][0], proj["pos"][1], self.OBSTACLES):
                proj["active"] = False
                continue
            
            for zombie in self.zombies:
                if not zombie["alive"]:
                    continue
                
                if np.linalg.norm(zombie["pos"] - self.agent["pos"]) < PROJECTILE_MIN_HIT_DIST:
                    continue
                
                if np.linalg.norm(proj["pos"] - zombie["pos"]) < PROJECTILE_RADIUS + 0.5:
                    zombie["health"] -= PROJECTILE_DAMAGE
                    self.hit_enemy = 1.0
                    reward += PROJECTILE_DAMAGE
                    proj["active"] = False
                    
                    if zombie["health"] <= 0:
                        zombie["alive"] = False
                        self.kills += 1
                        reward += 100
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
        old_x, old_y = entity["pos"][0], entity["pos"][1]
        entity["pos"][0] = np.clip(entity["pos"][0], 0, self.SIZE - 1)
        entity["pos"][1] = np.clip(entity["pos"][1], 0, self.SIZE - 1)
        
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
    
    def _get_obs(self):
        sorted_zombies = self._get_zombies_sorted_by_distance()
        zombie_positions = [z["pos"] for z in sorted_zombies[:2]]
        while len(zombie_positions) < 2:
            zombie_positions.append(None)
        
        obs = compute_observation(
            self.agent["pos"], self.agent["angle"], self.agent["health"],
            zombie_positions, self.OBSTACLES, self.SIZE,
            len(self.projectiles)
        )
        return np.concatenate([obs, obs])
    
    def render(self):
        pass
    
    def close(self):
        pass
