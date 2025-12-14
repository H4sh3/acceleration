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

# Observation: 1 health + 2 zombies x 4 + 1 shots + 4 move toggles = 14
OBS_DIM = 14

# Spawn distance for zombies (relative to agent) - just outside aiming range
ZOMBIE_SPAWN_DIST_MIN = 8.0   # Just outside PROJECTILE_MAX_RANGE (7.0)
ZOMBIE_SPAWN_DIST_MAX = 12.0


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
                        zombie_positions, num_active_projectiles, move_toggles=None):
    """Compute 14-dim observation for infinite world.
    
    Indices:
        0: Health (normalized 0-1)
        1-4: Zombie 1 (closest): cos_delta, sin_delta, dist_normalized, in_range
        5-8: Zombie 2 (next closest): cos_delta, sin_delta, dist_normalized, in_range
        9: Shots remaining (normalized 0-1)
        10-13: Movement toggles (forward, backward, strafe left, strafe right)
    """
    if move_toggles is None:
        move_toggles = [False, False, False, False]
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    
    x_self, y_self = agent_pos[0], agent_pos[1]
    theta_self = agent_angle
    max_dist = ZOMBIE_SPAWN_DIST_MAX  # Normalize by max spawn distance
    
    # Health
    obs[0] = np.clip(agent_health / MAX_HEALTH, 0.0, 1.0)
    
    # 2 closest zombies (relative to agent)
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
    
    # Shots remaining
    obs[9] = (MAX_PROJECTILES - num_active_projectiles) / MAX_PROJECTILES
    
    # Movement toggles
    for i, active in enumerate(move_toggles):
        obs[10 + i] = 1.0 if active else 0.0
    
    return obs


# =============================================================================
# Environment
# =============================================================================
class ZombieSurvivalEnv(gym.Env):
    """Zombie survival environment - infinite generative world."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    # Actions (discrete):
    # 0: noop
    # 1-4: set move direction (forward, backward, strafe left, strafe right)
    # 5-6: fine rotate (±1°)
    # 7-8: less fine rotate (±5°) 
    # 9-10: coarse rotate (±45°)
    # 11: shoot
    N_ACTIONS = 12
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.n_agents = 1
        
        # Discrete action space
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM * 2,), dtype=np.float32
        )
        
        self.zombies = []
        self.kills = 0
        # Movement toggles: [up, down, left, right]
        self.move_toggles = [False, False, False, False]
    
    def _random_spawn_pos(self):
        """Get a random spawn position - agent starts at origin."""
        return np.array([0.0, 0.0])
    
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
        self.move_toggles = [False, False, False, False]
        
        for _ in range(2):
            self._spawn_zombie()
        
        return self._get_obs(), {}
    
    def _spawn_zombie(self):
        """Spawn a zombie relative to agent position.
        
        Spawns at a random angle, at distance between ZOMBIE_SPAWN_DIST_MIN and ZOMBIE_SPAWN_DIST_MAX.
        """
        # Random angle with bias toward agent's facing direction (±45°)
        agent_angle = self.agent["angle"]
        offset = self.np_random.uniform(-math.pi / 4, math.pi / 4)
        spawn_angle = agent_angle + offset
        
        # Random distance
        spawn_dist = self.np_random.uniform(ZOMBIE_SPAWN_DIST_MIN, ZOMBIE_SPAWN_DIST_MAX)
        
        # Position relative to agent
        x = self.agent["pos"][0] + spawn_dist * math.cos(spawn_angle)
        y = self.agent["pos"][1] + spawn_dist * math.sin(spawn_angle)
        
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
        
        ROTATE_FINE = math.radians(1)       # ±1°
        ROTATE_LESS_FINE = math.radians(5)  # ±5°
        ROTATE_COARSE = math.radians(45)    # ±45°
        
        # Process discrete action
        action = int(action)
        
        # Movement: set direction (only one at a time, relative to facing)
        if action == 1:  # Forward
            self.move_toggles = [True, False, False, False]
        elif action == 2:  # Backward
            self.move_toggles = [False, True, False, False]
        elif action == 3:  # Strafe left
            self.move_toggles = [False, False, True, False]
        elif action == 4:  # Strafe right
            self.move_toggles = [False, False, False, True]
        elif action == 5:  # Fine rotate left
            agent["angle"] += ROTATE_FINE
        elif action == 6:  # Fine rotate right
            agent["angle"] -= ROTATE_FINE
        elif action == 7:  # Less fine rotate left
            agent["angle"] += ROTATE_LESS_FINE
        elif action == 8:  # Less fine rotate right
            agent["angle"] -= ROTATE_LESS_FINE
        elif action == 9:  # Coarse rotate left
            agent["angle"] += ROTATE_COARSE
        elif action == 10:  # Coarse rotate right
            agent["angle"] -= ROTATE_COARSE
        elif action == 11:  # Shoot
            if len(self.projectiles) < MAX_PROJECTILES:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                self.projectiles.append({
                    "pos": agent["pos"].copy(),
                    "vel": np.array([dx * PROJECTILE_SPEED, dy * PROJECTILE_SPEED]),
                    "active": True,
                    "distance_traveled": 0.0
                })
        
        # Apply continuous movement from toggles (relative to agent facing)
        angle = agent["angle"]
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        if self.move_toggles[0]:  # Forward
            agent["pos"][0] += cos_a * MOVE_STEP
            agent["pos"][1] += sin_a * MOVE_STEP
        if self.move_toggles[1]:  # Backward
            agent["pos"][0] -= cos_a * MOVE_STEP
            agent["pos"][1] -= sin_a * MOVE_STEP
        if self.move_toggles[2]:  # Strafe left (perpendicular, +90°)
            agent["pos"][0] -= sin_a * MOVE_STEP
            agent["pos"][1] += cos_a * MOVE_STEP
        if self.move_toggles[3]:  # Strafe right (perpendicular, -90°)
            agent["pos"][0] += sin_a * MOVE_STEP
            agent["pos"][1] -= cos_a * MOVE_STEP
        
        if agent["dash_cooldown"] > 0:
            agent["dash_cooldown"] -= 1
        
        # No collision handling in infinite world
        reward += self._update_projectiles()
        
        # Move zombies - only if within agent's aiming radius (PROJECTILE_MAX_RANGE)
        for zombie in self.zombies:
            if not zombie["alive"]:
                continue
            
            to_player = agent["pos"] - zombie["pos"]
            dist = np.linalg.norm(to_player)
            
            # Only move toward player if within aiming range
            if dist > 0.1 and dist <= PROJECTILE_MAX_RANGE:
                direction = to_player / dist
                zombie["pos"] = zombie["pos"] + direction * ZOMBIE_SPEED
                zombie["angle"] = math.atan2(direction[1], direction[0])
            
            if dist < ZOMBIE_ATTACK_RANGE:
                agent["health"] -= ZOMBIE_DAMAGE * 0.1
                self.was_hit = 1.0
                reward -= 1.0
            
        # Kill zombies that are too far away (2x view size = 80 units)
        max_distance = 80.0  # 2 * view_size (40)
        for zombie in self.zombies:
            if zombie["alive"]:
                dist = np.linalg.norm(zombie["pos"] - agent["pos"])
                if dist > max_distance:
                    zombie["alive"] = False
        
        # Maintain 2 zombies - spawn new ones relative to agent
        while sum(1 for z in self.zombies if z["alive"]) < 2:
            self._spawn_zombie()
        
        # Clean up dead zombies
        self.zombies = [z for z in self.zombies if z["alive"]]
        
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
    
    def _get_obs(self):
        sorted_zombies = self._get_zombies_sorted_by_distance()
        zombie_positions = [z["pos"] for z in sorted_zombies[:2]]
        while len(zombie_positions) < 2:
            zombie_positions.append(None)
        
        obs = compute_observation(
            self.agent["pos"], self.agent["angle"], self.agent["health"],
            zombie_positions, len(self.projectiles), self.move_toggles
        )
        return np.concatenate([obs, obs])
    
    def render(self):
        pass
    
    def close(self):
        pass
