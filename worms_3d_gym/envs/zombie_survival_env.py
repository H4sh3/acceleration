"""Zombie Survival Environment.

Single agent fights waves of zombies. Zombies spawn and chase the player.
Uses the same observation format as the 2-agent combat env for transfer learning.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from .worms_3d_env import (
    MAX_HEALTH, MAX_AMMO,
    DASH_COOLDOWN, DASH_DISTANCE,
    N_RAYS, RAY_MAX_RANGE,
    point_in_obstacles, line_hits_obstacle, ray_cast
)


# Zombie constants
ZOMBIE_SPEED = 0.15  # Slower than player
ZOMBIE_HEALTH = 50
ZOMBIE_DAMAGE = 10  # Damage per hit
ZOMBIE_ATTACK_RANGE = 1.5
ZOMBIE_SPAWN_INTERVAL = 50  # Ticks between spawns

# Fixed step movement constant
MOVE_STEP = 0.3  # Fixed movement step size per action

# Projectile constants
PROJECTILE_SPEED = 0.8
PROJECTILE_DAMAGE = 25
PROJECTILE_RADIUS = 0.5  # Hit detection radius
PROJECTILE_MIN_HIT_DIST = 0.5  # Minimum distance from agent for projectile to hit
PROJECTILE_MAX_RANGE = 7.0  # Maximum distance projectile can travel
MAX_PROJECTILES = 10


# Observation dimension for zombie env
# 1 health + 2 zombies x 4 (cos, sin, dist, in_range) + 8 wall rays + 1 shots + 9 quadrants = 27
ZOMBIE_OBS_DIM = 27


def compute_agent_observation_zombie(agent_pos, agent_angle, agent_health,
                                      zombie_positions, obstacles, arena_size,
                                      num_active_projectiles):
    """Compute observation for zombie survival env.
    
    Indices:
        0: Health (normalized 0-1)
        1-4: Zombie 1 (closest): cos_delta, sin_delta, dist, in_range
        5-8: Zombie 2 (next closest): cos_delta, sin_delta, dist, in_range
        9-16: Wall ray sensors (8 rays)
        17: Shots remaining (normalized 0-1)
        18-26: Quadrant position (3x3 grid, one-hot)
    """
    obs = np.zeros(ZOMBIE_OBS_DIM, dtype=np.float32)
    
    x_self, y_self = agent_pos[0], agent_pos[1]
    theta_self = agent_angle
    
    max_enemy_dist = arena_size * math.sqrt(2)
    
    # Health (index 0)
    obs[0] = np.clip(agent_health / MAX_HEALTH, 0.0, 1.0)
    
    # 2 zombies info (indices 1-8, 4 per zombie: cos, sin, dist, in_range)
    for i, zombie_pos in enumerate(zombie_positions[:2]):
        base_idx = 1 + i * 4
        if zombie_pos is not None:
            dx = zombie_pos[0] - x_self
            dy = zombie_pos[1] - y_self
            dist_enemy = math.sqrt(dx * dx + dy * dy)
            
            if dist_enemy > 0.001:
                angle_to_enemy = math.atan2(dy, dx)
                delta_theta = angle_to_enemy - theta_self
                obs[base_idx] = math.cos(delta_theta)
                obs[base_idx + 1] = math.sin(delta_theta)
            else:
                obs[base_idx] = 1.0
                obs[base_idx + 1] = 0.0
            obs[base_idx + 2] = np.clip(dist_enemy / max_enemy_dist, 0.0, 1.0)
            # In range flag: 1.0 if zombie is within projectile range, 0.0 otherwise
            obs[base_idx + 3] = 1.0 if dist_enemy <= PROJECTILE_MAX_RANGE else 0.0
        else:
            # No zombie in this slot - default values
            obs[base_idx] = 0.0
            obs[base_idx + 1] = 0.0
            obs[base_idx + 2] = 1.0  # Max distance
            obs[base_idx + 3] = 0.0  # Not in range
    
    # Ray sensors - wall/obstacle distance (indices 9-16)
    for i in range(N_RAYS):
        t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
        local_angle = -math.pi / 2 + t * math.pi
        global_angle = theta_self + local_angle
        dist = ray_cast(x_self, y_self, global_angle, RAY_MAX_RANGE, arena_size, obstacles)
        obs[9 + i] = dist / RAY_MAX_RANGE
    
    # Shots remaining (index 17)
    shots_remaining = MAX_PROJECTILES - num_active_projectiles
    obs[17] = shots_remaining / MAX_PROJECTILES
    
    # Quadrant inputs (indices 18-26): 3x3 grid, one-hot encoding
    # Quadrant layout (looking at map from above):
    #   6 | 7 | 8   (top row, y > 2/3)
    #   3 | 4 | 5   (middle row)
    #   0 | 1 | 2   (bottom row, y < 1/3)
    quadrant_x = int(np.clip(x_self / arena_size * 3, 0, 2))  # 0, 1, or 2
    quadrant_y = int(np.clip(y_self / arena_size * 3, 0, 2))  # 0, 1, or 2
    quadrant_idx = quadrant_y * 3 + quadrant_x  # 0-8
    obs[18 + quadrant_idx] = 1.0
    
    return obs


class ZombieSurvivalEnv(gym.Env):
    """Zombie survival environment - fight waves of zombies."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    SIZE = 30
    
    OBSTACLES = [
        [12.5, 12.5, 17.5, 17.5],
    ]
    
    # Action space: 0=noop, 1-4=move, 5-6=fine rotate, 7-8=coarse rotate, 9=shoot
    N_ACTIONS = 10
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Single agent
        self.n_agents = 1
        
        # Action space matches combat env (agent 0's actions)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        
        # Observation: 27 dims, duplicated to 54
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ZOMBIE_OBS_DIM * 2,), dtype=np.float32
        )
        
        self.zombies = []
        self.kills = 0
    
    def _random_spawn_pos(self):
        """Get a random spawn position not inside obstacles."""
        margin = 3
        for _ in range(100):  # Max attempts
            x = self.np_random.uniform(margin, self.SIZE - margin)
            y = self.np_random.uniform(margin, self.SIZE - margin)
            if not point_in_obstacles(x, y, self.OBSTACLES):
                return np.array([x, y])
        # Fallback to center
        return np.array([self.SIZE / 2, self.SIZE / 2])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Spawn player at random position (not on obstacle)
        spawn_pos = self._random_spawn_pos()
        random_angle = self.np_random.uniform(-math.pi, math.pi)
        
        self.agent = {
            "id": 0, "team": 0,
            "pos": spawn_pos,
            "health": MAX_HEALTH,
            "angle": random_angle,
            "alive": True,
            "ammo": MAX_AMMO,
            "cooldown": 0,
            "dash_cooldown": 0
        }
        
        # For renderer compatibility
        self.agents = [self.agent]
        
        self.zombies = []
        self.kills = 0
        self.current_step = 0
        
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        self.last_shots = []
        self.projectiles = []  # Active projectiles
        self.spawn_edge_index = 0  # For rotating spawn: 0=top, 1=left, 2=right, 3=bottom
        
        # Spawn initial 4 zombies
        for _ in range(4):
            self._spawn_zombie()
        
        return self._get_obs(), {}
    
    def _spawn_zombie(self):       
        # Rotate through edges: top -> left -> right -> bottom
        edge = self.spawn_edge_index
        self.spawn_edge_index = (self.spawn_edge_index + 1) % 4
        margin = 2
        
        if edge == 0:  # Top
            x = self.SIZE // 2
            y = self.SIZE - margin
        elif edge == 1:  # Left
            x = margin
            y = self.SIZE // 2
        elif edge == 2:  # Right
            x = self.SIZE - margin
            y = self.SIZE // 2
        else:  # Bottom
            x = self.SIZE // 2
            y = margin
        
        # Don't spawn inside obstacles
        if point_in_obstacles(x, y, self.OBSTACLES):
            return
        
        zombie = {
            "id": len(self.zombies),
            "team": 1,  # Enemy team for rendering
            "pos": np.array([x, y]),
            "health": ZOMBIE_HEALTH,
            "angle": 0.0,
            "alive": True
        }
        self.zombies.append(zombie)

    def _get_zombies_sorted_by_distance(self):
        """Get alive zombies sorted by distance to player (closest first)."""
        alive_zombies = [z for z in self.zombies if z["alive"]]
        alive_zombies.sort(key=lambda z: np.linalg.norm(z["pos"] - self.agent["pos"]))
        return alive_zombies
    
    def step(self, action):
        reward = 0.0
        self.last_shots = []
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        agent = self.agent
        
        if not agent["alive"]:
            return self._get_obs(), 0.0, True, False, {"kills": self.kills}
        
        ROTATE_FINE = 0.05  # Small rotation
        ROTATE_COARSE = 0.3  # Large rotation
        
        # Process player action
        # 0=noop, 1-4=move (fixed step), 5-6=fine rotate, 7-8=coarse rotate, 9=shoot
        if action == 1:  # Move Up (Y+)
            agent["pos"][1] += MOVE_STEP
        elif action == 2:  # Move Down (Y-)
            agent["pos"][1] -= MOVE_STEP
        elif action == 3:  # Move Left (X-)
            agent["pos"][0] -= MOVE_STEP
        elif action == 4:  # Move Right (X+)
            agent["pos"][0] += MOVE_STEP
        elif action == 5:  # Fine rotate left
            agent["angle"] += ROTATE_FINE
        elif action == 6:  # Fine rotate right
            agent["angle"] -= ROTATE_FINE
        elif action == 7:  # Coarse rotate left
            agent["angle"] += ROTATE_COARSE
        elif action == 8:  # Coarse rotate right
            agent["angle"] -= ROTATE_COARSE
        elif action == 9:  # Shoot projectile
            if len(self.projectiles) < MAX_PROJECTILES:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                self.projectiles.append({
                    "pos": agent["pos"].copy(),
                    "vel": np.array([dx * PROJECTILE_SPEED, dy * PROJECTILE_SPEED]),
                    "active": True,
                    "distance_traveled": 0.0
                })
        
        
        # Decrement dash cooldown
        if agent["dash_cooldown"] > 0:
            agent["dash_cooldown"] -= 1
        
        # Handle collision
        self._handle_collision(agent)
        
        # Update projectiles
        reward += self._update_projectiles()
        
        # Move zombies toward player with fixed steps (no acceleration)
        for zombie in self.zombies:
            if not zombie["alive"]:
                continue
            
            # Direction to player
            to_player = agent["pos"] - zombie["pos"]
            dist = np.linalg.norm(to_player)
            
            if dist > 0.1:
                direction = to_player / dist
                # Fixed step movement - move exactly ZOMBIE_SPEED units per tick
                zombie["pos"] = zombie["pos"] + direction * ZOMBIE_SPEED
                zombie["angle"] = math.atan2(direction[1], direction[0])
            
            # Attack if in range
            if dist < ZOMBIE_ATTACK_RANGE:
                agent["health"] -= ZOMBIE_DAMAGE * 0.1  # Damage per tick when close
                self.was_hit = 1.0
                reward -= 1.0  # Penalty for taking damage
            
            # Handle zombie collision
            self._handle_collision(zombie)
        
        # Always maintain 4 zombies
        while sum(1 for z in self.zombies if z["alive"]) < 4:
            self._spawn_zombie()
        
        # Hunting rewards: incentivize active engagement over camping
        alive_zombies = [z for z in self.zombies if z["alive"]]
        if alive_zombies:
            # Find closest zombie
            closest_zombie = min(alive_zombies, key=lambda z: np.linalg.norm(z["pos"] - agent["pos"]))
            closest_dist = np.linalg.norm(closest_zombie["pos"] - agent["pos"])
            
            # Reward for being at optimal engagement range (not too close, not too far)
            # Optimal range: 3-8 units (close enough to shoot, far enough to dodge)
            OPTIMAL_MIN = 3.0
            OPTIMAL_MAX = 8.0
            if closest_dist < OPTIMAL_MIN:
                # Too close - small penalty
                reward -= 0.05
            elif closest_dist <= OPTIMAL_MAX:
                # In optimal range - bonus!
                reward += 0.1
            else:
                # Too far - penalty that scales with distance
                reward -= 0.02 * (closest_dist - OPTIMAL_MAX)
            
        
        # Check player death
        if agent["health"] <= 0:
            agent["alive"] = False
            reward -= 100  # Death penalty
        
        self.current_step += 1
        
        terminated = not agent["alive"]
        truncated = self.current_step >= 1000  # Longer episodes for survival
        
        # Update agents list for renderer
        self.agents = [self.agent] + [z for z in self.zombies if z["alive"]]
        
        return self._get_obs(), reward, terminated, truncated, {"kills": self.kills}
    
    def _update_projectiles(self):
        """Update all projectiles - move them and check for hits. Returns reward."""
        reward = 0.0
        self.last_shots = []  # Clear for renderer
        
        for proj in self.projectiles:
            if not proj["active"]:
                continue
            
            # Move projectile
            proj["pos"] += proj["vel"]
            proj["distance_traveled"] += PROJECTILE_SPEED
            
            # Check max range - deactivate if traveled too far
            if proj["distance_traveled"] >= PROJECTILE_MAX_RANGE:
                proj["active"] = False
                continue
            
            # Check bounds - deactivate if out of arena
            if (proj["pos"][0] < 0 or proj["pos"][0] > self.SIZE or
                proj["pos"][1] < 0 or proj["pos"][1] > self.SIZE):
                proj["active"] = False
                continue
            
            # Check obstacle collision
            if point_in_obstacles(proj["pos"][0], proj["pos"][1], self.OBSTACLES):
                proj["active"] = False
                continue
            
            # Check zombie hits
            for zombie in self.zombies:
                if not zombie["alive"]:
                    continue
                
                # Check if zombie is too close to agent (can't hit point-blank)
                zombie_agent_dist = np.linalg.norm(zombie["pos"] - self.agent["pos"])
                if zombie_agent_dist < PROJECTILE_MIN_HIT_DIST:
                    continue
                
                dist = np.linalg.norm(proj["pos"] - zombie["pos"])
                if dist < PROJECTILE_RADIUS + 0.5:  # 0.5 = zombie radius
                    zombie["health"] -= PROJECTILE_DAMAGE
                    self.hit_enemy = 1.0
                    reward += PROJECTILE_DAMAGE
                    proj["active"] = False
                    
                    if zombie["health"] <= 0:
                        zombie["alive"] = False
                        self.kills += 1
                        reward += 100  # Kill bonus
                    break
            
            # Record for renderer
            if proj["active"]:
                self.last_shots.append({
                    "pos": proj["pos"].copy(),
                    "vel": proj["vel"].copy(),
                    "active": True
                })
        
        # Remove inactive projectiles
        self.projectiles = [p for p in self.projectiles if p["active"]]
        
        return reward
    
    def _handle_collision(self, entity):
        """Bounds and obstacle collision. Returns True if collision occurred."""
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
        """Get observation (54 dims = 27 per slot, duplicated)."""
        sorted_zombies = self._get_zombies_sorted_by_distance()
        
        # Get positions of up to 2 zombies (closest and next closest), pad with None if fewer
        zombie_positions = [z["pos"] for z in sorted_zombies[:2]]
        while len(zombie_positions) < 2:
            zombie_positions.append(None)
        
        obs = compute_agent_observation_zombie(
            self.agent["pos"], self.agent["angle"], self.agent["health"],
            zombie_positions, self.OBSTACLES, self.SIZE,
            len(self.projectiles)
        )
        
        # Duplicate to match format (slot 0 + slot 1)
        return np.concatenate([obs, obs])
    
    def render(self):
        """Render the environment. Use zombie_renderer.py for visualization."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
