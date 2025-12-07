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

# Velocity-based movement constants
ACCELERATION = 0.08  # How fast agent accelerates
MAX_VELOCITY = 0.5  # Maximum speed
FRICTION = 0.92  # Velocity decay per tick (1.0 = no friction)

# Projectile constants
PROJECTILE_SPEED = 0.8
PROJECTILE_DAMAGE = 25
PROJECTILE_RADIUS = 0.5  # Hit detection radius
MAX_PROJECTILES = 5


# Observation dimension for zombie env
# 1 health + 4 closest zombie + 8 wall rays + 8 enemy rays + 1 aim + 1 shots = 23
ZOMBIE_OBS_DIM = 23


def compute_agent_observation_zombie(agent_pos, agent_angle, agent_health,
                                      closest_zombie_pos, closest_zombie_alive, 
                                      obstacles, arena_size,
                                      num_active_projectiles):
    """Compute observation for zombie survival env.
    
    Indices:
        0: Health (normalized 0-1)
        1-4: Closest zombie (cos_delta, sin_delta, dist, LOS)
        5-12: Wall ray sensors (8 rays)
        13-20: Enemy ray sensors (8 rays)
        21: Aim indicator (1.0 if on target)
        22: Shots remaining (normalized 0-1)
    """
    obs = np.zeros(ZOMBIE_OBS_DIM, dtype=np.float32)
    
    x_self, y_self = agent_pos[0], agent_pos[1]
    theta_self = agent_angle
    
    cos_theta = math.cos(theta_self)
    sin_theta = math.sin(theta_self)
    
    max_enemy_dist = arena_size * math.sqrt(2)
    
    # Health (index 0)
    obs[0] = np.clip(agent_health / MAX_HEALTH, 0.0, 1.0)
    
    # Closest zombie info (indices 1-4)
    if closest_zombie_alive and closest_zombie_pos is not None:
        dx = closest_zombie_pos[0] - x_self
        dy = closest_zombie_pos[1] - y_self
        dist_enemy = math.sqrt(dx * dx + dy * dy)
        
        has_los = not line_hits_obstacle(x_self, y_self, closest_zombie_pos[0], closest_zombie_pos[1], obstacles)
        obs[4] = 1.0 if has_los else 0.0
        
        if has_los:
            if dist_enemy > 0.001:
                angle_to_enemy = math.atan2(dy, dx)
                delta_theta = angle_to_enemy - theta_self
                obs[1] = math.cos(delta_theta)
                obs[2] = math.sin(delta_theta)
            else:
                obs[1] = 1.0
                obs[2] = 0.0
            obs[3] = np.clip(dist_enemy / max_enemy_dist, 0.0, 1.0)
        else:
            obs[1] = 0.0
            obs[2] = 0.0
            obs[3] = 1.0
    else:
        obs[1] = 0.0
        obs[2] = 0.0
        obs[3] = 1.0
        obs[4] = 0.0
    
    # Ray sensors - wall/obstacle distance (indices 5-12)
    for i in range(N_RAYS):
        t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
        local_angle = -math.pi / 2 + t * math.pi
        global_angle = theta_self + local_angle
        dist = ray_cast(x_self, y_self, global_angle, RAY_MAX_RANGE, arena_size, obstacles)
        obs[5 + i] = dist / RAY_MAX_RANGE
    
    # Ray sensors - enemy detection (indices 13-20)
    if closest_zombie_alive and closest_zombie_pos is not None:
        x_enemy, y_enemy = closest_zombie_pos
        enemy_radius = 0.5
        
        for i in range(N_RAYS):
            t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
            local_angle = -math.pi / 2 + t * math.pi
            global_angle = theta_self + local_angle
            
            ray_dx = math.cos(global_angle)
            ray_dy = math.sin(global_angle)
            
            to_enemy_x = x_enemy - x_self
            to_enemy_y = y_enemy - y_self
            
            proj = to_enemy_x * ray_dx + to_enemy_y * ray_dy
            
            if proj > 0:
                closest_x = ray_dx * proj
                closest_y = ray_dy * proj
                perp_dist = math.sqrt((to_enemy_x - closest_x)**2 + (to_enemy_y - closest_y)**2)
                
                if perp_dist < enemy_radius:
                    if not line_hits_obstacle(x_self, y_self, closest_zombie_pos[0], closest_zombie_pos[1], obstacles):
                        obs[13 + i] = 1.0
    
    # Aim indicator (index 21) - 1.0 if aiming at closest zombie and would hit
    obs[21] = 0.0
    if closest_zombie_alive and closest_zombie_pos is not None:
        to_zombie_x = closest_zombie_pos[0] - x_self
        to_zombie_y = closest_zombie_pos[1] - y_self
        
        proj = to_zombie_x * cos_theta + to_zombie_y * sin_theta
        
        if proj > 0:  # Zombie is in front
            closest_on_ray_x = cos_theta * proj
            closest_on_ray_y = sin_theta * proj
            perp_dist = math.sqrt((to_zombie_x - closest_on_ray_x)**2 + (to_zombie_y - closest_on_ray_y)**2)
            
            hit_radius = 1.0
            if perp_dist < hit_radius:
                if not line_hits_obstacle(x_self, y_self, closest_zombie_pos[0], closest_zombie_pos[1], obstacles):
                    obs[21] = 1.0
    
    # Shots remaining (index 22)
    shots_remaining = MAX_PROJECTILES - num_active_projectiles
    obs[22] = shots_remaining / MAX_PROJECTILES
    
    return obs


class ZombieSurvivalEnv(gym.Env):
    """Zombie survival environment - fight waves of zombies."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    SIZE = 30
    
    OBSTACLES = [
        [12.5, 12.5, 17.5, 17.5],
    ]
    
    # Action space: 0=noop, 1-4=move, 5-6=fine rotate, 7-8=coarse rotate, 9=shoot, 10=dash
    N_ACTIONS = 11
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Single agent
        self.n_agents = 1
        
        # Action space matches combat env (agent 0's actions)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        
        # Observation: 28 dims, duplicated to 56
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
            "velocity": np.array([0.0, 0.0]),
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
        
        # Spawn initial 2 zombies
        for _ in range(2):
            self._spawn_zombie()
        
        return self._get_obs(), {}
    
    def _spawn_zombie(self):       
        # Pick random edge
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
        ROTATE_FINE = 0.05  # Small rotation
        ROTATE_COARSE = 0.3  # Large rotation
        
        # Track if agent accelerated
        moved = False
        
        # Process player action
        # 0=noop, 1-4=move, 5-6=fine rotate, 7-8=coarse rotate, 9=shoot, 10=dash
        if action == 1:  # Accelerate Up (Y+)
            agent["velocity"][1] += accel
            moved = True
        elif action == 2:  # Accelerate Down (Y-)
            agent["velocity"][1] -= accel
            moved = True
        elif action == 3:  # Accelerate Left (X-)
            agent["velocity"][0] -= accel
            moved = True
        elif action == 4:  # Accelerate Right (X+)
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
        elif action == 9:  # Shoot projectile
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
                dash_dist = DASH_DISTANCE
                agent["pos"][0] += dx * dash_dist
                agent["pos"][1] += dy * dash_dist
                agent["dash_cooldown"] = DASH_COOLDOWN
                moved = True
        
        # Movement reward
        if moved:
            reward += 0.05
        
        # Clamp velocity to max speed
        vel_magnitude = np.linalg.norm(agent["velocity"])
        if vel_magnitude > max_vel:
            agent["velocity"] = agent["velocity"] / vel_magnitude * max_vel
        
        # Apply velocity to position
        agent["pos"] += agent["velocity"]
        
        # Apply friction (velocity decay)
        agent["velocity"] *= FRICTION
        
        # Decrement dash cooldown
        if agent["dash_cooldown"] > 0:
            agent["dash_cooldown"] -= 1
        
        # Handle collision (also stops velocity on collision)
        if self._handle_collision(agent):
            agent["velocity"] *= 0.5  # Reduce velocity on collision
        
        # Update projectiles
        reward += self._update_projectiles()
        
        # Move zombies toward player
        for zombie in self.zombies:
            if not zombie["alive"]:
                continue
            
            # Direction to player
            to_player = agent["pos"] - zombie["pos"]
            dist = np.linalg.norm(to_player)
            
            if dist > 0.1:
                direction = to_player / dist
                zombie["pos"] += direction * ZOMBIE_SPEED
                zombie["angle"] = math.atan2(direction[1], direction[0])
            
            # Attack if in range
            if dist < ZOMBIE_ATTACK_RANGE:
                agent["health"] -= ZOMBIE_DAMAGE * 0.1  # Damage per tick when close
                self.was_hit = 1.0
                reward -= 1.0  # Penalty for taking damage
            
            # Handle zombie collision
            self._handle_collision(zombie)
        
        # Always maintain 2 zombies
        while sum(1 for z in self.zombies if z["alive"]) < 2:
            self._spawn_zombie()
        
        # Check player death
        if agent["health"] <= 0:
            agent["alive"] = False
            reward -= 100  # Death penalty
        
        # Survival bonus
        reward += 0.1
        
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
        """Get observation (46 dims = 23 per slot, duplicated)."""
        closest_zombie = self._get_closest_zombie()
        
        obs = compute_agent_observation_zombie(
            self.agent["pos"], self.agent["angle"], self.agent["health"],
            closest_zombie["pos"] if closest_zombie else None,
            closest_zombie is not None and closest_zombie["alive"] if closest_zombie else False,
            self.OBSTACLES, self.SIZE,
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
