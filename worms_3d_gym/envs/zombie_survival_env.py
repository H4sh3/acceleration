"""Zombie Survival Environment.

Single agent fights waves of zombies. Zombies spawn and chase the player.
Uses the same observation format as the 2-agent combat env for transfer learning.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

from .worms_3d_env import (
    MAX_HEALTH, MAX_AMMO, MAX_COOLDOWN, MAX_SPEED_FORWARD,
    DASH_COOLDOWN, DASH_DISTANCE,
    N_RAYS, RAY_MAX_RANGE, OBS_DIM,
    point_in_obstacles, line_hits_obstacle, ray_cast
)


# Zombie constants
ZOMBIE_SPEED = 0.15  # Slower than player
ZOMBIE_HEALTH = 50
ZOMBIE_DAMAGE = 10  # Damage per hit
ZOMBIE_ATTACK_RANGE = 1.5
ZOMBIE_SPAWN_INTERVAL = 50  # Ticks between spawns

# Powerup constants
POWERUP_SPAWN_INTERVAL = 100  # Ticks between powerup spawns
SPEED_BOOST_DURATION = 150  # Ticks the speed boost lasts
SPEED_BOOST_MULTIPLIER = 2.0  # 2x speed

# Velocity-based movement constants
ACCELERATION = 0.08  # How fast agent accelerates
MAX_VELOCITY = 0.5  # Maximum speed
FRICTION = 0.92  # Velocity decay per tick (1.0 = no friction)


# Extended observation dimension for zombie env (base 28 + 3 powerup info)
ZOMBIE_OBS_DIM = 31


def compute_agent_observation_zombie(agent_pos, agent_angle, agent_health, agent_velocity,
                                      closest_zombie_pos, closest_zombie_alive, 
                                      closest_powerup_pos, closest_powerup_active,
                                      obstacles, arena_size,
                                      was_hit_last_step=0.0, hit_enemy_last_step=0.0):
    """Compute observation - same format as 2-agent env but enemy = closest zombie.
    
    Added powerup info at indices 28-30:
        28: cos(delta_angle_to_powerup)
        29: sin(delta_angle_to_powerup) 
        30: normalized distance to powerup
    """
    obs = np.zeros(ZOMBIE_OBS_DIM, dtype=np.float32)
    
    x_self, y_self = agent_pos[0], agent_pos[1]
    theta_self = agent_angle
    health = agent_health
    velocity = agent_velocity
    
    cos_theta = math.cos(theta_self)
    sin_theta = math.sin(theta_self)
    
    max_enemy_dist = arena_size * math.sqrt(2)
    
    # Self state (indices 0-5)
    obs[0] = cos_theta
    obs[1] = sin_theta
    v_forward = velocity[0] * cos_theta + velocity[1] * sin_theta
    obs[2] = np.clip(v_forward / MAX_SPEED_FORWARD, -1.0, 1.0)
    obs[3] = np.clip(health / MAX_HEALTH, 0.0, 1.0)
    
    # Enemy info (indices 6-9) - closest zombie
    if closest_zombie_alive and closest_zombie_pos is not None:
        dx = closest_zombie_pos[0] - x_self
        dy = closest_zombie_pos[1] - y_self
        dist_enemy = math.sqrt(dx * dx + dy * dy)
        
        has_los = not line_hits_obstacle(x_self, y_self, closest_zombie_pos[0], closest_zombie_pos[1], obstacles)
        obs[9] = 1.0 if has_los else 0.0
        
        if has_los:
            if dist_enemy > 0.001:
                angle_to_enemy = math.atan2(dy, dx)
                delta_theta = angle_to_enemy - theta_self
                obs[6] = math.cos(delta_theta)
                obs[7] = math.sin(delta_theta)
            else:
                obs[6] = 1.0
                obs[7] = 0.0
            obs[8] = np.clip(dist_enemy / max_enemy_dist, 0.0, 1.0)
        else:
            obs[6] = 0.0
            obs[7] = 0.0
            obs[8] = 1.0
    else:
        obs[6] = 0.0
        obs[7] = 0.0
        obs[8] = 1.0
        obs[9] = 0.0
    
    # Ray sensors - wall/obstacle distance (indices 10-17)
    for i in range(N_RAYS):
        t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
        local_angle = -math.pi / 2 + t * math.pi
        global_angle = theta_self + local_angle
        dist = ray_cast(x_self, y_self, global_angle, RAY_MAX_RANGE, arena_size, obstacles)
        obs[10 + i] = dist / RAY_MAX_RANGE
    
    # Step feedback (indices 18-19)
    obs[18] = was_hit_last_step
    obs[19] = hit_enemy_last_step
    
    # Ray sensors - enemy detection (indices 20-27)
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
                        obs[20 + i] = 1.0
    
    # Powerup info (indices 28-30)
    if closest_powerup_active and closest_powerup_pos is not None:
        dx = closest_powerup_pos[0] - x_self
        dy = closest_powerup_pos[1] - y_self
        dist_powerup = math.sqrt(dx * dx + dy * dy)
        
        if dist_powerup > 0.001:
            angle_to_powerup = math.atan2(dy, dx)
            delta_theta = angle_to_powerup - theta_self
            obs[28] = math.cos(delta_theta)
            obs[29] = math.sin(delta_theta)
        else:
            obs[28] = 1.0
            obs[29] = 0.0
        obs[30] = np.clip(dist_powerup / max_enemy_dist, 0.0, 1.0)
    else:
        # No active powerup
        obs[28] = 0.0
        obs[29] = 0.0
        obs[30] = 1.0  # Max distance = no powerup
    
    return obs


class ZombieSurvivalEnv(gym.Env):
    """Zombie survival environment - fight waves of zombies."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    SIZE = 30
    
    OBSTACLES = [
        [12.5, 12.5, 17.5, 17.5],
    ]
    
    # Same action space as combat env
    N_ACTIONS = 9
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # Single agent
        self.n_agents = 1
        
        # Action space matches combat env (agent 0's actions)
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        
        # Observation: 31 dims (28 base + 3 powerup info), duplicated to 62
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
            "dash_cooldown": 0,
            "speed_boost": 0  # Remaining ticks of speed boost
        }
        
        self.powerups = []
        
        # For renderer compatibility
        self.agents = [self.agent]
        
        self.zombies = []
        self.kills = 0
        self.current_step = 0
        
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        self.last_shots = []
        
        # Spawn initial 2 zombies and 1 powerup
        for _ in range(2):
            self._spawn_zombie()
        self._spawn_powerup()
        
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
    
    def _spawn_powerup(self):
        """Spawn a speed powerup at a random location."""
        pos = self._random_spawn_pos()
        powerup = {
            "pos": pos,
            "type": "speed",
            "active": True
        }
        self.powerups.append(powerup)
    
    def _check_powerup_collection(self):
        """Check if player collected any powerups."""
        agent_pos = self.agent["pos"]
        collection_radius = 1.5

        reward = 0        
        for powerup in self.powerups:
            if not powerup["active"]:
                continue
            
            dist = np.linalg.norm(powerup["pos"] - agent_pos)
            if dist < collection_radius:
                powerup["active"] = False
                reward += 100
                if powerup["type"] == "speed":
                    self.agent["speed_boost"] = SPEED_BOOST_DURATION

        return reward
    
    def step(self, action):
        reward = 0.0
        self.last_shots = []
        self.was_hit = 0.0
        self.hit_enemy = 0.0
        
        agent = self.agent
        
        if not agent["alive"]:
            return self._get_obs(), 0.0, True, False, {"kills": self.kills}
        
        # Speed boost affects max velocity
        speed_mult = SPEED_BOOST_MULTIPLIER if agent["speed_boost"] > 0 else 1.0
        max_vel = MAX_VELOCITY * speed_mult
        accel = ACCELERATION * speed_mult
        ROTATE_STEP = 0.3
        
        # Track if agent accelerated
        moved = False
        
        # Process player action - now controls acceleration, not position
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
        elif action == 5:  # Rotate left
            agent["angle"] += ROTATE_STEP
        elif action == 6:  # Rotate right
            agent["angle"] -= ROTATE_STEP
        elif action == 7:  # Shoot
            dx = math.cos(agent["angle"])
            dy = math.sin(agent["angle"])
            reward += self._shoot_zombies(dx, dy)
        elif action == 8:  # Dash - instant velocity boost in facing direction
            if agent["dash_cooldown"] <= 0:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                dash_dist = DASH_DISTANCE * speed_mult
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
        
        # Decrement speed boost
        if agent["speed_boost"] > 0:
            agent["speed_boost"] -= 1
        
        # Handle collision (also stops velocity on collision)
        if self._handle_collision(agent):
            agent["velocity"] *= 0.5  # Reduce velocity on collision
        
        # Check powerup collection
        reward += self._check_powerup_collection()
        
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
        
        # Always maintain 1 powerup
        if sum(1 for p in self.powerups if p["active"]) < 1:
            self._spawn_powerup()
        
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
    
    def _shoot_zombies(self, dx, dy):
        """Shoot in direction, damage zombies. Returns reward."""
        reward = 0.0
        origin = self.agent["pos"].copy()
        hit = False
        
        for zombie in self.zombies:
            if not zombie["alive"]:
                continue
            
            to_zombie = zombie["pos"] - origin
            
            if dx != 0 or dy != 0:
                proj = to_zombie[0] * dx + to_zombie[1] * dy
                
                if proj > 0:
                    closest = np.array([dx * proj, dy * proj])
                    perp_dist = np.linalg.norm(to_zombie - closest)
                    
                    if perp_dist < 1.5:
                        if line_hits_obstacle(origin[0], origin[1], zombie["pos"][0], zombie["pos"][1], self.OBSTACLES):
                            continue
                        
                        damage = 25
                        zombie["health"] -= damage
                        hit = True
                        self.hit_enemy = 1.0
                        reward += damage  # Reward for damage
                        
                        if zombie["health"] <= 0:
                            zombie["alive"] = False
                            self.kills += 1
                            reward += 100  # Kill bonus
        
        self.last_shots.append({
            "origin": origin,
            "dx": dx, "dy": dy,
            "hit": hit,
            "team": 0
        })
        
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
    
    def _get_closest_powerup(self):
        """Get the closest active powerup to the player."""
        closest = None
        closest_dist = float('inf')
        
        for p in self.powerups:
            if not p["active"]:
                continue
            dist = np.linalg.norm(p["pos"] - self.agent["pos"])
            if dist < closest_dist:
                closest_dist = dist
                closest = p
        
        return closest
    
    def _get_obs(self):
        """Get observation (62 dims = 31 per slot, duplicated)."""
        closest_zombie = self._get_closest_zombie()
        closest_powerup = self._get_closest_powerup()
        
        obs = compute_agent_observation_zombie(
            self.agent["pos"], self.agent["angle"], 
            self.agent["health"], self.agent.get("velocity", np.array([0.0, 0.0])),
            closest_zombie["pos"] if closest_zombie else None,
            closest_zombie is not None and closest_zombie["alive"] if closest_zombie else False,
            closest_powerup["pos"] if closest_powerup else None,
            closest_powerup is not None and closest_powerup["active"] if closest_powerup else False,
            self.OBSTACLES, self.SIZE,
            was_hit_last_step=self.was_hit,
            hit_enemy_last_step=self.hit_enemy
        )
        
        # Duplicate to match format (slot 0 + slot 1)
        return np.concatenate([obs, obs])
    
    def render(self):
        """Render the environment. Use zombie_renderer.py for visualization."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
