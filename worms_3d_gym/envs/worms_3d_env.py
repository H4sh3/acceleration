"""Minimal 2D Combat Gym Environment.

Two agents on a grid. Move or shoot in 4 directions. First to kill wins.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# =============================================================================
# Environment Constants
# =============================================================================
MAX_HEALTH = 100.0
MAX_AMMO = 10
MAX_COOLDOWN = 5  # steps
MAX_SPEED_FORWARD = 0.3  # units per step
N_RAYS = 8
RAY_MAX_RANGE = 30.0  # max ray distance (matches arena size)
OBS_DIM = 28  # total observation dimensions (20 base + 8 enemy rays)


# =============================================================================
# Geometry Helpers
# =============================================================================
def line_hits_obstacle(x1, y1, x2, y2, obstacle):
    """Check if line segment from (x1,y1) to (x2,y2) intersects obstacle.
    
    Args:
        x1, y1: Start point
        x2, y2: End point  
        obstacle: [x_min, y_min, x_max, y_max] bounding box, or None
    
    Returns:
        True if line intersects obstacle, False otherwise
    """
    if obstacle is None:
        return False
    
    ox_min, oy_min, ox_max, oy_max = obstacle
    
    # Check multiple points along the line
    steps = 20
    for i in range(steps + 1):
        t = i / steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        if ox_min <= px <= ox_max and oy_min <= py <= oy_max:
            return True
    return False


def ray_cast(x, y, angle, max_range, arena_size, obstacle):
    """Cast a ray and return distance to nearest hit.
    
    Args:
        x, y: Ray origin
        angle: Ray direction in radians
        max_range: Maximum ray distance
        arena_size: Size of the arena (assumes square [0, arena_size])
        obstacle: [x_min, y_min, x_max, y_max] bounding box
    
    Returns:
        Distance to nearest hit, capped at max_range
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # Step along ray
    step_size = 0.2
    dist = 0.0
    
    while dist < max_range:
        px = x + dx * dist
        py = y + dy * dist
        
        # Check arena bounds
        if px < 0 or px >= arena_size or py < 0 or py >= arena_size:
            return dist
        
        # Check obstacle
        if obstacle is not None:
            ox_min, oy_min, ox_max, oy_max = obstacle
            if ox_min <= px <= ox_max and oy_min <= py <= oy_max:
                return dist
        
        dist += step_size
    
    return max_range


# =============================================================================
# Observation Computation
# =============================================================================
def compute_agent_observation(agent, enemy, obstacle, arena_size, 
                               was_hit_last_step=0.0, hit_enemy_last_step=0.0):
    """Compute 20-dimensional egocentric observation vector for a single agent.
    
    Args:
        agent: Dict with keys:
            - 'pos': np.array[2] position
            - 'health': float
            - 'angle': float (radians)
            - 'alive': bool
            - 'velocity': np.array[2] (optional, defaults to [0,0])
            - 'ammo': int (optional, defaults to MAX_AMMO)
            - 'cooldown': int (optional, defaults to 0)
        enemy: Dict with same structure as agent
        obstacle: [x_min, y_min, x_max, y_max] bounding box
        arena_size: Size of the square arena
        was_hit_last_step: 1.0 if agent took damage last step, else 0.0
        hit_enemy_last_step: 1.0 if agent hit enemy last step, else 0.0
    
    Returns:
        np.array of 20 floats (see observation spec)
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    
    # Get agent state
    x_self, y_self = agent["pos"][0], agent["pos"][1]
    theta_self = agent["angle"]
    health = agent["health"]
    velocity = agent.get("velocity", np.array([0.0, 0.0]))
    ammo = agent.get("ammo", MAX_AMMO)
    cooldown = agent.get("cooldown", 0)
    
    cos_theta = math.cos(theta_self)
    sin_theta = math.sin(theta_self)
    
    # Max enemy distance (arena diagonal)
    max_enemy_dist = arena_size * math.sqrt(2)
    
    # =========================================================================
    # 3.1 Self state (indices 0-5)
    # =========================================================================
    obs[0] = cos_theta  # cos_theta_self
    obs[1] = sin_theta  # sin_theta_self
    
    # Forward velocity (component along facing direction)
    v_forward = velocity[0] * cos_theta + velocity[1] * sin_theta
    obs[2] = np.clip(v_forward / MAX_SPEED_FORWARD, -1.0, 1.0)  # v_forward_norm
    
    obs[3] = np.clip(health / MAX_HEALTH, 0.0, 1.0)  # health_norm
    obs[4] = np.clip(ammo / MAX_AMMO, 0.0, 1.0)  # ammo_norm
    obs[5] = np.clip(cooldown / MAX_COOLDOWN, 0.0, 1.0)  # cooldown_norm
    
    # =========================================================================
    # 3.2 Enemy info (indices 6-9)
    # =========================================================================
    if enemy["alive"]:
        dx = enemy["pos"][0] - x_self
        dy = enemy["pos"][1] - y_self
        dist_enemy = math.sqrt(dx * dx + dy * dy)
        
        if dist_enemy > 0.001:  # Avoid division by zero
            angle_to_enemy = math.atan2(dy, dx)
            delta_theta = angle_to_enemy - theta_self
            
            obs[6] = math.cos(delta_theta)  # cos_delta_enemy
            obs[7] = math.sin(delta_theta)  # sin_delta_enemy
        else:
            # Same position edge case
            obs[6] = 1.0  # cos_delta_enemy
            obs[7] = 0.0  # sin_delta_enemy
        
        obs[8] = np.clip(dist_enemy / max_enemy_dist, 0.0, 1.0)  # dist_enemy_norm
        
        # Line of sight check
        has_los = not line_hits_obstacle(x_self, y_self, enemy["pos"][0], enemy["pos"][1], obstacle)
        obs[9] = 1.0 if has_los else 0.0  # has_los
    else:
        # Enemy dead defaults
        obs[6] = 0.0  # cos_delta_enemy
        obs[7] = 0.0  # sin_delta_enemy
        obs[8] = 1.0  # dist_enemy_norm
        obs[9] = 0.0  # has_los
    
    # =========================================================================
    # 3.3 Ray sensors - wall/obstacle distance (indices 10-17)
    # =========================================================================
    for i in range(N_RAYS):
        # Ray angles from -π/2 to +π/2 around agent's heading
        t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
        local_angle = -math.pi / 2 + t * math.pi
        global_angle = theta_self + local_angle
        
        dist = ray_cast(x_self, y_self, global_angle, RAY_MAX_RANGE, arena_size, obstacle)
        obs[10 + i] = dist / RAY_MAX_RANGE  # ray_i_dist normalized
    
    # =========================================================================
    # 3.4 Step feedback (indices 18-19)
    # =========================================================================
    obs[18] = was_hit_last_step
    obs[19] = hit_enemy_last_step
    
    # =========================================================================
    # 3.5 Ray sensors - enemy detection (indices 20-27)
    # 1.0 if ray hits enemy, 0.0 otherwise
    # =========================================================================
    if enemy["alive"]:
        x_enemy, y_enemy = enemy["pos"]
        enemy_radius = 0.5  # Enemy hitbox radius for ray detection
        
        for i in range(N_RAYS):
            t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
            local_angle = -math.pi / 2 + t * math.pi
            global_angle = theta_self + local_angle
            
            # Ray direction
            ray_dx = math.cos(global_angle)
            ray_dy = math.sin(global_angle)
            
            # Vector from agent to enemy
            to_enemy_x = x_enemy - x_self
            to_enemy_y = y_enemy - y_self
            
            # Project enemy onto ray direction
            proj = to_enemy_x * ray_dx + to_enemy_y * ray_dy
            
            if proj > 0:  # Enemy is in front along this ray
                # Perpendicular distance from ray to enemy center
                closest_x = ray_dx * proj
                closest_y = ray_dy * proj
                perp_dist = math.sqrt((to_enemy_x - closest_x)**2 + (to_enemy_y - closest_y)**2)
                
                # Check if ray hits enemy (within radius) and no obstacle blocks
                if perp_dist < enemy_radius:
                    # Check if obstacle blocks the view
                    if not line_hits_obstacle(x_self, y_self, x_enemy, y_enemy, obstacle):
                        obs[20 + i] = 1.0
    # Enemy dead or not hit by any ray: indices 20-27 stay 0.0
    
    return obs


class Worms3DEnv(gym.Env):
    """Simple 2D grid combat environment."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Grid size
    SIZE = 30
    
    # Obstacle: None for open arena, or [x_min, y_min, x_max, y_max]
    OBSTACLE = [10, 10, 20, 20]  # 10x10 box in center
    
    # Actions: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_left, 6=rotate_right, 7=shoot
    N_ACTIONS = 8
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 2 agents
        self.n_agents = 2
        
        # Action: one discrete action per agent
        self.action_space = spaces.MultiDiscrete([self.N_ACTIONS] * self.n_agents)
        
        # Observation: 20 dims per agent = 40 total
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM * self.n_agents,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Spawn agents on opposite sides of arena
        # Agent 0 on left side, Agent 1 on right side
        margin = 5
        y_center = self.SIZE / 2
        
        pos0 = np.array([margin, y_center])
        pos1 = np.array([self.SIZE - margin, y_center])
        
        # Calculate angle to face each other, then add random offset
        # Agent 0 should face right (0), Agent 1 should face left (π)
        # Add random rotation offset of ±45° to ±90° so they need to aim
        rotation_offset_0 = self.np_random.uniform(-math.pi/2, math.pi/2)
        rotation_offset_1 = self.np_random.uniform(-math.pi/2, math.pi/2)
        
        angle0 = 0 + rotation_offset_0  # Facing right-ish
        angle1 = math.pi + rotation_offset_1  # Facing left-ish
        
        self.agents = [
            {
                "id": 0, "team": 0, 
                "pos": pos0, 
                "health": MAX_HEALTH, "angle": angle0, "alive": True,
                "velocity": np.array([0.0, 0.0]),
                "ammo": MAX_AMMO, "cooldown": 0
            },
            {
                "id": 1, "team": 1, 
                "pos": pos1, 
                "health": MAX_HEALTH, "angle": angle1, "alive": True,
                "velocity": np.array([0.0, 0.0]),
                "ammo": MAX_AMMO, "cooldown": 0
            },
        ]
        
        # Track shots for rendering
        self.last_shots = []
        
        # Step feedback tracking
        self.was_hit = [0.0, 0.0]  # per agent
        self.hit_enemy = [0.0, 0.0]  # per agent
        
        # Exploration tracking - visited cells per agent
        self.visited = [set(), set()]
        
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, actions):
        # actions is array of shape (TOTAL_AGENTS,) - one action per agent
        # Action: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_left, 6=rotate_right, 7=shoot
        
        rewards = np.zeros(self.n_agents)
        self.last_shots = []  # Clear shots from previous step
        
        # Store previous health for damage calculation
        prev_health = [a["health"] for a in self.agents]
        self.was_hit = [0.0, 0.0]
        self.hit_enemy = [0.0, 0.0]
        
        MOVE_STEP = MAX_SPEED_FORWARD
        ROTATE_STEP = 0.3  # ~17 degrees per step
        ARENA_DIAGONAL = self.SIZE * math.sqrt(2)  # ~42.426 for 30x30
        
        for i, agent in enumerate(self.agents):
            if not agent["alive"]:
                continue
            
            act = actions[i]
            enemy = self.agents[1 - i]
            
            # Movement actions (1-4)
            if act == 1:  # Up (Y+)
                agent["pos"][1] += MOVE_STEP
            elif act == 2:  # Down (Y-)
                agent["pos"][1] -= MOVE_STEP
            elif act == 3:  # Left (X-)
                agent["pos"][0] -= MOVE_STEP
            elif act == 4:  # Right (X+)
                agent["pos"][0] += MOVE_STEP
            
            # Rotation actions (5-6) - absolute left/right
            elif act == 5:  # Rotate left (counter-clockwise)
                agent["angle"] += ROTATE_STEP
            elif act == 6:  # Rotate right (clockwise)
                agent["angle"] -= ROTATE_STEP
            
            # Shoot in facing direction (7)
            elif act == 7:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                self._shoot_direction(agent, dx, dy)  # Damage applied inside, rewards computed below
            
            # Collision / Boundary Check
            self._handle_collision(agent)
            
            # =================================================================
            # SIMPLE REWARD SYSTEM: Only hits and kills + exploration
            # =================================================================
            
            # Exploration bonus: reward visiting new cells
            cell = (int(agent["pos"][0]), int(agent["pos"][1]))
            if cell not in self.visited[i]:
                self.visited[i].add(cell)
                rewards[i] += 0.1  # Small bonus for new cell
        
        # Damage rewards (computed after all actions processed)
        for i, agent in enumerate(self.agents):
            enemy = self.agents[1 - i]
            
            # Reward for dealing damage: +1.0 per damage point
            enemy_damage = prev_health[1 - i] - enemy["health"]
            if enemy_damage > 0 and self.hit_enemy[i] > 0:
                rewards[i] += enemy_damage  # +25 per hit
            
            # Kill bonus
            if prev_health[1 - i] > 0 and not enemy["alive"]:
                rewards[i] += 1000
            
            # Death penalty
            if prev_health[i] > 0 and not agent["alive"]:
                rewards[i] -= 1000

        # Track steps
        self.current_step = getattr(self, 'current_step', 0) + 1
        
        # Check if done (only 1 team left)
        alive_teams = set(a["team"] for a in self.agents if a["alive"])
        terminated = len(alive_teams) <= 1
        truncated = self.current_step >= 200  # Max 200 steps per episode
        
        total_reward = float(np.sum(rewards))
        
        return self._get_obs(), total_reward, terminated, truncated, {"alive_teams": list(alive_teams)}

    def _get_obs(self):
        """Get 40-dim observation (20 per agent)."""
        a0, a1 = self.agents[0], self.agents[1]
        obs0 = compute_agent_observation(
            a0, a1, self.OBSTACLE, self.SIZE,
            was_hit_last_step=self.was_hit[0],
            hit_enemy_last_step=self.hit_enemy[0]
        )
        obs1 = compute_agent_observation(
            a1, a0, self.OBSTACLE, self.SIZE,
            was_hit_last_step=self.was_hit[1],
            hit_enemy_last_step=self.hit_enemy[1]
        )
        return np.concatenate([obs0, obs1])
    
    def _check_aiming(self, agent, enemy):
        """Check if agent is aiming at enemy. Returns 1.0 if cos_delta_enemy > 0.9 and has_los."""
        obs = compute_agent_observation(agent, enemy, self.OBSTACLE, self.SIZE)
        # cos_delta_enemy is at index 6, has_los at index 9
        return 1.0 if obs[6] > 0.9 and obs[9] > 0.5 else 0.0
    
    def _check_enemy_in_rays(self, agent, enemy):
        """Check if enemy is detected by any of the agent's ray sensors.
        
        Returns True if enemy is within any ray's detection cone (±90° around heading).
        """
        # Get angle to enemy
        dx = enemy["pos"][0] - agent["pos"][0]
        dy = enemy["pos"][1] - agent["pos"][1]
        angle_to_enemy = math.atan2(dy, dx)
        
        # Relative angle from agent's heading
        delta = angle_to_enemy - agent["angle"]
        # Normalize to [-π, π]
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        
        # Check if enemy is within ray sensor cone (±90°)
        if abs(delta) > math.pi / 2:
            return False
        
        # Check if there's a clear line to enemy (no obstacle blocking)
        if self._line_hits_obstacle(agent["pos"][0], agent["pos"][1], 
                                     enemy["pos"][0], enemy["pos"][1]):
            return False
        
        return True
    
    def _check_clear_shot(self, agent, enemy):
        """Check if a hitscan shot would hit the enemy right now.
        
        Returns True if enemy is in front, within hit radius, and no obstacle blocking.
        """
        dx = math.cos(agent["angle"])
        dy = math.sin(agent["angle"])
        
        to_enemy = enemy["pos"] - agent["pos"]
        
        # Project onto shooting direction
        proj = to_enemy[0] * dx + to_enemy[1] * dy
        
        if proj <= 0:  # Enemy behind us
            return False
        
        # Perpendicular distance from shot line
        closest = np.array([dx * proj, dy * proj])
        perp_dist = np.linalg.norm(to_enemy - closest)
        
        if perp_dist >= 1.5:  # Outside hit radius
            return False
        
        # Check if obstacle blocks
        if self._line_hits_obstacle(agent["pos"][0], agent["pos"][1],
                                     enemy["pos"][0], enemy["pos"][1]):
            return False
        
        return True
    
    def _shoot_direction(self, agent, dx, dy):
        """Shoot in a 2D direction (dx, dy). Applies damage, tracks hits."""
        hit = False
        shooter_id = agent["id"]
        
        origin = agent["pos"].copy()
        
        # Check all enemies along the line
        for other in self.agents:
            if other["team"] == agent["team"] or not other["alive"]:
                continue
            
            # Vector to other agent (2D)
            to_other = other["pos"] - origin
            
            # Project onto direction
            if dx != 0 or dy != 0:
                # Distance along shoot direction
                proj = to_other[0] * dx + to_other[1] * dy
                
                if proj > 0:  # In front of us
                    # Perpendicular distance
                    closest = np.array([dx * proj, dy * proj])
                    perp_dist = np.linalg.norm(to_other - closest)
                    
                    if perp_dist < 1.5:  # Hit radius
                        # Check if obstacle blocks the shot
                        if self._line_hits_obstacle(origin[0], origin[1], other["pos"][0], other["pos"][1]):
                            continue  # Shot blocked by obstacle
                        
                        damage = 25
                        other["health"] -= damage
                        hit = True
                        
                        # Track step feedback
                        self.was_hit[other["id"]] = 1.0
                        self.hit_enemy[shooter_id] = 1.0
                        
                        if other["health"] <= 0:
                            other["alive"] = False
        
        # Record shot for rendering
        self.last_shots.append({
            "origin": origin,
            "dx": dx, "dy": dy,
            "hit": hit,
            "team": agent["team"]
        })

    def _point_in_obstacle(self, x, y):
        """Check if point is inside the obstacle."""
        if self.OBSTACLE is None:
            return False
        ox_min, oy_min, ox_max, oy_max = self.OBSTACLE
        return ox_min <= x <= ox_max and oy_min <= y <= oy_max
    
    def _line_hits_obstacle(self, x1, y1, x2, y2):
        """Check if line segment from (x1,y1) to (x2,y2) intersects obstacle."""
        return line_hits_obstacle(x1, y1, x2, y2, self.OBSTACLE)
    
    def _handle_collision(self, agent):
        """Bounds check and obstacle collision. Returns True if collision occurred."""
        collided = False
        
        # Bounds check
        old_x, old_y = agent["pos"][0], agent["pos"][1]
        agent["pos"][0] = np.clip(agent["pos"][0], 0, self.SIZE-1)
        agent["pos"][1] = np.clip(agent["pos"][1], 0, self.SIZE-1)
        if agent["pos"][0] != old_x or agent["pos"][1] != old_y:
            collided = True
        
        # Push out of obstacle if inside
        if self._point_in_obstacle(agent["pos"][0], agent["pos"][1]):
            ox_min, oy_min, ox_max, oy_max = self.OBSTACLE
            cx, cy = (ox_min + ox_max) / 2, (oy_min + oy_max) / 2
            # Push away from center
            dx = agent["pos"][0] - cx
            dy = agent["pos"][1] - cy
            if abs(dx) > abs(dy):
                agent["pos"][0] = ox_max + 0.1 if dx > 0 else ox_min - 0.1
            else:
                agent["pos"][1] = oy_max + 0.1 if dy > 0 else oy_min - 0.1
            collided = True
        
        return collided

    def render(self):
        if self.render_mode in ["human", "rgb_array"]:
            import matplotlib.pyplot as plt
            
            # Initialize plot if needed
            if not hasattr(self, 'fig'):
                self.fig = plt.figure(figsize=(8, 6)) # Smaller size for faster rendering
                self.ax = self.fig.add_subplot(111)
                if self.render_mode == "human":
                    plt.ion()
                    self.fig.show()

            self.ax.clear()
            self.ax.set_xlim(0, self.SIZE)
            self.ax.set_ylim(0, self.SIZE)
            self.ax.set_title("Worms 2D Gym")

            # 2. Render Agents
            team_colors = ['red', 'blue', 'green']
            for agent in self.agents:
                if agent['alive']:
                    pos = agent['pos']
                    color = team_colors[agent['team'] % len(team_colors)]
                    self.ax.scatter(pos[0], pos[1], c=color, marker='o', s=100, label=f"Team {agent['team']}")
                    
            if self.render_mode == "human":
                plt.pause(0.01)
            elif self.render_mode == "rgb_array":
                self.fig.canvas.draw()
                buffer = self.fig.canvas.buffer_rgba()
                image = np.asarray(buffer)
                return image[:, :, :3]

    def close(self):
        if hasattr(self, 'fig'):
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            del self.fig
