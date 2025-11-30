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
MAX_COOLDOWN = 5  # steps
MAX_SPEED_FORWARD = 0.3  # units per step
N_RAYS = 8
RAY_MAX_RANGE = 30.0  # max ray distance (matches arena size)
OBS_DIM = 14  # total observation dimensions (simplified)

MAX_PERP_DIST = 0.4 # maximum perpendicular distance to target that would still hit 
# ROTATE_STEP = 0.25  # ~14 degrees per step
ROTATE_STEP = 0.05
AIM_SNAP_THRESHOLD = 0.08  # ~4.5 degrees - if within this, snap to perfect aim


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


def line_hits_obstacle(x1, y1, x2, y2, obstacles):
    """Check if line segment from (x1,y1) to (x2,y2) intersects obstacle(s).
    
    Args:
        x1, y1: Start point
        x2, y2: End point  
        obstacle: Single [x_min, y_min, x_max, y_max] or list of obstacles, or None
    
    Returns:
        True if line intersects any obstacle, False otherwise
    """

    # Check multiple points along the line, but skip endpoints
    # This prevents false positives when shooter/target is near a wall
    steps = 20
    for i in range(1, steps):  # Skip i=0 (start) and i=steps (end)
        t = i / steps
        px = x1 + t * (x2 - x1)
        py = y1 + t * (y2 - y1)
        for obs in obstacles:
            ox_min, oy_min, ox_max, oy_max = obs
            if ox_min <= px <= ox_max and oy_min <= py <= oy_max:
                return True
    return False


def ray_cast(x, y, angle, max_range, arena_size, obstacles):
    """Cast a ray and return distance to nearest hit.
    
    Args:
        x, y: Ray origin
        angle: Ray direction in radians
        max_range: Maximum ray distance
        arena_size: Size of the arena (assumes square [0, arena_size])
        obstacle: Single [x_min, y_min, x_max, y_max] or list of obstacles
    
    Returns:
        Distance to nearest hit, capped at max_range
    """
    dx = math.cos(angle)
    dy = math.sin(angle)
    
    # Normalize to list of obstacles
    if obstacles is None:
        obstacles = []
    
    # Step along ray
    step_size = 0.2
    dist = 0.0
    
    while dist < max_range:
        px = x + dx * dist
        py = y + dy * dist
        
        # Check arena bounds
        if px < 0 or px >= arena_size or py < 0 or py >= arena_size:
            return dist
        
        # Check obstacles
        for obs in obstacles:
            ox_min, oy_min, ox_max, oy_max = obs
            if ox_min <= px <= ox_max and oy_min <= py <= oy_max:
                return dist
        
        dist += step_size
    
    return max_range


# =============================================================================
# Observation Computation
# =============================================================================
def compute_agent_observation(agent, enemy, obstacles, arena_size, 
                               was_hit_last_step=0.0, hit_enemy_last_step=0.0):
    """Compute 14-dimensional egocentric observation vector for a single agent.
    
    Simplified observation layout (14 dims):
        0: cos_delta_enemy - cosine of angle to enemy (1.0 = facing enemy)
        1: sin_delta_enemy - sine of angle to enemy (sign = turn direction)
        2: dist_enemy_norm - normalized distance to enemy
        3: has_los - 1.0 if line of sight to enemy
        4: cooldown_norm - shot cooldown (0 = can shoot)
        5: would_hit - 1.0 if shooting now would hit
        6-13: ray distances (8 rays) - wall/obstacle detection
    
    Args:
        agent: Dict with pos, health, angle, alive, cooldown
        enemy: Dict with same structure as agent
        obstacles: List of [x_min, y_min, x_max, y_max] bounding boxes
        arena_size: Size of the square arena
        was_hit_last_step: (unused, kept for API compatibility)
        hit_enemy_last_step: (unused, kept for API compatibility)
    
    Returns:
        np.array of 14 floats
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    
    # Get agent state
    x_self, y_self = agent["pos"][0], agent["pos"][1]
    theta_self = agent["angle"]
    cooldown = agent.get("cooldown", 0)
    
    # Max enemy distance (arena diagonal)
    max_enemy_dist = arena_size * math.sqrt(2)
    
    # =========================================================================
    # Enemy info (indices 0-3)
    # =========================================================================
    if enemy["alive"]:
        dx = enemy["pos"][0] - x_self
        dy = enemy["pos"][1] - y_self
        dist_enemy = math.sqrt(dx * dx + dy * dy)
        
        # Line of sight check
        has_los = not line_hits_obstacle(x_self, y_self, enemy["pos"][0], enemy["pos"][1], obstacles)
        obs[3] = 1.0 if has_los else 0.0  # has_los
        
        if has_los:
            if dist_enemy > 0.001:
                angle_to_enemy = math.atan2(dy, dx)
                delta_theta = angle_to_enemy - theta_self
                obs[0] = math.cos(delta_theta)  # cos_delta_enemy
                obs[1] = math.sin(delta_theta)  # sin_delta_enemy
            else:
                obs[0] = 1.0
                obs[1] = 0.0
            obs[2] = np.clip(dist_enemy / max_enemy_dist, 0.0, 1.0)  # dist_enemy_norm
        else:
            obs[0] = 0.0
            obs[1] = 0.0
            obs[2] = 1.0  # Unknown distance
    else:
        obs[0] = 0.0
        obs[1] = 0.0
        obs[2] = 1.0
        obs[3] = 0.0
    
    # =========================================================================
    # Self state (indices 4-5)
    # =========================================================================
    obs[4] = np.clip(cooldown / MAX_COOLDOWN, 0.0, 1.0)  # cooldown_norm
    
    # Would-hit indicator
    if enemy["alive"]:
        dx = math.cos(theta_self)
        dy = math.sin(theta_self)
        to_enemy_x = enemy["pos"][0] - x_self
        to_enemy_y = enemy["pos"][1] - y_self
        proj = to_enemy_x * dx + to_enemy_y * dy
        
        if proj > 0:
            closest_x = dx * proj
            closest_y = dy * proj
            perp_dist = math.sqrt((to_enemy_x - closest_x)**2 + (to_enemy_y - closest_y)**2)
            if perp_dist < MAX_PERP_DIST:
                if not line_hits_obstacle(x_self, y_self, enemy["pos"][0], enemy["pos"][1], obstacles):
                    obs[5] = 1.0  # Would hit!
    
    # =========================================================================
    # Ray sensors - wall/obstacle distance (indices 6-13)
    # =========================================================================
    for i in range(N_RAYS):
        t = i / (N_RAYS - 1) if N_RAYS > 1 else 0.5
        local_angle = -math.pi / 2 + t * math.pi
        global_angle = theta_self + local_angle
        
        dist = ray_cast(x_self, y_self, global_angle, RAY_MAX_RANGE, arena_size, obstacles)
        obs[6 + i] = dist / RAY_MAX_RANGE
    
    return obs


class Worms3DEnv(gym.Env):
    """Simple 2D grid combat environment."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Grid size
    SIZE = 30
    
    # Obstacle configurations for curriculum
    s = 2.5
    OBSTACLES_PHASE2 = [
        [s*5, s*5, s*7, s*7]  # Center obstacle
    ]
    OBSTACLES_PHASE1 = []  # No obstacles
    
    # Current phase (0 = phase 1, 1 = phase 2)
    curriculum_phase = 0
    
    @property
    def OBSTACLES(self):
        """Return obstacles based on current curriculum phase."""
        if self.curriculum_phase >= 1:
            return self.OBSTACLES_PHASE2
        return self.OBSTACLES_PHASE1
        
    # Actions: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_left, 6=rotate_right, 7=shoot
    N_ACTIONS = 8
    
    def __init__(self, render_mode=None, curriculum_phase=0):
        super().__init__()
        self.render_mode = render_mode
        self.curriculum_phase = curriculum_phase
        
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
        
        # Spawn agents at fixed X positions, random Y within center half
        margin = 5
        y_center = self.SIZE / 2
        y_range = self.SIZE / 4  # ±quarter of arena from center
        
        #y0 = self.np_random.uniform(y_center - y_range, y_center + y_range)
        #y1 = self.np_random.uniform(y_center - y_range, y_center + y_range)
        y0 = y_center
        y1 = y_center
        
        # Fixed X positions: left and right
        pos0 = np.array([margin, y0])
        pos1 = np.array([self.SIZE - margin, y1])
        
        # Base angles facing away from each other, with small random offset ±30°
        base_angle0 = 0
        base_angle1 = math.pi
        
        rotation_offset_0 = self.np_random.uniform(-math.pi/6, math.pi/6)  # ±30°
        rotation_offset_1 = self.np_random.uniform(-math.pi/6, math.pi/6)  # ±30°
        
        angle0 = base_angle0 + rotation_offset_0
        angle1 = base_angle1 + rotation_offset_1
        
        self.agents = [
            {
                "id": 0, "team": 0, 
                "pos": pos0, 
                "health": MAX_HEALTH,
                "angle": angle0,
                "alive": True,
                "velocity": np.array([0.0, 0.0]),
                "cooldown": 0
            },
            {
                "id": 1, "team": 1, 
                "pos": pos1, 
                "health": MAX_HEALTH,
                "angle": angle1,
                "alive": True,
                "velocity": np.array([0.0, 0.0]),
                "cooldown": 0
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
            # If close enough to aiming at enemy, snap to perfect aim
            elif act == 5 or act == 6:
                if enemy["alive"]:
                    dx_to_enemy = enemy["pos"][0] - agent["pos"][0]
                    dy_to_enemy = enemy["pos"][1] - agent["pos"][1]
                    angle_to_enemy = math.atan2(dy_to_enemy, dx_to_enemy)
                    delta_angle = angle_to_enemy - agent["angle"]
                    # Normalize to [-π, π]
                    while delta_angle > math.pi:
                        delta_angle -= 2 * math.pi
                    while delta_angle < -math.pi:
                        delta_angle += 2 * math.pi
                    
                    if abs(delta_angle) < AIM_SNAP_THRESHOLD:
                        # Snap to perfect aim
                        agent["angle"] = angle_to_enemy
                    elif act == 5:
                        agent["angle"] += ROTATE_STEP
                    else:
                        agent["angle"] -= ROTATE_STEP
                elif act == 5:
                    agent["angle"] += ROTATE_STEP
                else:
                    agent["angle"] -= ROTATE_STEP
            
            # Shoot in facing direction (7)
            elif act == 7:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                hit = self._shoot_direction(agent, dx, dy)
                if not hit:
                    rewards[i] -= 2.0  # Miss penalty to discourage spray-and-pray
            
            # Collision / Boundary Check
            self._handle_collision(agent)
            
            # =================================================================
            # REWARD SYSTEM: Combat-focused with time pressure
            # =================================================================
            
            # Time penalty - encourage faster kills
            rewards[i] -= 0.1
            
            # Aiming reward: bonus for facing enemy with LOS
            if enemy["alive"]:
                dx_to_enemy = enemy["pos"][0] - agent["pos"][0]
                dy_to_enemy = enemy["pos"][1] - agent["pos"][1]
                dist_to_enemy = math.sqrt(dx_to_enemy**2 + dy_to_enemy**2)
                if dist_to_enemy > 0.1:
                    angle_to_enemy = math.atan2(dy_to_enemy, dx_to_enemy)
                    delta_angle = abs(angle_to_enemy - agent["angle"])
                    # Normalize to [0, pi]
                    while delta_angle > math.pi:
                        delta_angle = abs(delta_angle - 2 * math.pi)
                    # Reward for aiming within ~10 degrees AND having LOS
                    if delta_angle < 0.17:  # ~10 degrees
                        has_los = not self._line_hits_obstacle(
                            agent["pos"][0], agent["pos"][1],
                            enemy["pos"][0], enemy["pos"][1]
                        )
                        if has_los:
                            rewards[i] += 0.3  # Aiming bonus
        
        # Damage rewards (computed after all actions processed)
        for i, agent in enumerate(self.agents):
            enemy = self.agents[1 - i]
            
            # Reward for dealing damage: +2.0 per damage point (50 per hit)
            enemy_damage = prev_health[1 - i] - enemy["health"]
            if enemy_damage > 0 and self.hit_enemy[i] > 0:
                rewards[i] += enemy_damage * 2  # +50 per hit (25 dmg * 2)
            
            # Kill bonus (moderate, since 4 hits already give +200)
            if prev_health[1 - i] > 0 and not enemy["alive"]:
                rewards[i] += 200
            
            # Death penalty
            if prev_health[i] > 0 and not agent["alive"]:
                rewards[i] -= 400

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
            a0, a1, self.OBSTACLES, self.SIZE,
            was_hit_last_step=self.was_hit[0],
            hit_enemy_last_step=self.hit_enemy[0]
        )
        obs1 = compute_agent_observation(
            a1, a0, self.OBSTACLES, self.SIZE,
            was_hit_last_step=self.was_hit[1],
            hit_enemy_last_step=self.hit_enemy[1]
        )
        return np.concatenate([obs0, obs1])
    
    def _check_aiming(self, agent, enemy):
        """Check if agent is aiming at enemy. Returns 1.0 if cos_delta_enemy > 0.9 and has_los."""
        obs = compute_agent_observation(agent, enemy, self.OBSTACLES, self.SIZE)
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
            to_other = np.array([other["pos"][0] - origin[0], other["pos"][1] - origin[1]])
            
            # Project onto direction
            if dx != 0 or dy != 0:
                # Ensure direction is normalized (should already be from cos/sin)
                dir_len = math.sqrt(dx*dx + dy*dy)
                dx_norm = dx / dir_len
                dy_norm = dy / dir_len
                
                # Distance along shoot direction
                proj = to_other[0] * dx_norm + to_other[1] * dy_norm
                
                # Debug output
                closest = np.array([dx_norm * proj, dy_norm * proj])
                perp_dist = np.linalg.norm(to_other - closest)
                blocked = self._line_hits_obstacle(origin[0], origin[1], other["pos"][0], other["pos"][1])
                
                # More detailed debug
                if False:
                    print(f"SHOT DEBUG:")
                    print(f"  shooter pos: ({origin[0]:.2f}, {origin[1]:.2f})")
                    print(f"  target pos:  ({other['pos'][0]:.2f}, {other['pos'][1]:.2f})")
                    print(f"  direction:   ({dx:.3f}, {dy:.3f})")
                    print(f"  to_other:    ({to_other[0]:.2f}, {to_other[1]:.2f})")
                    print(f"  proj (dot):  {proj:.2f}")
                    print(f"  closest pt:  ({closest[0]:.2f}, {closest[1]:.2f})")
                    print(f"  perp_dist:   {perp_dist:.2f}")
                    print(f"  blocked:     {blocked}")
                
                if proj > 0:  # In front of us
                    if perp_dist < MAX_PERP_DIST:  # Hit radius (tighter for precision)
                        # Check if obstacle blocks the shot
                        if blocked:
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
        
        return hit

    def _point_in_obstacle(self, x, y):
        """Check if point is inside any obstacle."""
        return point_in_obstacles(x, y, self.OBSTACLES)
    
    def _line_hits_obstacle(self, x1, y1, x2, y2):
        """Check if line segment from (x1,y1) to (x2,y2) intersects obstacle."""
        return line_hits_obstacle(x1, y1, x2, y2, self.OBSTACLES)
    
    def _handle_collision(self, agent):
        """Bounds check and obstacle collision. Returns True if collision occurred."""
        collided = False
        
        # Bounds check
        old_x, old_y = agent["pos"][0], agent["pos"][1]
        agent["pos"][0] = np.clip(agent["pos"][0], 0, self.SIZE-1)
        agent["pos"][1] = np.clip(agent["pos"][1], 0, self.SIZE-1)
        if agent["pos"][0] != old_x or agent["pos"][1] != old_y:
            collided = True
        
        # Push out of obstacles if inside any
        for obs in self.OBSTACLES:
            ox_min, oy_min, ox_max, oy_max = obs
            if ox_min <= agent["pos"][0] <= ox_max and oy_min <= agent["pos"][1] <= oy_max:
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
