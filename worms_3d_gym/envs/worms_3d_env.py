"""Minimal 2D Combat Gym Environment.

Two agents on a grid. Move or shoot in 4 directions. First to kill wins.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math


class Worms3DEnv(gym.Env):
    """Simple 2D grid combat environment."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Grid size
    SIZE = 15
    
    # Actions: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_toward, 6=rotate_away, 7=shoot
    N_ACTIONS = 8
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 2 agents
        self.n_agents = 2
        
        # Action: one discrete action per agent
        self.action_space = spaces.MultiDiscrete([self.N_ACTIONS] * self.n_agents)
        
        # Observation per agent: [my_x, my_y, my_health, my_angle, enemy_x, enemy_y, enemy_health, aiming_at_enemy]
        # Total: 8 values per agent = 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random spawn with minimum distance
        positions = []
        for i in range(2):
            while True:
                x = self.np_random.uniform(2, self.SIZE - 2)
                y = self.np_random.uniform(2, self.SIZE - 2)
                # Check distance from other agents
                if all(np.sqrt((x - px)**2 + (y - py)**2) > 5 for px, py in positions):
                    positions.append((x, y))
                    break
        
        self.agents = [
            {"id": 0, "team": 0, "pos": np.array([positions[0][0], positions[0][1]]), "health": 100.0, "angle": 0.0, "alive": True},
            {"id": 1, "team": 1, "pos": np.array([positions[1][0], positions[1][1]]), "health": 100.0, "angle": math.pi, "alive": True},
        ]
        
        # Track shots for rendering
        self.last_shots = []
        
        self.current_step = 0
        return self._get_obs(), {}

    def step(self, actions):
        # actions is array of shape (TOTAL_AGENTS,) - one action per agent
        # Action: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_left, 6=rotate_right, 7=shoot
        
        rewards = np.zeros(self.n_agents)
        self.last_shots = []  # Clear shots from previous step
        
        MOVE_STEP = 0.3
        ROTATE_STEP = 0.3  # ~17 degrees per step
        
        for i, agent in enumerate(self.agents):
            if not agent["alive"]:
                continue
            
            act = actions[i]
            
            # Movement actions (1-4)
            if act == 1:  # Up (Y+)
                agent["pos"][1] += MOVE_STEP
            elif act == 2:  # Down (Y-)
                agent["pos"][1] -= MOVE_STEP
            elif act == 3:  # Left (X-)
                agent["pos"][0] -= MOVE_STEP
            elif act == 4:  # Right (X+)
                agent["pos"][0] += MOVE_STEP
            
            # Rotation actions (5-6) - relative to enemy
            elif act == 5 or act == 6:
                # Find enemy
                enemy = self.agents[1 - i]
                if enemy["alive"]:
                    # Angle to enemy
                    to_enemy = enemy["pos"] - agent["pos"]
                    target_angle = math.atan2(to_enemy[1], to_enemy[0])
                    
                    # Current angle difference
                    diff = target_angle - agent["angle"]
                    # Normalize to [-pi, pi]
                    while diff > math.pi: diff -= 2 * math.pi
                    while diff < -math.pi: diff += 2 * math.pi
                    
                    if act == 5:  # Rotate toward enemy
                        if abs(diff) < ROTATE_STEP:
                            # Snap to perfect aim
                            agent["angle"] = target_angle
                        elif diff > 0:
                            agent["angle"] += ROTATE_STEP
                        else:
                            agent["angle"] -= ROTATE_STEP
                    else:  # act == 6: Rotate away from enemy
                        if diff > 0:
                            agent["angle"] -= ROTATE_STEP
                        else:
                            agent["angle"] += ROTATE_STEP
            
            # Shoot in facing direction (7)
            elif act == 7:
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                reward = self._shoot_direction(agent, dx, dy)
                rewards[i] += reward
            
            # Reward shaping: BIG penalty for not shooting when aiming
            enemy = self.agents[1 - i]
            if enemy["alive"] and self._check_aiming(agent, enemy) > 0.5:
                if act != 7:  # Not shooting while aiming = very bad
                    rewards[i] -= 5.0
            
            # Collision / Boundary Check
            self._handle_collision(agent)

        # Track steps
        self.current_step = getattr(self, 'current_step', 0) + 1
        
        # Check if done (only 1 team left)
        alive_teams = set(a["team"] for a in self.agents if a["alive"])
        terminated = len(alive_teams) <= 1
        truncated = self.current_step >= 200  # Max 200 steps per episode
        
        # Small time penalty to encourage faster kills
        time_penalty = -0.5
        total_reward = float(np.sum(rewards)) + time_penalty
        
        return self._get_obs(), total_reward, terminated, truncated, {"alive_teams": list(alive_teams)}

    def _get_obs(self):
        """Flat obs: [my_x, my_y, my_hp, my_angle, enemy_x, enemy_y, enemy_hp, aiming_at_enemy] per agent."""
        a0, a1 = self.agents[0], self.agents[1]
        
        # Check if each agent is aiming at the other
        aim0 = self._check_aiming(a0, a1)
        aim1 = self._check_aiming(a1, a0)
        
        return np.array([
            # Agent 0's view
            a0["pos"][0], a0["pos"][1], a0["health"], a0["angle"],
            a1["pos"][0], a1["pos"][1], a1["health"], aim0,
            # Agent 1's view  
            a1["pos"][0], a1["pos"][1], a1["health"], a1["angle"],
            a0["pos"][0], a0["pos"][1], a0["health"], aim1,
        ], dtype=np.float32)
    
    def _check_aiming(self, agent, target):
        """Return 1.0 if agent is aiming at target, 0.0 otherwise."""
        if not agent["alive"] or not target["alive"]:
            return 0.0
        
        # Direction agent is facing
        dx = math.cos(agent["angle"])
        dy = math.sin(agent["angle"])
        
        # Vector to target
        to_target = target["pos"] - agent["pos"]
        dist = np.linalg.norm(to_target)
        if dist < 0.1:
            return 0.0
        
        to_target = to_target / dist  # Normalize
        
        # Dot product - how aligned are we?
        dot = dx * to_target[0] + dy * to_target[1]
        
        # If dot > 0.9, we're aiming at them (~25 degree cone)
        return 1.0 if dot > 0.9 else 0.0
    
    def _shoot_direction(self, agent, dx, dy):
        """Shoot in a 2D direction (dx, dy). Returns reward."""
        reward = 0
        hit = False
        
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
                    
                    if perp_dist < 3.0:  # Hit radius (wider beam)
                        damage = 25
                        other["health"] -= damage
                        reward += damage
                        hit = True
                        
                        if other["health"] <= 0:
                            other["alive"] = False
                            reward += 100  # Kill bonus
        
        # Record shot for rendering
        self.last_shots.append({
            "origin": origin,
            "dx": dx, "dy": dy,
            "hit": hit,
            "team": agent["team"]
        })
        
        return reward

    def _handle_collision(self, agent):
        # Simple bounds check
        agent["pos"][0] = np.clip(agent["pos"][0], 0, self.SIZE-1)
        agent["pos"][1] = np.clip(agent["pos"][1], 0, self.SIZE-1)

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
