"""Pygame renderer for multi-agent (PettingZoo) Worms3D environment.

Loads two independent PPO models and visualizes their gameplay.
"""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import math
import argparse

from stable_baselines3 import PPO
from worms_3d_gym.envs import Worms3DPettingZooEnv


def find_multiagent_models(models_dir="models"):
    """Find the newest multiagent run directory with both agent models."""
    # Look for multiagent run directories
    pattern = os.path.join(models_dir, "population_*")
    run_dirs = glob.glob(pattern)
    
    if not run_dirs:
        raise FileNotFoundError(f"No multiagent runs found in {models_dir}")
    
    # Sort by modification time, newest first
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    
    for run_dir in run_dirs:
        # Look for agent models (prefer final, fallback to latest checkpoint)
        agent0_final = os.path.join(run_dir, "agent_0_final.zip")
        agent1_final = os.path.join(run_dir, "agent_1_final.zip")
        
        if os.path.exists(agent0_final) and os.path.exists(agent1_final):
            print(f"Using final models from: {run_dir}")
            return agent0_final, agent1_final
        
        # Try checkpoints
        agent0_checkpoints = sorted(glob.glob(os.path.join(run_dir, "agent0_round*.zip")))
        agent1_checkpoints = sorted(glob.glob(os.path.join(run_dir, "agent1_round*.zip")))
        
        if agent0_checkpoints and agent1_checkpoints:
            print(f"Using checkpoint models from: {run_dir}")
            return agent0_checkpoints[-1], agent1_checkpoints[-1]
    
    raise FileNotFoundError("No complete multiagent model pairs found")


# Colors
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
TEAM_COLORS = [
    (255, 80, 80),   # Red team (agent 0)
    (80, 80, 255),   # Blue team (agent 1)
]


class MultiAgentRenderer:
    """Pygame renderer for multi-agent Worms3D."""
    
    def __init__(self, map_size, scale=20, show_obs=True, obstacles=None):
        self.map_size = map_size
        self.scale = scale
        self.show_obs = show_obs
        self.obstacles = obstacles or []
        
        self.game_width = map_size * scale
        self.game_height = map_size * scale
        
        self.obs_panel_width = 500 if show_obs else 0
        self.screen_width = self.game_width + self.obs_panel_width
        self.screen_height = self.game_height
        
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.initialized = False
        
        # Per-agent observations
        self.agent_obs = {"agent_0": None, "agent_1": None}
        
        # History for replay
        self.history = []
    
    def init(self):
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Worms 3D - Multi-Agent")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.initialized = True
    
    def set_observations(self, obs_dict):
        """Set per-agent observations."""
        self.agent_obs = obs_dict.copy()
    
    def world_to_screen(self, x, y):
        sx = int(x * self.scale)
        sy = int((self.map_size - 1 - y) * self.scale)
        return sx, sy
    
    def record_state(self, agents, shots=None, obs_dict=None):
        """Record state for replay."""
        agent_snapshot = []
        for a in agents:
            agent_snapshot.append({
                "pos": a["pos"].copy(),
                "health": a["health"],
                "team": a["team"],
                "angle": a.get("angle", 0),
                "alive": a["alive"]
            })
        
        shot_snapshot = []
        if shots:
            for s in shots:
                shot_snapshot.append({
                    "origin": s["origin"].copy(),
                    "dx": s["dx"], "dy": s["dy"],
                    "hit": s["hit"],
                    "team": s["team"]
                })
        
        self.history.append({
            "agents": agent_snapshot,
            "shots": shot_snapshot,
            "obs": {k: v.copy() if v is not None else None for k, v in obs_dict.items()} if obs_dict else None
        })
    
    def clear_history(self):
        self.history = []
    
    def render_frame(self, agents, shots=None, step=None, rewards=None):
        """Render a single frame."""
        self.init()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Background
        self.screen.fill((210, 180, 140))
        
        # Grid
        for x in range(0, self.game_width, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.game_height), 1)
        for y in range(0, self.game_height, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.game_width, y), 1)
        
        # Obstacles
        for obs in self.obstacles:
            ox1, oy1 = self.world_to_screen(obs[0], obs[3])
            ox2, oy2 = self.world_to_screen(obs[2], obs[1])
            rect = pygame.Rect(ox1, oy1, ox2 - ox1, oy2 - oy1)
            pygame.draw.rect(self.screen, (101, 67, 33), rect)
            pygame.draw.rect(self.screen, BLACK, rect, 3)
        
        # Shots
        if shots:
            for shot in shots:
                ox, oy = self.world_to_screen(shot["origin"][0], shot["origin"][1])
                dx, dy = shot["dx"], -shot["dy"]
                end_x = ox + dx * self.screen_width
                end_y = oy + dy * self.screen_height
                
                color = (255, 255, 0) if shot["hit"] else TEAM_COLORS[shot["team"]]
                width = 3 if shot["hit"] else 2
                pygame.draw.line(self.screen, color, (ox, oy), (int(end_x), int(end_y)), width)
        
        # Agents
        for agent in agents:
            if not agent["alive"]:
                continue
            
            sx, sy = self.world_to_screen(agent["pos"][0], agent["pos"][1])
            team_color = TEAM_COLORS[agent["team"]]
            
            # Body
            radius = int(self.scale * 0.4)
            pygame.draw.circle(self.screen, team_color, (sx, sy), radius)
            pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
            
            # Facing direction
            angle = agent.get("angle", 0)
            dir_len = self.scale * 0.6
            end_x = sx + math.cos(angle) * dir_len
            end_y = sy - math.sin(angle) * dir_len
            pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 3)
            
            # Label
            label = self.font.render(f"A{agent['team']}", True, BLACK)
            self.screen.blit(label, (sx - 8, sy + radius + 2))
            
            # Health bar
            health_pct = agent["health"] / 100.0
            bar_width = self.scale
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 8
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, 4))
            health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), 4))
        
        # Step counter and rewards
        y_text = 10
        if step is not None:
            text = self.font.render(f"Step: {step}", True, BLACK)
            self.screen.blit(text, (10, y_text))
            y_text += 20
        
        if rewards:
            for agent_id, rew in rewards.items():
                color = TEAM_COLORS[0] if agent_id == "agent_0" else TEAM_COLORS[1]
                text = self.font.render(f"{agent_id}: {rew:.1f}", True, color)
                self.screen.blit(text, (10, y_text))
                y_text += 18
        
        # Observation panel
        if self.show_obs:
            self._render_obs_panel()
        
        pygame.display.flip()
        return True
    
    def _render_obs_panel(self):
        """Render observation panel for both agents.
        
        Current observation layout (14 dims per agent):
            0: cos_delta_enemy - cosine of angle to enemy
            1: sin_delta_enemy - sine of angle to enemy  
            2: dist_enemy_norm - normalized distance to enemy
            3: has_los - line of sight to enemy
            4: cooldown_norm - shot cooldown
            5: would_hit - 1.0 if shooting now would hit
            6-13: ray distances (8 rays)
        """
        panel_x = self.game_width
        pygame.draw.rect(self.screen, (40, 40, 50), 
                        (panel_x, 0, self.obs_panel_width, self.screen_height))
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (panel_x, 0), (panel_x, self.screen_height), 2)
        
        y_start = 10
        title = self.font.render("OBSERVATIONS", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, y_start))
        y_start += 30
        
        col_width = 240
        for idx, agent_id in enumerate(["agent_0", "agent_1"]):
            obs = self.agent_obs.get(agent_id)
            if obs is None:
                continue
            
            col_x = panel_x + idx * col_width
            y = y_start
            
            color = TEAM_COLORS[idx]
            header = self.font.render(agent_id, True, color)
            self.screen.blit(header, (col_x + 10, y))
            y += 22
            
            # Enemy info (0-3)
            y = self._draw_section(col_x, y, "Enemy", [
                ("cos Δ", obs[0]), ("sin Δ", obs[1]), 
                ("dist", obs[2]), ("LOS", obs[3]),
            ])
            
            # Self state (4-5)
            y = self._draw_section(col_x, y, "Self", [
                ("cooldown", obs[4]), ("would_hit", obs[5]),
            ])
            
            # Ray viz (6-13)
            y = self._draw_ray_viz(col_x, y, obs[6:14], obs[0], obs[1])
    
    def _draw_section(self, panel_x, y, title, items):
        text = self.small_font.render(title, True, (180, 180, 180))
        self.screen.blit(text, (panel_x + 15, y))
        y += 16
        
        for name, value in items:
            label = self.small_font.render(f"{name}:", True, (150, 150, 150))
            self.screen.blit(label, (panel_x + 15, y))
            
            bar_x = panel_x + 70
            bar_width = 60
            bar_height = 10
            
            pygame.draw.rect(self.screen, (60, 60, 70), (bar_x, y + 2, bar_width, bar_height))
            
            if value >= 0:
                fill = int(bar_width * min(value, 1.0))
                color = (80, 200, 80) if value < 0.9 else (255, 255, 80)
                pygame.draw.rect(self.screen, color, (bar_x, y + 2, fill, bar_height))
            else:
                fill = int(bar_width * min(abs(value), 1.0))
                pygame.draw.rect(self.screen, (200, 80, 80), (bar_x + bar_width - fill, y + 2, fill, bar_height))
            
            val_text = self.small_font.render(f"{value:.2f}", True, (200, 200, 200))
            self.screen.blit(val_text, (bar_x + bar_width + 3, y))
            y += 14
        
        return y + 2
    
    def _draw_ray_viz(self, panel_x, y, rays, cos_delta, sin_delta):
        """Draw ray sensor visualization.
        
        Args:
            rays: 8 ray distances (normalized 0-1)
            cos_delta, sin_delta: angle to enemy (for heading indicator)
        """
        text = self.small_font.render("Rays (wall dist)", True, (180, 180, 180))
        self.screen.blit(text, (panel_x + 15, y))
        y += 18
        
        cx, cy = panel_x + 60, y + 35
        radius = 30
        
        pygame.draw.circle(self.screen, (60, 60, 70), (cx, cy), radius)
        pygame.draw.circle(self.screen, (100, 100, 110), (cx, cy), radius, 1)
        
        n_rays = len(rays)
        for i, ray_dist in enumerate(rays):
            # Rays span -90° to +90° relative to heading
            rel_angle = -math.pi/2 + (i / (n_rays - 1)) * math.pi if n_rays > 1 else 0
            
            ray_len = radius * ray_dist
            end_x = cx + math.cos(rel_angle) * ray_len
            end_y = cy - math.sin(rel_angle) * ray_len
            
            # Color: red=close, green=far
            color = (int(255 * (1 - ray_dist)), int(255 * ray_dist), 50)
            
            pygame.draw.line(self.screen, color, (cx, cy), (int(end_x), int(end_y)), 2)
            pygame.draw.circle(self.screen, color, (int(end_x), int(end_y)), 3)
        
        # Enemy direction indicator (cyan arrow)
        if abs(cos_delta) > 0.01 or abs(sin_delta) > 0.01:
            enemy_angle = math.atan2(sin_delta, cos_delta)
            enemy_x = cx + math.cos(enemy_angle) * (radius + 8)
            enemy_y = cy - math.sin(enemy_angle) * (radius + 8)
            pygame.draw.line(self.screen, (0, 255, 255), (cx, cy), (int(enemy_x), int(enemy_y)), 2)
            pygame.draw.circle(self.screen, (0, 255, 255), (int(enemy_x), int(enemy_y)), 4)
        
        return y + 75
    
    def play_history(self, fps=5):
        self.init()
        
        for i, state in enumerate(self.history):
            if state.get("obs"):
                self.agent_obs = state["obs"]
            if not self.render_frame(state["agents"], state.get("shots"), step=i):
                break
            self.clock.tick(fps)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
        
        pygame.quit()
        self.initialized = False


def run_multiagent_renderer(model_path_0=None, model_path_1=None, n_episodes=3):
    """Run multi-agent environment with pygame rendering."""

    if model_path_0 is None or model_path_1 is None:
        model_path_0, model_path_1 = find_multiagent_models()
    
    print(f"Loading agent_0 from: {model_path_0}")
    print(f"Loading agent_1 from: {model_path_1}")
    model_0 = PPO.load(model_path_0)
    model_1 = PPO.load(model_path_1)
    
    # Create environment
    env = Worms3DPettingZooEnv(render_mode=None,curriculum_phase=1)
    unwrapped = env.unwrapped
    
    # Create renderer
    renderer = MultiAgentRenderer(
        map_size=unwrapped.SIZE,
        scale=20,
        show_obs=True,
        obstacles=unwrapped.OBSTACLES
    )
    
    action_names = ["nothing", "up", "down", "left", "right", "rot_L", "rot_R", "shoot"]
    
    episodes_shown = 0
    episodes_tried = 0
    
    while episodes_shown < n_episodes:
        episodes_tried += 1
        print(f"\n=== Searching for kill episode (attempt {episodes_tried}, shown {episodes_shown}/{n_episodes}) ===")
        
        # Clear history for new episode
        renderer.history = []
        
        obs, info = env.reset()
        renderer.set_observations(obs)
        renderer.record_state(unwrapped.agents, unwrapped.last_shots, obs)
        
        total_rewards = {"agent_0": 0, "agent_1": 0}
        step = 0
        done = False
        
        while not done:
            # Get actions
            if model_0 is not None:
                action_0, _ = model_0.predict(obs["agent_0"], deterministic=True)
            else:
                action_0 = env.action_space("agent_0").sample()
            
            if model_1 is not None:
                action_1, _ = model_1.predict(obs["agent_1"], deterministic=True)
            else:
                action_1 = env.action_space("agent_1").sample()
            
            actions = {"agent_0": int(action_0), "agent_1": int(action_1)}
            
            # Step
            obs, rewards, terminations, truncations, infos = env.step(actions)
            
            total_rewards["agent_0"] += rewards["agent_0"]
            total_rewards["agent_1"] += rewards["agent_1"]
            
            renderer.set_observations(obs)
            renderer.record_state(unwrapped.agents, unwrapped.last_shots, obs)
            
            step += 1
            done = terminations["agent_0"] or truncations["agent_0"]
        
        # Check if there was exactly 1 kill (not a draw)
        a0_alive = unwrapped.agents[0]["alive"]
        a1_alive = unwrapped.agents[1]["alive"]
        has_kill = (a0_alive and not a1_alive) or (a1_alive and not a0_alive)
        
        if has_kill:
            winner = "agent_0" if a0_alive else "agent_1"
            print(f"KILL FOUND! Winner: {winner} | Steps: {step} | Total rewards: {total_rewards}")
            print(f"\nRecorded {len(renderer.history)} frames. Playing back...")
            renderer.play_history(fps=5)
            episodes_shown += 1
        else:
            print(f"Draw/timeout after {step} steps, skipping...")
    
    env.close()
    print(f"\nDone! Showed {episodes_shown} kill episodes out of {episodes_tried} attempts.")


if __name__ == "__main__":
    en = 10
    run_multiagent_renderer(
        model_path_0=f"models/population_5agents_20251130_213102/agent_2_round{en}.zip",
        model_path_1=f"models/population_5agents_20251130_213102/agent_3_round{en}.zip",
        n_episodes=1,
    )
