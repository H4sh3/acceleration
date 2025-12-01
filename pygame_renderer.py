import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import pygame
import numpy as np
import math


def get_newest_model(models_dir="models"):
    """Find the newest .zip model file in the models directory."""
    pattern = os.path.join(models_dir, "**", "*.zip")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        raise FileNotFoundError(f"No .zip model files found in {models_dir}")
    
    # Sort by modification time, newest first
    newest = max(model_files, key=os.path.getmtime)
    print(f"Using newest model: {newest}")
    return newest


MODEL_PATH = get_newest_model()

# Colors
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
TEAM_COLORS = [
    (255, 80, 80),   # Red team
    (80, 80, 255),   # Blue team
    (80, 255, 80),   # Green team
]

class PygameRenderer:
    def __init__(self, map_width, map_depth, scale=30, show_obs=True, obstacles=None):
        """
        Top-down 2D renderer for Worms 3D environment.
        
        Args:
            map_width: Width of the map in world units
            map_depth: Depth of the map in world units  
            scale: Pixels per world unit
            show_obs: Whether to show observation panel
            obstacle: [x_min, y_min, x_max, y_max] obstacle bounds
        """
        self.map_width = map_width
        self.map_depth = map_depth
        self.scale = scale
        self.show_obs = show_obs
        self.obstacles = obstacles
        
        self.game_width = map_width * scale
        self.game_height = map_depth * scale
        
        # Add observation panel width if enabled (wider for side-by-side agents)
        self.obs_panel_width = 500 if show_obs else 0
        self.screen_width = self.game_width + self.obs_panel_width
        self.screen_height = self.game_height
        
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.initialized = False
        
        # Current observation for visualization
        self.current_obs = None
        
        # State history for replay
        self.history = []
        
    def init(self):
        """Initialize pygame display."""
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Worms 3D - Top Down View")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.initialized = True
    
    def set_observation(self, obs):
        """Set current observation for visualization."""
        self.current_obs = obs
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        sx = int(x * self.scale)
        sy = int((self.map_depth - 1 - y) * self.scale)  # Flip Y axis
        return sx, sy
    
    def record_state(self, agents, shots=None, obs=None):
        """Record current state for replay."""
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
            "obs": obs.copy() if obs is not None else None
        })
    
    def clear_history(self):
        """Clear recorded history."""
        self.history = []
    
    def render_frame(self, agents, shots=None, step=None):
        """Render a single frame."""
        self.init()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen - tan background
        self.screen.fill((210, 180, 140))
        
        # Draw grid lines
        for x in range(0, self.screen_width, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.screen_height), 1)
        for y in range(0, self.screen_height, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.screen_width, y), 1)

        # Draw obstacles (brown boxes)
        for obs in self.obstacles:
            ox1, oy1 = self.world_to_screen(obs[0], obs[3])  # Top-left in screen
            ox2, oy2 = self.world_to_screen(obs[2], obs[1])  # Bottom-right in screen
            obstacle_rect = pygame.Rect(ox1, oy1, ox2 - ox1, oy2 - oy1)
            pygame.draw.rect(self.screen, (101, 67, 33), obstacle_rect)  # Dark brown fill
            pygame.draw.rect(self.screen, BLACK, obstacle_rect, 3)  # Black border
        
        # Draw shots (laser beams)
        if shots:
            for shot in shots:
                ox, oy = self.world_to_screen(shot["origin"][0], shot["origin"][1])
                # Extend line to edge of screen
                dx, dy = shot["dx"], -shot["dy"]  # Flip dy for screen coords
                end_x = ox + dx * self.screen_width
                end_y = oy + dy * self.screen_height
                
                # Color based on team and hit
                if shot["hit"]:
                    color = (255, 255, 0)  # Yellow for hit
                    width = 3
                else:
                    color = TEAM_COLORS[shot["team"]]
                    width = 2
                
                pygame.draw.line(self.screen, color, (ox, oy), (int(end_x), int(end_y)), width)
        
        # Draw agents
        for agent in agents:
            if not agent["alive"]:
                continue
                
            pos = agent["pos"]
            sx, sy = self.world_to_screen(pos[0], pos[1])
            
            team_color = TEAM_COLORS[agent["team"] % len(TEAM_COLORS)]
            
            # Draw agent body (circle)
            radius = int(self.scale * 0.4)
            pygame.draw.circle(self.screen, team_color, (sx, sy), radius)
            pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
            
            # Draw facing direction (line from center)
            angle = agent.get("angle", 0)
            dir_len = self.scale * 0.6
            end_x = sx + math.cos(angle) * dir_len
            end_y = sy - math.sin(angle) * dir_len  # Flip Y
            pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 3)
            
            # Draw team label
            team_label = self.font.render(f"T{agent['team']}", True, BLACK)
            self.screen.blit(team_label, (sx - 8, sy + radius + 2))
            
            # Draw health bar
            health_pct = agent["health"] / 100.0
            bar_width = self.scale
            bar_height = 4
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 8
            
            # Background
            pygame.draw.rect(self.screen, BLACK, 
                           (bar_x, bar_y, bar_width, bar_height))
            # Health
            health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color,
                           (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        
        # Draw step counter
        if step is not None:
            text = self.font.render(f"Step: {step}", True, BLACK)
            self.screen.blit(text, (10, 10))
        
        # Draw observation panel
        if self.show_obs and self.current_obs is not None:
            self._render_obs_panel()
        
        pygame.display.flip()
        return True
    
    def _render_obs_panel(self):
        """Render the observation visualization panel."""
        if self.current_obs is None:
            return
        
        # Panel background
        panel_x = self.game_width
        pygame.draw.rect(self.screen, (40, 40, 50), 
                        (panel_x, 0, self.obs_panel_width, self.screen_height))
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (panel_x, 0), (panel_x, self.screen_height), 2)
        
        # Get latest frame from stacked observation
        # Frame stacking: 4 frames x 56 dims = 224 total (28 per agent x 2 agents)
        obs = self.current_obs.flatten()
        agent_obs_dim = 28  # Updated observation dimension per agent
        total_obs_dim = agent_obs_dim * 2  # 56 for both agents
        
        if len(obs) >= total_obs_dim * 4:
            # Get last frame (most recent) from stacked obs
            latest_start = 3 * total_obs_dim
            obs = obs[latest_start:latest_start + total_obs_dim]
        elif len(obs) >= total_obs_dim:
            obs = obs[:total_obs_dim]
        
        # Split into agent 0 and agent 1 observations (28 dims each)
        obs0 = obs[:agent_obs_dim] if len(obs) >= agent_obs_dim else obs
        obs1 = obs[agent_obs_dim:agent_obs_dim*2] if len(obs) >= agent_obs_dim*2 else None
        
        y_start = 10
        
        # Title
        title = self.font.render("OBSERVATION", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, y_start))
        y_start += 30
        
        # Render agents side by side
        col_width = 240
        for agent_idx, agent_obs in enumerate([obs0, obs1]):
            if agent_obs is None:
                continue
            
            col_x = panel_x + agent_idx * col_width
            y = y_start
            
            color = TEAM_COLORS[agent_idx]
            header = self.font.render(f"Agent {agent_idx}", True, color)
            self.screen.blit(header, (col_x + 10, y))
            y += 22
            
            # Self state (0-5)
            y = self._draw_obs_section(col_x, y, "Self State", [
                ("cos θ", agent_obs[0]),
                ("sin θ", agent_obs[1]),
                ("v_fwd", agent_obs[2]),
                ("health", agent_obs[3]),
                ("ammo", agent_obs[4]),
                ("cooldown", agent_obs[5]),
            ])
            
            # Enemy info (6-9)
            y = self._draw_obs_section(col_x, y, "Enemy Info", [
                ("cos Δ", agent_obs[6]),
                ("sin Δ", agent_obs[7]),
                ("dist", agent_obs[8]),
                ("LOS", agent_obs[9]),
            ])
            
            # Ray sensors (10-17 wall dist, 20-27 enemy detect) - draw as mini visualization
            enemy_rays = agent_obs[20:28] if len(agent_obs) >= 28 else [0]*8
            y = self._draw_ray_viz(col_x, y, agent_obs[10:18], agent_obs[0], agent_obs[1], enemy_rays)
            
            # Step feedback (18-19)
            y = self._draw_obs_section(col_x, y, "Feedback", [
                ("was_hit", agent_obs[18]),
                ("hit_enemy", agent_obs[19]),
            ])
    
    def _draw_obs_section(self, panel_x, y, title, items):
        """Draw a section of observation values."""
        # Section title
        text = self.small_font.render(title, True, (180, 180, 180))
        self.screen.blit(text, (panel_x + 15, y))
        y += 16
        
        # Draw items as horizontal bars
        bar_width = 60
        bar_height = 10
        
        for name, value in items:
            # Label
            label = self.small_font.render(f"{name}:", True, (150, 150, 150))
            self.screen.blit(label, (panel_x + 15, y))
            
            # Value bar
            bar_x = panel_x + 70
            
            # Background
            pygame.draw.rect(self.screen, (60, 60, 70), 
                           (bar_x, y + 2, bar_width, bar_height))
            
            # Value bar (handle negative values)
            if value >= 0:
                fill_width = int(bar_width * min(value, 1.0))
                fill_color = (80, 200, 80) if value < 0.9 else (255, 255, 80)
                pygame.draw.rect(self.screen, fill_color,
                               (bar_x, y + 2, fill_width, bar_height))
            else:
                fill_width = int(bar_width * min(abs(value), 1.0))
                pygame.draw.rect(self.screen, (200, 80, 80),
                               (bar_x + bar_width - fill_width, y + 2, fill_width, bar_height))
            
            # Numeric value
            val_text = self.small_font.render(f"{value:.2f}", True, (200, 200, 200))
            self.screen.blit(val_text, (bar_x + bar_width + 3, y))
            
            y += 14
        
        return y + 2
    
    def _draw_ray_viz(self, panel_x, y, rays, cos_theta, sin_theta, enemy_rays=None):
        """Draw ray sensor visualization as a mini radar.
        
        Args:
            rays: Wall/obstacle distance for each ray (0-1 normalized)
            enemy_rays: 1.0 if ray detects enemy, 0.0 otherwise
        """
        # Section title
        text = self.small_font.render("Ray Sensors", True, (180, 180, 180))
        self.screen.blit(text, (panel_x + 15, y))
        y += 18
        
        # Draw mini radar
        center_x = panel_x + 60
        center_y = y + 35
        radius = 30
        
        # Background circle
        pygame.draw.circle(self.screen, (60, 60, 70), (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (100, 100, 110), (center_x, center_y), radius, 1)
        
        if enemy_rays is None:
            enemy_rays = [0] * len(rays)
        
        # Draw rays - 8 rays covering ±90° from heading
        n_rays = len(rays)
        for i, ray_dist in enumerate(rays):
            # Ray angle relative to heading: from -90° to +90°
            rel_angle = -math.pi/2 + (i / (n_rays - 1)) * math.pi
            
            # Convert to world angle using agent's heading
            world_angle = math.atan2(sin_theta, cos_theta) + rel_angle
            
            # Ray endpoint (inverted for screen coords)
            ray_len = radius * ray_dist
            end_x = center_x + math.cos(world_angle) * ray_len
            end_y = center_y - math.sin(world_angle) * ray_len
            
            # Color: CYAN if enemy detected, otherwise red/green gradient for distance
            enemy_hit = enemy_rays[i] > 0.5 if i < len(enemy_rays) else False
            if enemy_hit:
                color = (0, 255, 255)  # Cyan = enemy detected
            else:
                # Red = close wall, green = far wall
                r = int(255 * (1 - ray_dist))
                g = int(255 * ray_dist)
                color = (r, g, 50)
            
            pygame.draw.line(self.screen, color, (center_x, center_y), 
                           (int(end_x), int(end_y)), 2)
            
            # Draw endpoint dot (larger if enemy)
            dot_size = 5 if enemy_hit else 3
            pygame.draw.circle(self.screen, color, (int(end_x), int(end_y)), dot_size)
        
        # Draw agent heading indicator
        head_len = radius + 5
        head_x = center_x + cos_theta * head_len
        head_y = center_y - sin_theta * head_len
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, center_y),
                        (int(head_x), int(head_y)), 2)
        
        return y + 75
    
    def play_history(self, fps=10):
        """
        Play back recorded history.
        
        Args:
            fps: Frames per second for playback
        """
        self.init()
        
        for i, state in enumerate(self.history):
            shots = state.get("shots", [])
            # Set observation for panel rendering
            if state.get("obs") is not None:
                self.current_obs = state["obs"]
            if not self.render_frame(state["agents"], shots=shots, step=i):
                break
            self.clock.tick(fps)
        
        # Keep window open until closed
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
        
        pygame.quit()
        self.initialized = False
    
    def save_history_as_images(self, output_dir="frames"):
        """Save each frame as an image."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.init()
        
        for i, state in enumerate(self.history):
            self.render_frame(state["heightmap"], state["agents"], step=i)
            pygame.image.save(self.screen, f"{output_dir}/frame_{i:04d}.png")
        
        print(f"Saved {len(self.history)} frames to {output_dir}/")


def run_with_pygame_renderer():
    """Run the environment with pygame rendering."""
    import gymnasium as gym
    import worms_3d_gym
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    
    # Wrap env with frame stacking to match training setup
    env = DummyVecEnv([lambda: gym.make("Worms3D-v0")])
    env = VecFrameStack(env, n_stack=4)
    unwrapped = env.envs[0].unwrapped
    
    # Create renderer (scale reduced for larger arena)
    renderer = PygameRenderer(
        map_width=unwrapped.SIZE,
        map_depth=unwrapped.SIZE,
        scale=20,
        obstacles=unwrapped.OBSTACLES
    )
    
    # Load newest model
    model = None
    model_path = MODEL_PATH
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path)
    
    # Run simulation
    obs = env.reset()
    renderer.set_observation(obs)
    renderer.record_state(unwrapped.agents, unwrapped.last_shots, obs)
    
    print("Running simulation...")
    total_reward = 0
    episode = 1
    episode_steps = 0
    hits = 0
    
    for step in range(200):
        if model:
            actions, _ = model.predict(obs, deterministic=True)
        else:
            actions = env.action_space.sample()
        
        # actions is shape (1, 2) from vectorized env - flatten it
        actions = actions.flatten()
        
        # Log actions
        action_names = ["nothing", "up", "down", "left", "right", "rot_left", "rot_right", "shoot", "dash"]
        a0_name = action_names[int(actions[0])]
        a1_name = action_names[int(actions[1])]
        
        old_hp = [a["health"] for a in unwrapped.agents]
        
        obs, reward, done, info = env.step([actions])
        terminated = done[0]
        truncated = False
        
        new_hp = [a["health"] for a in unwrapped.agents]
        
        # Check for hits
        for i in range(2):
            if new_hp[i] < old_hp[i]:
                hits += 1
                print(f"  Step {step}: Agent {1-i} HIT Agent {i}! HP: {old_hp[i]:.0f} -> {new_hp[i]:.0f}")
        
        total_reward += reward[0]  # Extract scalar from vectorized env
        episode_steps += 1
        
        renderer.set_observation(obs)
        renderer.record_state(unwrapped.agents, unwrapped.last_shots, obs)
        
        # Log every step
        a0 = unwrapped.agents[0]
        a1 = unwrapped.agents[1]
        # obs is now (1, 160) with frame stacking - 40 dims per frame, 4 frames
        # Latest frame starts at index 120 (3 * 40)
        # Agent 0: indices 0-19, Agent 1: indices 20-39 within each frame
        # cos_delta_enemy at index 6, has_los at index 9
        flat_obs = obs[0]
        latest_frame_start = 3 * 40  # Last of 4 stacked frames
        a0_cos_delta = flat_obs[latest_frame_start + 6]
        a0_has_los = flat_obs[latest_frame_start + 9]
        a1_cos_delta = flat_obs[latest_frame_start + 20 + 6]
        a1_has_los = flat_obs[latest_frame_start + 20 + 9]
        aim0 = "AIM!" if (a0_cos_delta > 0.9 and a0_has_los > 0.5) else "    "
        aim1 = "AIM!" if (a1_cos_delta > 0.9 and a1_has_los > 0.5) else "    "
        print(f"Step {step:3}: A0={a0_name:10} {aim0} | A1={a1_name:10} {aim1} | HP: {new_hp[0]:.0f} vs {new_hp[1]:.0f}")
        
        if terminated or truncated:
            winner = "T0" if unwrapped.agents[0]["alive"] else "T1" if unwrapped.agents[1]["alive"] else "Draw"
            print(f"=== Episode {episode} ended at step {episode_steps}: {winner} wins! Total reward: {total_reward:.1f}, Hits: {hits} ===")
            episode += 1
            total_reward = 0
            episode_steps = 0
            hits = 0
            obs = env.reset()
            renderer.set_observation(obs)
            renderer.record_state(unwrapped.agents, unwrapped.last_shots, obs)

    env.close()
    
    print(f"Recorded {len(renderer.history)} frames. Playing back...")
    renderer.play_history(fps=5)


if __name__ == "__main__":
    run_with_pygame_renderer()
