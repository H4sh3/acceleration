"""Pygame renderer for ZombieSurvivalEnv with model loading."""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import math
import pygame


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


class ZombieRenderer:
    """Pygame-based renderer for the zombie survival environment."""
    
    # Colors
    BG_COLOR = (20, 20, 30)
    GRID_COLOR = (40, 40, 50)
    WALL_COLOR = (80, 80, 90)
    OBSTACLE_COLOR = (101, 67, 33)  # Dark brown like pygame_renderer
    
    PLAYER_COLOR = (50, 200, 100)
    PLAYER_OUTLINE = (30, 150, 70)
    ZOMBIE_COLOR = (200, 60, 60)
    ZOMBIE_OUTLINE = (150, 40, 40)
    
    SHOT_COLOR_HIT = (255, 255, 0)  # Yellow for hit
    SHOT_COLOR_MISS = (200, 200, 200)
    
    HEALTH_BAR_BG = (60, 60, 60)
    HEALTH_BAR_PLAYER = (50, 200, 100)
    HEALTH_BAR_ZOMBIE = (200, 60, 60)
    
    UI_TEXT_COLOR = (220, 220, 220)
    UI_BG_COLOR = (30, 30, 40, 200)
    
    def __init__(self, map_width, map_depth, scale=20, show_obs=True, obstacles=None):
        """
        Top-down 2D renderer for Zombie Survival environment.
        
        Args:
            map_width: Width of the map in world units
            map_depth: Depth of the map in world units  
            scale: Pixels per world unit
            show_obs: Whether to show observation panel
            obstacles: List of [x_min, y_min, x_max, y_max] obstacle bounds
        """
        self.map_width = map_width
        self.map_depth = map_depth
        self.scale = scale
        self.show_obs = show_obs
        self.obstacles = obstacles or []
        
        self.game_width = map_width * scale
        self.game_height = map_depth * scale
        
        # Add observation panel width if enabled
        self.obs_panel_width = 300 if show_obs else 0
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
            pygame.display.set_caption("Zombie Survival")
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
    
    def record_state(self, agent, zombies, shots=None, obs=None, kills=0):
        """Record current state for replay."""
        agent_snapshot = {
            "pos": agent["pos"].copy(),
            "health": agent["health"],
            "angle": agent.get("angle", 0),
            "alive": agent["alive"]
        }
        
        zombie_snapshot = []
        for z in zombies:
            zombie_snapshot.append({
                "pos": z["pos"].copy(),
                "health": z["health"],
                "angle": z.get("angle", 0),
                "alive": z["alive"]
            })
        
        shot_snapshot = []
        if shots:
            for s in shots:
                # Handle new projectile format (pos, vel) or legacy format (origin, dx, dy)
                if "pos" in s:
                    shot_snapshot.append({
                        "pos": s["pos"].copy(),
                        "vel": s["vel"].copy(),
                        "active": s.get("active", True)
                    })
                else:
                    shot_snapshot.append({
                        "origin": s["origin"].copy(),
                        "dx": s["dx"], "dy": s["dy"],
                        "hit": s.get("hit", False)
                    })
        
        self.history.append({
            "agent": agent_snapshot,
            "zombies": zombie_snapshot,
            "shots": shot_snapshot,
            "obs": obs.copy() if obs is not None else None,
            "kills": kills
        })
    
    def clear_history(self):
        """Clear recorded history."""
        self.history = []
    
    def _draw_grid(self):
        """Draw background grid."""
        for x in range(0, self.game_width, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.game_height), 1)
        for y in range(0, self.game_height, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.game_width, y), 1)
    
    def _draw_obstacles(self):
        """Draw obstacles."""
        for obs in self.obstacles:
            ox1, oy1 = self.world_to_screen(obs[0], obs[3])  # Top-left in screen
            ox2, oy2 = self.world_to_screen(obs[2], obs[1])  # Bottom-right in screen
            obstacle_rect = pygame.Rect(ox1, oy1, ox2 - ox1, oy2 - oy1)
            pygame.draw.rect(self.screen, self.OBSTACLE_COLOR, obstacle_rect)
            pygame.draw.rect(self.screen, BLACK, obstacle_rect, 3)
    
    def _draw_player(self, agent):
        """Draw the player."""
        if not agent.get("alive", True):
            return
            
        pos = agent["pos"]
        sx, sy = self.world_to_screen(pos[0], pos[1])
        
        # Player body (circle)
        radius = int(self.scale * 0.5)
        pygame.draw.circle(self.screen, self.PLAYER_COLOR, (sx, sy), radius)
        pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
        
        # Draw facing direction (line from center)
        angle = agent.get("angle", 0)
        dir_len = self.scale * 0.7
        end_x = sx + math.cos(angle) * dir_len
        end_y = sy - math.sin(angle) * dir_len  # Flip Y
        pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 3)
        
        # Health bar
        health_pct = agent["health"] / 100.0
        bar_width = self.scale
        bar_height = 4
        bar_x = sx - bar_width // 2
        bar_y = sy - radius - 8
        
        pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
        pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), bar_height))
    
    def _draw_zombie(self, zombie):
        """Draw a zombie."""
        if not zombie.get("alive", True):
            return
            
        pos = zombie["pos"]
        sx, sy = self.world_to_screen(pos[0], pos[1])
        
        # Zombie body (circle)
        radius = int(self.scale * 0.4)
        pygame.draw.circle(self.screen, self.ZOMBIE_COLOR, (sx, sy), radius)
        pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
        
        # Draw facing direction
        angle = zombie.get("angle", 0)
        dir_len = self.scale * 0.5
        end_x = sx + math.cos(angle) * dir_len
        end_y = sy - math.sin(angle) * dir_len
        pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 2)
        
        # Health bar
        health_pct = zombie["health"] / 50.0  # ZOMBIE_HEALTH = 50
        bar_width = self.scale * 0.8
        bar_height = 3
        bar_x = sx - bar_width // 2
        bar_y = sy - radius - 6
        
        pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, self.ZOMBIE_COLOR, (bar_x, bar_y, int(bar_width * health_pct), bar_height))
    
    def _draw_shots(self, shots):
        """Draw projectiles as circles."""
        for shot in shots:
            # New projectile format: pos, vel, active
            if "pos" in shot:
                sx, sy = self.world_to_screen(shot["pos"][0], shot["pos"][1])
                pygame.draw.circle(self.screen, self.SHOT_COLOR_HIT, (sx, sy), 5)
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 5, 1)
            # Legacy format: origin, dx, dy
            elif "origin" in shot:
                ox, oy = self.world_to_screen(shot["origin"][0], shot["origin"][1])
                dx, dy = shot["dx"], -shot["dy"]
                end_x = ox + dx * self.game_width
                end_y = oy + dy * self.game_height
                color = self.SHOT_COLOR_HIT if shot.get("hit") else self.PLAYER_COLOR
                pygame.draw.line(self.screen, color, (ox, oy), (int(end_x), int(end_y)), 2)
    
    def render_frame(self, agent, zombies, shots=None, step=None, kills=0):
        """Render a single frame."""
        self.init()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Clear screen - tan background like pygame_renderer
        self.screen.fill((210, 180, 140))
        
        # Draw grid
        self._draw_grid()
        
        # Draw obstacles
        self._draw_obstacles()
        
        # Draw shots
        if shots:
            self._draw_shots(shots)
        
        # Draw zombies
        for zombie in zombies:
            self._draw_zombie(zombie)
        
        # Draw player
        self._draw_player(agent)
        
        # Draw step counter and stats
        if step is not None:
            text = self.font.render(f"Step: {step}", True, BLACK)
            self.screen.blit(text, (10, 10))
        
        kills_text = self.font.render(f"Kills: {kills}", True, BLACK)
        self.screen.blit(kills_text, (10, 30))
        
        alive_zombies = sum(1 for z in zombies if z.get("alive", True))
        zombie_text = self.font.render(f"Zombies: {alive_zombies}", True, BLACK)
        self.screen.blit(zombie_text, (10, 50))
        
        hp_text = self.font.render(f"HP: {int(agent['health'])}", True, BLACK)
        self.screen.blit(hp_text, (10, 70))
        
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
        obs = self.current_obs.flatten()
        agent_obs_dim = 27  # 1 health + 2 zombies x 4 + 8 wall + 1 shots + 9 quadrants
        total_obs_dim = agent_obs_dim * 2  # 54 for zombie env (duplicated)
        
        if len(obs) >= total_obs_dim * 4:
            # Get last frame from stacked obs
            latest_start = 3 * total_obs_dim
            obs = obs[latest_start:latest_start + total_obs_dim]
        elif len(obs) >= total_obs_dim:
            obs = obs[:total_obs_dim]
        
        # Use first 27 dims (player observation)
        agent_obs = obs[:agent_obs_dim] if len(obs) >= agent_obs_dim else obs
        
        y_start = 10
        
        # Title
        title = self.font.render("OBSERVATION", True, (255, 255, 255))
        self.screen.blit(title, (panel_x + 10, y_start))
        y_start += 30
        
        y = y_start
        
        # Health (0)
        y = self._draw_obs_section(panel_x, y, "Self State", [
            ("health", agent_obs[0]),
        ])
        
        # 2 Zombies (indices 1-8, 4 per zombie: cos, sin, dist, in_range)
        for i in range(2):
            base_idx = 1 + i * 4
            in_range = agent_obs[base_idx + 3] if len(agent_obs) > base_idx + 3 else 0
            range_str = "YES" if in_range > 0.5 else "NO"
            y = self._draw_obs_section(panel_x, y, f"Zombie {i+1} (range: {range_str})", [
                ("cos Δ", agent_obs[base_idx]),
                ("sin Δ", agent_obs[base_idx + 1]),
                ("dist", agent_obs[base_idx + 2]),
                ("in_range", agent_obs[base_idx + 3]),
            ])
        
        # Ray sensors visualization (9-16 wall only)
        wall_rays = agent_obs[9:17] if len(agent_obs) >= 17 else [0]*8
        # Use cos/sin from closest zombie direction for heading indicator
        cos_theta = agent_obs[1] if len(agent_obs) > 1 else 1.0
        sin_theta = agent_obs[2] if len(agent_obs) > 2 else 0.0
        y = self._draw_ray_viz(panel_x, y, wall_rays, cos_theta, sin_theta, None)
        
        # Shots remaining (17)
        if len(agent_obs) >= 18:
            y = self._draw_obs_section(panel_x, y, "Status", [
                ("shots_left", agent_obs[17]),
            ])
    
    def _draw_obs_section(self, panel_x, y, title, items):
        """Draw a section of observation values."""
        text = self.small_font.render(title, True, (180, 180, 180))
        self.screen.blit(text, (panel_x + 15, y))
        y += 16
        
        bar_width = 60
        bar_height = 10
        
        for name, value in items:
            label = self.small_font.render(f"{name}:", True, (150, 150, 150))
            self.screen.blit(label, (panel_x + 15, y))
            
            bar_x = panel_x + 70
            pygame.draw.rect(self.screen, (60, 60, 70), (bar_x, y + 2, bar_width, bar_height))
            
            if value >= 0:
                fill_width = int(bar_width * min(value, 1.0))
                fill_color = (80, 200, 80) if value < 0.9 else (255, 255, 80)
                pygame.draw.rect(self.screen, fill_color, (bar_x, y + 2, fill_width, bar_height))
            else:
                fill_width = int(bar_width * min(abs(value), 1.0))
                pygame.draw.rect(self.screen, (200, 80, 80), (bar_x + bar_width - fill_width, y + 2, fill_width, bar_height))
            
            val_text = self.small_font.render(f"{value:.2f}", True, (200, 200, 200))
            self.screen.blit(val_text, (bar_x + bar_width + 3, y))
            
            y += 14
        
        return y + 2
    
    def _draw_ray_viz(self, panel_x, y, rays, cos_theta, sin_theta, enemy_rays=None):
        """Draw ray sensor visualization as a mini radar."""
        text = self.small_font.render("Ray Sensors", True, (180, 180, 180))
        self.screen.blit(text, (panel_x + 15, y))
        y += 18
        
        center_x = panel_x + 80
        center_y = y + 40
        radius = 35
        
        pygame.draw.circle(self.screen, (60, 60, 70), (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (100, 100, 110), (center_x, center_y), radius, 1)
        
        if enemy_rays is None:
            enemy_rays = [0] * len(rays)
        
        n_rays = len(rays)
        for i, ray_dist in enumerate(rays):
            rel_angle = -math.pi/2 + (i / (n_rays - 1)) * math.pi
            world_angle = math.atan2(sin_theta, cos_theta) + rel_angle
            
            ray_len = radius * ray_dist
            end_x = center_x + math.cos(world_angle) * ray_len
            end_y = center_y - math.sin(world_angle) * ray_len
            
            enemy_hit = enemy_rays[i] > 0.5 if i < len(enemy_rays) else False
            if enemy_hit:
                color = (0, 255, 255)  # Cyan = enemy detected
            else:
                r = int(255 * (1 - ray_dist))
                g = int(255 * ray_dist)
                color = (r, g, 50)
            
            pygame.draw.line(self.screen, color, (center_x, center_y), (int(end_x), int(end_y)), 2)
            dot_size = 5 if enemy_hit else 3
            pygame.draw.circle(self.screen, color, (int(end_x), int(end_y)), dot_size)
        
        # Draw agent heading indicator
        head_len = radius + 5
        head_x = center_x + cos_theta * head_len
        head_y = center_y - sin_theta * head_len
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, center_y), (int(head_x), int(head_y)), 2)
        
        return y + 85
    
    def play_history(self, fps=10):
        """Play back recorded history."""
        self.init()
        
        for i, state in enumerate(self.history):
            if state.get("obs") is not None:
                self.current_obs = state["obs"]
            if not self.render_frame(
                state["agent"], 
                state["zombies"], 
                shots=state.get("shots", []),
                step=i,
                kills=state.get("kills", 0)
            ):
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


def run_zombie_renderer():
    """Run 5 zombie environments, pick the one with most kills, and render it."""
    import gymnasium as gym
    import worms_3d_gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    
    # Load model once
    model = None
    model_path = MODEL_PATH
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path)
    
    # Run 5 episodes and record each
    num_runs = 50
    all_histories = []
    all_kills = []
    
    print(f"Running {num_runs} simulations to find best episode...")
    
    for run_idx in range(num_runs):
        # Create fresh env for each run
        env = DummyVecEnv([lambda: gym.make("ZombieSurvival-v0")])
        env = VecFrameStack(env, n_stack=8)
        unwrapped = env.envs[0].unwrapped
        
        # Create renderer for recording (not displaying yet)
        renderer = ZombieRenderer(
            map_width=unwrapped.SIZE,
            map_depth=unwrapped.SIZE,
            scale=20,
            obstacles=unwrapped.OBSTACLES
        )
        
        # Run simulation
        obs = env.reset()
        renderer.set_observation(obs)
        renderer.record_state(
            unwrapped.agent, 
            unwrapped.zombies, 
            unwrapped.last_shots, 
            obs,
            kills=unwrapped.kills
        )
        
        kills = 0
        for step in range(1000):
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            action = action.flatten()
            
            obs, reward, done, info = env.step([action])
            terminated = done[0]
            
            renderer.set_observation(obs)
            renderer.record_state(
                unwrapped.agent, 
                unwrapped.zombies, 
                unwrapped.last_shots, 
                obs,
                kills=unwrapped.kills
            )
            
            if terminated:
                break
                
            kills = unwrapped.kills
        
        final_kills = kills
        all_histories.append(renderer.history)
        all_kills.append(final_kills)
        
        print(f"  Run {run_idx + 1}/{num_runs}: {final_kills} kills, {len(renderer.history)} frames")
        env.close()
    
    # Find best run
    best_idx = all_kills.index(max(all_kills))
    best_kills = all_kills[best_idx]
    best_history = all_histories[best_idx]
    
    print(f"\nBest run: #{best_idx + 1} with {best_kills} kills")
    print(f"All kills: {all_kills}")
    
    # Create renderer for playback
    env = DummyVecEnv([lambda: gym.make("ZombieSurvival-v0")])
    unwrapped = env.envs[0].unwrapped
    
    renderer = ZombieRenderer(
        map_width=unwrapped.SIZE,
        map_depth=unwrapped.SIZE,
        scale=20,
        obstacles=unwrapped.OBSTACLES
    )
    renderer.history = best_history
    
    env.close()
    
    print(f"Playing back best episode ({len(best_history)} frames)...")
    renderer.play_history(fps=10)


if __name__ == "__main__":
    run_zombie_renderer()
