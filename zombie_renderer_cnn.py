"""Pygame renderer for ZombieSurvivalCNNEnv showing arena + agent's egocentric POV."""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import math
import pygame


def get_newest_cnn_model(models_dir="models"):
    """Find the newest CNN model file."""
    pattern = os.path.join(models_dir, "**", "*cnn*.zip")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        # Fall back to any model
        pattern = os.path.join(models_dir, "**", "*.zip")
        model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        return None
    
    newest = max(model_files, key=os.path.getmtime)
    print(f"Using model: {newest}")
    return newest


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (40, 40, 50)

# Channel colors for POV visualization
CHANNEL_COLORS = [
    (100, 100, 120),   # Channel 0: Walls - gray-blue
    (200, 60, 60),     # Channel 1: Zombies - red
    (255, 255, 0),     # Channel 2: Projectiles - yellow
    (100, 255, 100),   # Channel 3: Aim direction - green
    (255, 150, 50),    # Channel 4: Enemy radar - orange
]


class ZombieRendererCNN:
    """Renderer showing both arena view and agent's egocentric CNN observation."""
    
    # Colors
    OBSTACLE_COLOR = (101, 67, 33)
    PLAYER_COLOR = (50, 200, 100)
    ZOMBIE_COLOR = (200, 60, 60)
    SHOT_COLOR = (255, 255, 0)
    
    def __init__(self, map_size, scale=20, pov_scale=10, obstacles=None):
        """
        Args:
            map_size: Size of the arena
            scale: Pixels per world unit for arena view
            pov_scale: Pixels per cell for POV view
            obstacles: List of obstacle bounds
        """
        self.map_size = map_size
        self.scale = scale
        self.pov_scale = pov_scale
        self.obstacles = obstacles or []
        
        # Arena dimensions
        self.arena_width = map_size * scale
        self.arena_height = map_size * scale
        
        # POV panel dimensions (will be set when we know grid size)
        self.pov_panel_width = 300
        self.pov_panel_height = 300
        
        # Total screen size
        self.screen_width = self.arena_width + self.pov_panel_width + 20
        self.screen_height = max(self.arena_height, self.pov_panel_height + 150)
        
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.initialized = False
        
        self.current_obs = None
        self.history = []
    
    def init(self):
        """Initialize pygame display."""
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Zombie Survival - CNN View")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.initialized = True
    
    def set_observation(self, obs):
        """Set current observation for visualization."""
        self.current_obs = obs
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to arena screen coordinates."""
        sx = int(x * self.scale)
        sy = int((self.map_size - 1 - y) * self.scale)
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
                if "pos" in s:
                    shot_snapshot.append({
                        "pos": s["pos"].copy(),
                        "vel": s["vel"].copy(),
                        "active": s.get("active", True)
                    })
        
        # Handle dict observation (CNN env)
        obs_copy = None
        if obs is not None:
            if isinstance(obs, dict):
                obs_copy = {
                    "image": obs["image"].copy(),
                    "vector": obs["vector"].copy()
                }
            else:
                obs_copy = obs.copy()
        
        self.history.append({
            "agent": agent_snapshot,
            "zombies": zombie_snapshot,
            "shots": shot_snapshot,
            "obs": obs_copy,
            "kills": kills
        })
    
    def clear_history(self):
        """Clear recorded history."""
        self.history = []
    
    def _draw_arena(self, agent, zombies, shots, step, kills):
        """Draw the main arena view."""
        # Background
        self.screen.fill((210, 180, 140), (0, 0, self.arena_width, self.arena_height))
        
        # Grid
        for x in range(0, self.arena_width, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.arena_height), 1)
        for y in range(0, self.arena_height, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.arena_width, y), 1)
        
        # Obstacles
        for obs in self.obstacles:
            ox1, oy1 = self.world_to_screen(obs[0], obs[3])
            ox2, oy2 = self.world_to_screen(obs[2], obs[1])
            rect = pygame.Rect(ox1, oy1, ox2 - ox1, oy2 - oy1)
            pygame.draw.rect(self.screen, self.OBSTACLE_COLOR, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 3)
        
        # Shots/projectiles
        if shots:
            for shot in shots:
                if "pos" in shot:
                    sx, sy = self.world_to_screen(shot["pos"][0], shot["pos"][1])
                    pygame.draw.circle(self.screen, self.SHOT_COLOR, (sx, sy), 5)
                    pygame.draw.circle(self.screen, WHITE, (sx, sy), 5, 1)
        
        # Zombies
        for zombie in zombies:
            if not zombie.get("alive", True):
                continue
            pos = zombie["pos"]
            sx, sy = self.world_to_screen(pos[0], pos[1])
            radius = int(self.scale * 0.4)
            pygame.draw.circle(self.screen, self.ZOMBIE_COLOR, (sx, sy), radius)
            pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
            
            # Facing direction
            angle = zombie.get("angle", 0)
            end_x = sx + math.cos(angle) * self.scale * 0.5
            end_y = sy - math.sin(angle) * self.scale * 0.5
            pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 2)
        
        # Player
        if agent.get("alive", True):
            pos = agent["pos"]
            sx, sy = self.world_to_screen(pos[0], pos[1])
            radius = int(self.scale * 0.5)
            pygame.draw.circle(self.screen, self.PLAYER_COLOR, (sx, sy), radius)
            pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
            
            # Facing direction
            angle = agent.get("angle", 0)
            dir_len = self.scale * 0.7
            end_x = sx + math.cos(angle) * dir_len
            end_y = sy - math.sin(angle) * dir_len
            pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 3)
            
            # Health bar
            health_pct = agent["health"] / 100.0
            bar_width = self.scale
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 8
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, 4))
            health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), 4))
        
        # Stats overlay
        if step is not None:
            text = self.font.render(f"Step: {step}", True, BLACK)
            self.screen.blit(text, (10, 10))
        
        kills_text = self.font.render(f"Kills: {kills}", True, BLACK)
        self.screen.blit(kills_text, (10, 30))
        
        hp_text = self.font.render(f"HP: {int(agent['health'])}", True, BLACK)
        self.screen.blit(hp_text, (10, 50))
    
    def _draw_pov_panel(self):
        """Draw the agent's egocentric POV visualization."""
        panel_x = self.arena_width + 10
        panel_y = 10
        
        # Panel background
        pygame.draw.rect(self.screen, DARK_GRAY, 
                        (panel_x, 0, self.pov_panel_width + 10, self.screen_height))
        
        # Title
        title = self.font.render("AGENT POV (Egocentric)", True, WHITE)
        self.screen.blit(title, (panel_x + 5, panel_y))
        panel_y += 30
        
        if self.current_obs is None:
            return
        
        # Get image observation
        if isinstance(self.current_obs, dict):
            image = self.current_obs["image"]
            vector = self.current_obs["vector"]
        else:
            # Not a CNN observation
            text = self.small_font.render("(Not CNN observation)", True, GRAY)
            self.screen.blit(text, (panel_x + 5, panel_y))
            return
        
        n_channels, grid_h, grid_w = image.shape
        cell_size = min(self.pov_panel_width // grid_w, 200 // grid_h)
        
        # Draw combined view (all channels overlaid)
        combined_y = panel_y
        label = self.small_font.render("Combined View:", True, WHITE)
        self.screen.blit(label, (panel_x + 5, combined_y))
        combined_y += 18
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                px = panel_x + 5 + gx * cell_size
                py = combined_y + gy * cell_size
                
                # Blend channels
                r, g, b = 30, 30, 40  # Base dark color
                
                for ch in range(n_channels):
                    val = image[ch, gy, gx]
                    if val > 0.1:
                        cr, cg, cb = CHANNEL_COLORS[ch]
                        r = min(255, r + int(cr * val * 0.8))
                        g = min(255, g + int(cg * val * 0.8))
                        b = min(255, b + int(cb * val * 0.8))
                
                pygame.draw.rect(self.screen, (r, g, b), (px, py, cell_size, cell_size))
        
        # Draw grid lines
        for i in range(grid_w + 1):
            x = panel_x + 5 + i * cell_size
            pygame.draw.line(self.screen, (60, 60, 70), 
                           (x, combined_y), (x, combined_y + grid_h * cell_size), 1)
        for i in range(grid_h + 1):
            y = combined_y + i * cell_size
            pygame.draw.line(self.screen, (60, 60, 70),
                           (panel_x + 5, y), (panel_x + 5 + grid_w * cell_size, y), 1)
        
        # Mark center (agent position)
        center_x = panel_x + 5 + (grid_w // 2) * cell_size + cell_size // 2
        center_y = combined_y + (grid_h // 2) * cell_size + cell_size // 2
        pygame.draw.circle(self.screen, WHITE, (center_x, center_y), cell_size // 3, 2)
        
        panel_y = combined_y + grid_h * cell_size + 15
        
        # Channel legend
        legend_y = panel_y
        label = self.small_font.render("Channels:", True, WHITE)
        self.screen.blit(label, (panel_x + 5, legend_y))
        legend_y += 16
        
        channel_names = ["Walls", "Zombies", "Projectiles", "Aim", "Radar"]
        for i, name in enumerate(channel_names):
            color = CHANNEL_COLORS[i] if i < len(CHANNEL_COLORS) else GRAY
            pygame.draw.rect(self.screen, color, (panel_x + 10, legend_y, 12, 12))
            text = self.small_font.render(name, True, GRAY)
            self.screen.blit(text, (panel_x + 28, legend_y))
            legend_y += 16
        
        legend_y += 10
        
        # Vector observation
        label = self.small_font.render("Vector Obs:", True, WHITE)
        self.screen.blit(label, (panel_x + 5, legend_y))
        legend_y += 18
        
        vector_names = ["Health", "Shots Left", "Aim On Target"]
        for i, name in enumerate(vector_names):
            val = vector[i] if i < len(vector) else 0
            
            # Draw bar
            bar_x = panel_x + 10
            bar_width = 100
            pygame.draw.rect(self.screen, (60, 60, 70), (bar_x, legend_y, bar_width, 12))
            fill_color = (80, 200, 80) if val > 0.5 else (200, 200, 80)
            pygame.draw.rect(self.screen, fill_color, (bar_x, legend_y, int(bar_width * val), 12))
            
            # Label
            text = self.small_font.render(f"{name}: {val:.2f}", True, GRAY)
            self.screen.blit(text, (bar_x + bar_width + 5, legend_y))
            legend_y += 16
        
        # Draw individual channel views (smaller)
        legend_y += 15
        label = self.small_font.render("Individual Channels:", True, WHITE)
        self.screen.blit(label, (panel_x + 5, legend_y))
        legend_y += 18
        
        small_cell = 6
        for ch in range(n_channels):
            ch_x = panel_x + 5 + (ch % 2) * (grid_w * small_cell + 10)
            ch_y = legend_y + (ch // 2) * (grid_h * small_cell + 20)
            
            # Channel label
            text = self.small_font.render(channel_names[ch], True, CHANNEL_COLORS[ch])
            self.screen.blit(text, (ch_x, ch_y))
            ch_y += 14
            
            # Draw channel
            for gy in range(grid_h):
                for gx in range(grid_w):
                    val = image[ch, gy, gx]
                    if val > 0.1:
                        color = CHANNEL_COLORS[ch]
                    else:
                        color = (20, 20, 25)
                    px = ch_x + gx * small_cell
                    py = ch_y + gy * small_cell
                    pygame.draw.rect(self.screen, color, (px, py, small_cell, small_cell))
    
    def render_frame(self, agent, zombies, shots=None, step=None, kills=0):
        """Render a single frame."""
        self.init()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Draw arena
        self._draw_arena(agent, zombies, shots, step, kills)
        
        # Draw POV panel
        self._draw_pov_panel()
        
        pygame.display.flip()
        return True
    
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
        
        # Keep window open
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
    
    def close(self):
        """Close the renderer."""
        if self.initialized:
            pygame.quit()
            self.initialized = False


def run_zombie_renderer_cnn():
    """Run the CNN zombie environment with visualization."""
    import gymnasium as gym
    import worms_3d_gym
    from stable_baselines3 import PPO
    
    # Create environment
    env = gym.make("ZombieSurvivalCNN-v0")
    unwrapped = env.unwrapped
    
    # Create renderer
    renderer = ZombieRendererCNN(
        map_size=unwrapped.SIZE,
        scale=20,
        pov_scale=12,
        obstacles=unwrapped.OBSTACLES
    )
    
    # Load model if available (must match current observation space)
    model = None
    model_path = get_newest_cnn_model()
    
    if model_path and os.path.exists(model_path):
        print(f"Found model: {model_path}")
        try:
            loaded_model = PPO.load(model_path)
            # Check if observation spaces match
            model_img_shape = loaded_model.observation_space.spaces["image"].shape
            env_img_shape = env.observation_space.spaces["image"].shape
            if model_img_shape != env_img_shape:
                print(f"Model image shape {model_img_shape} != env image shape {env_img_shape}")
                print("Using random actions (model incompatible)")
            else:
                model = loaded_model
                print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using random actions")
    else:
        print("No CNN model found, using random actions")
    
    # Run simulation
    obs, _ = env.reset()
    renderer.set_observation(obs)
    renderer.record_state(
        unwrapped.agent,
        unwrapped.zombies,
        unwrapped.last_shots,
        obs,
        kills=unwrapped.kills
    )
    
    print("Running zombie survival CNN simulation...")
    total_reward = 0
    episode = 1
    episode_steps = 0
    
    for step in range(1000):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        action_val = int(action) if isinstance(action, (int, np.integer)) else int(action.flatten()[0])
        
        action_names = ["nothing", "up", "down", "left", "right", 
                       "fine_rot_L", "fine_rot_R", "coarse_rot_L", "coarse_rot_R", 
                       "shoot", "dash"]
        action_name = action_names[action_val] if action_val < len(action_names) else f"action_{action_val}"
        
        old_hp = unwrapped.agent["health"]
        old_kills = unwrapped.kills
        
        obs, reward, terminated, truncated, info = env.step(action_val)
        
        new_hp = unwrapped.agent["health"]
        new_kills = unwrapped.kills
        
        if new_hp < old_hp:
            print(f"  Step {step}: Player HIT! HP: {old_hp:.0f} -> {new_hp:.0f}")
        if new_kills > old_kills:
            print(f"  Step {step}: ZOMBIE KILLED! Total kills: {new_kills}")
        
        total_reward += reward
        episode_steps += 1
        
        renderer.set_observation(obs)
        renderer.record_state(
            unwrapped.agent,
            unwrapped.zombies,
            unwrapped.last_shots,
            obs,
            kills=unwrapped.kills
        )
        
        if step % 50 == 0:
            alive_zombies = sum(1 for z in unwrapped.zombies if z["alive"])
            print(f"Step {step:3}: action={action_name:12} | HP: {new_hp:.0f} | Kills: {new_kills} | Zombies: {alive_zombies}")
        
        if terminated or truncated:
            print(f"=== Episode {episode} ended at step {episode_steps}: Kills: {unwrapped.kills}, Reward: {total_reward:.1f} ===")
            episode += 1
            total_reward = 0
            episode_steps = 0
            obs, _ = env.reset()
            renderer.set_observation(obs)
            renderer.record_state(
                unwrapped.agent,
                unwrapped.zombies,
                unwrapped.last_shots,
                obs,
                kills=unwrapped.kills
            )
    
    env.close()
    
    print(f"Recorded {len(renderer.history)} frames. Playing back...")
    renderer.play_history(fps=10)


if __name__ == "__main__":
    run_zombie_renderer_cnn()
