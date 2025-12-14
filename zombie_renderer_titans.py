"""Pygame renderer for Titans-memory agents showing arena + POV + memory visualization."""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import math
import pygame


def get_newest_titans_model(models_dir="models"):
    """Find the newest Titans model file."""
    # Look for titans-specific models first
    for pattern_name in ["titans", "yaad", "moneta", "memora"]:
        pattern = os.path.join(models_dir, f"**/*{pattern_name}*.zip")
        model_files = glob.glob(pattern, recursive=True)
        if model_files:
            newest = max(model_files, key=os.path.getmtime)
            print(f"Found {pattern_name} model: {newest}")
            return newest
    
    # Fall back to CNN models
    pattern = os.path.join(models_dir, "**/*cnn*.zip")
    model_files = glob.glob(pattern, recursive=True)
    if model_files:
        newest = max(model_files, key=os.path.getmtime)
        print(f"Found CNN model: {newest}")
        return newest
    
    return None


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (40, 40, 50)
DARKER_GRAY = (25, 25, 35)

# Channel colors for POV visualization
CHANNEL_COLORS = [
    (100, 100, 120),   # Channel 0: Walls - gray-blue
    (200, 60, 60),     # Channel 1: Zombies - red
    (255, 255, 0),     # Channel 2: Projectiles - yellow
    (100, 255, 100),   # Channel 3: Aim direction - green
    (255, 150, 50),    # Channel 4: Enemy radar - orange
]

# Memory visualization colors
MEMORY_POSITIVE = (80, 200, 255)  # Cyan for positive values
MEMORY_NEGATIVE = (255, 100, 80)  # Red-orange for negative values
SURPRISE_COLOR = (255, 220, 50)   # Yellow-gold for surprise


class ZombieRendererTitans:
    """Renderer showing arena + POV + Titans memory state."""
    
    OBSTACLE_COLOR = (101, 67, 33)
    PLAYER_COLOR = (50, 200, 100)
    ZOMBIE_COLOR = (200, 60, 60)
    SHOT_COLOR = (255, 255, 0)
    
    def __init__(self, map_size, scale=18, obstacles=None):
        self.map_size = map_size
        self.scale = scale
        self.obstacles = obstacles or []
        
        # Arena dimensions
        self.arena_width = map_size * scale
        self.arena_height = map_size * scale
        
        # Side panel for POV + memory
        self.panel_width = 350
        
        # Total screen size
        self.screen_width = self.arena_width + self.panel_width + 20
        self.screen_height = max(self.arena_height, 700)
        
        self.screen = None
        self.clock = None
        self.font = None
        self.small_font = None
        self.tiny_font = None
        self.initialized = False
        
        self.current_obs = None
        self.surprise_value = 0.0
        self.surprise_history = []
        self.memory_state = None
        self.kills_history = []
        self.hp_history = []
        self.history = []
    
    def init(self):
        """Initialize pygame display."""
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Zombie Survival - Titans Memory View")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)
            self.tiny_font = pygame.font.Font(None, 14)
            self.initialized = True
    
    def set_observation(self, obs):
        """Set current observation."""
        self.current_obs = obs
    
    def set_memory_info(self, surprise=0.0, memory_state=None):
        """Set Titans memory information for visualization."""
        self.surprise_value = surprise
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > 200:
            self.surprise_history = self.surprise_history[-200:]
        
        if memory_state is not None:
            self.memory_state = memory_state.cpu().numpy() if hasattr(memory_state, 'cpu') else memory_state
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to arena screen coordinates."""
        sx = int(x * self.scale)
        sy = int((self.map_size - 1 - y) * self.scale)
        return sx, sy
    
    def record_state(self, agent, zombies, shots=None, obs=None, kills=0, surprise=0.0, memory_state=None):
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
            "kills": kills,
            "surprise": surprise,
            "memory_state": memory_state.copy() if memory_state is not None else None
        })
        
        self.kills_history.append(kills)
        self.hp_history.append(agent["health"])
    
    def clear_history(self):
        """Clear recorded history."""
        self.history = []
        self.surprise_history = []
        self.kills_history = []
        self.hp_history = []
    
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
    
    def _draw_panel(self):
        """Draw the side panel with POV and memory visualization."""
        panel_x = self.arena_width + 10
        panel_y = 10
        
        # Panel background
        pygame.draw.rect(self.screen, DARK_GRAY, 
                        (panel_x, 0, self.panel_width + 10, self.screen_height))
        
        # === TITANS MEMORY SECTION ===
        title = self.font.render("TITANS MEMORY", True, SURPRISE_COLOR)
        self.screen.blit(title, (panel_x + 5, panel_y))
        panel_y += 25
        
        # Surprise meter
        label = self.small_font.render("Surprise:", True, WHITE)
        self.screen.blit(label, (panel_x + 5, panel_y))
        
        # Surprise bar
        bar_x = panel_x + 70
        bar_width = 150
        pygame.draw.rect(self.screen, DARKER_GRAY, (bar_x, panel_y, bar_width, 14))
        
        # Normalize surprise for display (log scale works better)
        surprise_normalized = min(1.0, self.surprise_value / 50000) if self.surprise_value > 0 else 0
        fill_width = int(bar_width * surprise_normalized)
        
        # Color gradient based on surprise level
        if surprise_normalized > 0.7:
            bar_color = (255, 80, 80)  # High surprise - red
        elif surprise_normalized > 0.3:
            bar_color = SURPRISE_COLOR  # Medium - yellow
        else:
            bar_color = (80, 200, 80)  # Low - green
        
        pygame.draw.rect(self.screen, bar_color, (bar_x, panel_y, fill_width, 14))
        pygame.draw.rect(self.screen, WHITE, (bar_x, panel_y, bar_width, 14), 1)
        
        # Surprise value
        val_text = self.tiny_font.render(f"{self.surprise_value:.0f}", True, GRAY)
        self.screen.blit(val_text, (bar_x + bar_width + 5, panel_y + 1))
        panel_y += 20
        
        # Surprise history graph
        if len(self.surprise_history) > 1:
            label = self.small_font.render("Surprise History:", True, GRAY)
            self.screen.blit(label, (panel_x + 5, panel_y))
            panel_y += 16
            
            graph_width = self.panel_width - 20
            graph_height = 40
            pygame.draw.rect(self.screen, DARKER_GRAY, (panel_x + 5, panel_y, graph_width, graph_height))
            
            # Plot surprise history
            max_surprise = max(self.surprise_history) if self.surprise_history else 1
            if max_surprise > 0:
                points = []
                for i, s in enumerate(self.surprise_history[-graph_width:]):
                    x = panel_x + 5 + i
                    y = panel_y + graph_height - int((s / max_surprise) * (graph_height - 2))
                    points.append((x, y))
                
                if len(points) > 1:
                    pygame.draw.lines(self.screen, SURPRISE_COLOR, False, points, 1)
            
            pygame.draw.rect(self.screen, GRAY, (panel_x + 5, panel_y, graph_width, graph_height), 1)
            panel_y += graph_height + 10
        
        # Memory state visualization
        if self.memory_state is not None:
            label = self.small_font.render("Memory State (128 dims):", True, WHITE)
            self.screen.blit(label, (panel_x + 5, panel_y))
            panel_y += 16
            
            # Draw memory as a heatmap grid (16x8 = 128)
            mem_flat = self.memory_state.flatten()[:128]
            grid_w, grid_h = 16, 8
            cell_size = 8
            
            max_val = max(abs(mem_flat.max()), abs(mem_flat.min()), 0.1)
            
            for i, val in enumerate(mem_flat):
                gx = i % grid_w
                gy = i // grid_w
                px = panel_x + 5 + gx * cell_size
                py = panel_y + gy * cell_size
                
                # Color based on value (positive = cyan, negative = red)
                intensity = min(1.0, abs(val) / max_val)
                if val >= 0:
                    r = int(80 * (1 - intensity))
                    g = int(80 + 120 * intensity)
                    b = int(80 + 175 * intensity)
                else:
                    r = int(80 + 175 * intensity)
                    g = int(80 * (1 - intensity))
                    b = int(80 * (1 - intensity))
                
                pygame.draw.rect(self.screen, (r, g, b), (px, py, cell_size - 1, cell_size - 1))
            
            panel_y += grid_h * cell_size + 10
            
            # Memory stats
            mem_mean = np.mean(mem_flat)
            mem_std = np.std(mem_flat)
            stats_text = self.tiny_font.render(f"μ={mem_mean:.2f} σ={mem_std:.2f}", True, GRAY)
            self.screen.blit(stats_text, (panel_x + 5, panel_y))
            panel_y += 15
        
        panel_y += 10
        pygame.draw.line(self.screen, GRAY, (panel_x + 5, panel_y), (panel_x + self.panel_width - 5, panel_y), 1)
        panel_y += 10
        
        # === POV SECTION ===
        title = self.small_font.render("AGENT POV (Egocentric)", True, WHITE)
        self.screen.blit(title, (panel_x + 5, panel_y))
        panel_y += 20
        
        if self.current_obs is not None and isinstance(self.current_obs, dict):
            image = self.current_obs["image"]
            vector = self.current_obs["vector"]
            
            n_channels, grid_h, grid_w = image.shape
            cell_size = min((self.panel_width - 20) // grid_w, 10)
            
            # Draw combined view
            for gy in range(grid_h):
                for gx in range(grid_w):
                    px = panel_x + 5 + gx * cell_size
                    py = panel_y + gy * cell_size
                    
                    r, g, b = 30, 30, 40
                    for ch in range(n_channels):
                        val = image[ch, gy, gx]
                        if val > 0.1:
                            cr, cg, cb = CHANNEL_COLORS[ch]
                            r = min(255, r + int(cr * val * 0.8))
                            g = min(255, g + int(cg * val * 0.8))
                            b = min(255, b + int(cb * val * 0.8))
                    
                    pygame.draw.rect(self.screen, (r, g, b), (px, py, cell_size, cell_size))
            
            # Center marker
            center_x = panel_x + 5 + (grid_w // 2) * cell_size + cell_size // 2
            center_y = panel_y + (grid_h // 2) * cell_size + cell_size // 2
            pygame.draw.circle(self.screen, WHITE, (center_x, center_y), cell_size // 2, 1)
            
            panel_y += grid_h * cell_size + 10
            
            # Vector observation bars
            vector_names = ["Health", "Shots", "Aim"]
            for i, name in enumerate(vector_names):
                val = vector[i] if i < len(vector) else 0
                
                label = self.tiny_font.render(f"{name}:", True, GRAY)
                self.screen.blit(label, (panel_x + 5, panel_y))
                
                bar_x = panel_x + 50
                bar_width = 80
                pygame.draw.rect(self.screen, DARKER_GRAY, (bar_x, panel_y, bar_width, 10))
                fill_color = (80, 200, 80) if val > 0.5 else (200, 200, 80) if val > 0.25 else (200, 80, 80)
                pygame.draw.rect(self.screen, fill_color, (bar_x, panel_y, int(bar_width * val), 10))
                
                val_text = self.tiny_font.render(f"{val:.2f}", True, GRAY)
                self.screen.blit(val_text, (bar_x + bar_width + 5, panel_y))
                panel_y += 14
        
        panel_y += 10
        
        # Channel legend
        label = self.tiny_font.render("Channels:", True, GRAY)
        self.screen.blit(label, (panel_x + 5, panel_y))
        panel_y += 14
        
        channel_names = ["Walls", "Zombies", "Projectiles", "Aim", "Radar"]
        for i in range(0, len(channel_names), 2):
            for j in range(2):
                if i + j < len(channel_names):
                    idx = i + j
                    x_offset = j * 80
                    color = CHANNEL_COLORS[idx]
                    pygame.draw.rect(self.screen, color, (panel_x + 5 + x_offset, panel_y, 10, 10))
                    text = self.tiny_font.render(channel_names[idx], True, GRAY)
                    self.screen.blit(text, (panel_x + 18 + x_offset, panel_y))
            panel_y += 14
    
    def render_frame(self, agent, zombies, shots=None, step=None, kills=0):
        """Render a single frame."""
        self.init()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        self._draw_arena(agent, zombies, shots, step, kills)
        self._draw_panel()
        
        pygame.display.flip()
        return True
    
    def play_history(self, fps=10):
        """Play back recorded history."""
        self.init()
        
        for i, state in enumerate(self.history):
            if state.get("obs") is not None:
                self.current_obs = state["obs"]
            if state.get("memory_state") is not None:
                self.memory_state = state["memory_state"]
            self.surprise_value = state.get("surprise", 0)
            
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


def run_titans_renderer():
    """Run the Titans zombie environment with visualization."""
    import gymnasium as gym
    import torch
    import worms_3d_gym
    from stable_baselines3 import PPO
    
    # Create environment
    env = gym.make("ZombieSurvivalCNN-v0")
    unwrapped = env.unwrapped
    
    # Create renderer
    renderer = ZombieRendererTitans(
        map_size=unwrapped.SIZE,
        scale=18,
        obstacles=unwrapped.OBSTACLES
    )
    
    # Load model
    model = None
    model_path = get_newest_titans_model()
    has_titans_memory = False
    
    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        try:
            model = PPO.load(model_path)
            
            # Check if it has Titans memory
            if hasattr(model.policy, 'features_extractor'):
                extractor = model.policy.features_extractor
                if hasattr(extractor, 'get_surprise'):
                    has_titans_memory = True
                    print("✓ Model has Titans memory module")
                else:
                    print("Model is standard CNN (no Titans memory)")
            
            print("Model loaded successfully")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Using random actions")
    else:
        print("No model found, using random actions")
    
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
    
    print("\n" + "="*50)
    print("TITANS MEMORY ZOMBIE SURVIVAL")
    print("="*50)
    print("Controls: ESC to quit")
    print("="*50 + "\n")
    
    total_reward = 0
    episode = 1
    episode_steps = 0
    
    for step in range(2000):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        action_val = int(action) if isinstance(action, (int, np.integer)) else int(action.flatten()[0])
        
        old_hp = unwrapped.agent["health"]
        old_kills = unwrapped.kills
        
        obs, reward, terminated, truncated, info = env.step(action_val)
        
        new_hp = unwrapped.agent["health"]
        new_kills = unwrapped.kills
        
        # Get memory info if available
        surprise = 0.0
        memory_state = None
        if has_titans_memory and model:
            try:
                extractor = model.policy.features_extractor
                surprise = extractor.get_surprise()
                
                # Navigate to memory state:
                # TitansFeatureExtractorSB3.extractor -> TitansCNNExtractor
                # TitansCNNExtractor.titans_memory -> TitansFeatureExtractor  
                # TitansFeatureExtractor._memory_state
                if hasattr(extractor, 'extractor'):
                    cnn_extractor = extractor.extractor  # TitansCNNExtractor
                    if hasattr(cnn_extractor, 'titans_memory'):
                        titans_mem = cnn_extractor.titans_memory  # TitansFeatureExtractor
                        if hasattr(titans_mem, '_memory_state') and titans_mem._memory_state is not None:
                            memory_state = titans_mem._memory_state[0].detach().cpu().numpy()
            except Exception as e:
                if step == 0:
                    print(f"Note: Could not get memory state: {e}")
        
        renderer.set_observation(obs)
        renderer.set_memory_info(surprise, memory_state)
        
        if new_hp < old_hp:
            print(f"  Step {step}: HIT! HP: {old_hp:.0f} -> {new_hp:.0f} | Surprise: {surprise:.0f}")
        if new_kills > old_kills:
            print(f"  Step {step}: KILL! Total: {new_kills} | Surprise: {surprise:.0f}")
        
        total_reward += reward
        episode_steps += 1
        
        renderer.record_state(
            unwrapped.agent,
            unwrapped.zombies,
            unwrapped.last_shots,
            obs,
            kills=unwrapped.kills,
            surprise=surprise,
            memory_state=memory_state
        )
        
        # Render live
        if not renderer.render_frame(
            unwrapped.agent,
            unwrapped.zombies,
            unwrapped.last_shots,
            step=step,
            kills=unwrapped.kills
        ):
            break
        
        renderer.clock.tick(15)  # 15 FPS for live view
        
        if terminated or truncated:
            print(f"\n=== Episode {episode}: Steps={episode_steps}, Kills={unwrapped.kills}, Reward={total_reward:.1f} ===\n")
            episode += 1
            total_reward = 0
            episode_steps = 0
            obs, _ = env.reset()
            renderer.set_observation(obs)
            
            # Reset memory at episode boundary
            if has_titans_memory and model:
                try:
                    extractor = model.policy.features_extractor
                    if hasattr(extractor, 'reset_memory'):
                        extractor.reset_memory(1)
                except Exception:
                    pass
    
    env.close()
    renderer.close()


if __name__ == "__main__":
    run_titans_renderer()
