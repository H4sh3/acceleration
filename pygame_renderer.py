import pygame
import numpy as np
import math

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BROWN = (139, 69, 19)
GRAY = (128, 128, 128)
TEAM_COLORS = [
    (255, 80, 80),   # Red team
    (80, 80, 255),   # Blue team
    (80, 255, 80),   # Green team
]

class PygameRenderer:
    def __init__(self, map_width, map_depth, scale=30):
        """
        Top-down 2D renderer for Worms 3D environment.
        
        Args:
            map_width: Width of the map in world units
            map_depth: Depth of the map in world units  
            scale: Pixels per world unit
        """
        self.map_width = map_width
        self.map_depth = map_depth
        self.scale = scale
        
        self.screen_width = map_width * scale
        self.screen_height = map_depth * scale
        
        self.screen = None
        self.clock = None
        self.font = None
        self.initialized = False
        
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
            self.initialized = True
    
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        sx = int(x * self.scale)
        sy = int((self.map_depth - 1 - y) * self.scale)  # Flip Y axis
        return sx, sy
    
    def record_state(self, agents, shots=None):
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
        
        self.history.append({"agents": agent_snapshot, "shots": shot_snapshot})
    
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
            health_pct = agent["health"] / 200.0
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
        
        pygame.display.flip()
        return True
    
    def play_history(self, fps=10):
        """
        Play back recorded history.
        
        Args:
            fps: Frames per second for playback
        """
        self.init()
        
        for i, state in enumerate(self.history):
            shots = state.get("shots", [])
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
    
    env = gym.make("Worms3D-v0")
    unwrapped = env.unwrapped
    
    # Create renderer
    renderer = PygameRenderer(
        map_width=unwrapped.SIZE,
        map_depth=unwrapped.SIZE,
        scale=30
    )
    
    # Load model if exists
    model = None
    model_path = "models/worms_final_model.zip"
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path)
    
    # Run simulation
    obs, info = env.reset()
    renderer.record_state(unwrapped.agents, unwrapped.last_shots)
    
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
        
        # Log actions
        action_names = ["nothing", "up", "down", "left", "right", "rot_toward", "rot_away", "shoot"]
        a0_name = action_names[actions[0]]
        a1_name = action_names[actions[1]]
        
        old_hp = [a["health"] for a in unwrapped.agents]
        
        obs, reward, terminated, truncated, info = env.step(actions)
        
        new_hp = [a["health"] for a in unwrapped.agents]
        
        # Check for hits
        for i in range(2):
            if new_hp[i] < old_hp[i]:
                hits += 1
                print(f"  Step {step}: Agent {1-i} HIT Agent {i}! HP: {old_hp[i]:.0f} -> {new_hp[i]:.0f}")
        
        total_reward += reward
        episode_steps += 1
        
        renderer.record_state(unwrapped.agents, unwrapped.last_shots)
        
        # Log every step
        a0 = unwrapped.agents[0]
        a1 = unwrapped.agents[1]
        aim0 = "AIM!" if obs[7] > 0.5 else "    "
        aim1 = "AIM!" if obs[15] > 0.5 else "    "
        print(f"Step {step:3}: A0={a0_name:10} {aim0} | A1={a1_name:10} {aim1} | HP: {new_hp[0]:.0f} vs {new_hp[1]:.0f}")
        
        if terminated or truncated:
            winner = "T0" if unwrapped.agents[0]["alive"] else "T1" if unwrapped.agents[1]["alive"] else "Draw"
            print(f"=== Episode {episode} ended at step {episode_steps}: {winner} wins! Total reward: {total_reward:.1f}, Hits: {hits} ===")
            episode += 1
            total_reward = 0
            episode_steps = 0
            hits = 0
            obs, info = env.reset()
            renderer.record_state(unwrapped.agents, unwrapped.last_shots)
    
    env.close()
    
    print(f"Recorded {len(renderer.history)} frames. Playing back...")
    renderer.play_history(fps=15)


if __name__ == "__main__":
    run_with_pygame_renderer()
