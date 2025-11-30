"""Play against a trained agent using keyboard controls.

Controls:
  W/S - Move up/down
  A/D - Move left/right
  Arrow Left/Right - Rotate
  Space - Shoot
  R - Reset episode
  ESC - Quit
"""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import math
import argparse
import random

from stable_baselines3 import PPO
from worms_3d_gym.envs import Worms3DPettingZooEnv


# Colors
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
TEAM_COLORS = [
    (80, 255, 80),   # Green - Player
    (255, 80, 80),   # Red - AI opponent
]


def find_best_agent(models_dir="models"):
    """Find the newest trained agent model."""
    patterns = [
        os.path.join(models_dir, "population_*", "agent_*_final.zip"),
        os.path.join(models_dir, "multiagent_*", "agent*_final.zip"),
        os.path.join(models_dir, "**", "*.zip"),
    ]
    
    for pattern in patterns:
        files = glob.glob(pattern, recursive=True)
        if files:
            newest = max(files, key=os.path.getmtime)
            print(f"Found agent: {newest}")
            return newest
    
    raise FileNotFoundError(f"No model files found in {models_dir}")


class GameRenderer:
    """Pygame renderer for human vs AI gameplay."""
    
    def __init__(self, map_size, scale=20, obstacles=None):
        self.map_size = map_size
        self.scale = scale
        self.obstacles = obstacles or []
        
        self.game_width = map_size * scale
        self.game_height = map_size * scale
        self.info_panel_width = 250
        self.screen_width = self.game_width + self.info_panel_width
        self.screen_height = self.game_height
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Worms 3D - Play vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.title_font = pygame.font.Font(None, 36)
    
    def world_to_screen(self, x, y):
        sx = int(x * self.scale)
        sy = int((self.map_size - 1 - y) * self.scale)
        return sx, sy
    
    def render(self, agents, shots=None, step=0, player_score=0, ai_score=0, actions=None, observations=None):
        """Render the game state.
        
        Args:
            actions: List of [player_action, ai_action] to display on agents
            observations: Dict of agent observations to display
        """
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
                # Use same Y-inversion as facing direction indicator
                shot_dx = shot["dx"]
                shot_dy = -shot["dy"]  # Negate for screen coords (Y is inverted)
                
                # Draw shot line extending from origin
                line_len = self.screen_width  # Long enough to cross screen
                end_x = ox + shot_dx * line_len
                end_y = oy + shot_dy * line_len
                
                color = (255, 255, 0) if shot["hit"] else TEAM_COLORS[shot["team"]]
                width = 4 if shot["hit"] else 2
                pygame.draw.line(self.screen, color, (ox, oy), (int(end_x), int(end_y)), width)
        
        # Agents
        labels = ["YOU", "AI"]
        for i, agent in enumerate(agents):
            if not agent["alive"]:
                continue
            
            sx, sy = self.world_to_screen(agent["pos"][0], agent["pos"][1])
            team_color = TEAM_COLORS[i]
            
            # Body
            radius = int(self.scale * 0.5)
            pygame.draw.circle(self.screen, team_color, (sx, sy), radius)
            pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
            
            # Facing direction
            angle = agent.get("angle", 0)
            dir_len = self.scale * 0.7
            end_x = sx + math.cos(angle) * dir_len
            end_y = sy - math.sin(angle) * dir_len
            pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 3)
            
            # Label
            label = self.font.render(labels[i], True, BLACK)
            self.screen.blit(label, (sx - label.get_width()//2, sy + radius + 5))
            
            # Action indicator
            if actions and i < len(actions):
                action = actions[i]
                action_text = self._get_action_symbol(action)
                if action_text:
                    action_label = self.font.render(action_text, True, (255, 255, 255))
                    # Draw action above health bar
                    self.screen.blit(action_label, (sx - action_label.get_width()//2, sy - radius - 28))
            
            # Observation display
            if observations and i < len(observations):
                self._render_agent_obs(sx, sy, radius, observations[i], agent.get("angle", 0))
            
            # Health bar
            health_pct = agent["health"] / 100.0
            bar_width = self.scale * 1.5
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 12
            pygame.draw.rect(self.screen, BLACK, (bar_x - 1, bar_y - 1, bar_width + 2, 8))
            health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), 6))
        
        # Info panel
        self._render_info_panel(agents, step, player_score, ai_score)
        
        pygame.display.flip()
    
    def _get_action_symbol(self, action):
        """Convert action number to display symbol."""
        symbols = {
            0: "",       # Nothing
            1: "↑",      # Up
            2: "↓",      # Down
            3: "←",      # Left
            4: "→",      # Right
            5: "↺",      # Rotate left
            6: "↻",      # Rotate right
            7: "🔥",     # Shoot
        }
        return symbols.get(action, "")
    
    def _render_agent_obs(self, sx, sy, radius, obs, agent_angle):
        """Render observation values around the agent.
        
        Observation indices (14 dims):
        0: cos_delta_enemy (1.0 = facing enemy)
        1: sin_delta_enemy (sign = turn direction)
        2: dist_enemy_norm
        3: has_los (line of sight)
        4: cooldown_norm
        5: would_hit
        6-13: ray distances (8 rays)
        """
        tiny_font = pygame.font.Font(None, 16)
        
        # Draw ray sensors as lines from agent
        ray_length = self.scale * 2
        for i in range(8):
            # Ray angles from -π/2 to +π/2 around agent's heading
            t = i / 7
            local_angle = -math.pi / 2 + t * math.pi
            global_angle = agent_angle + local_angle
            
            # Wall ray distance (indices 6-13)
            wall_dist = obs[6 + i]  # 0-1 normalized
            
            # Calculate ray endpoint
            ray_end_x = sx + math.cos(global_angle) * ray_length * wall_dist
            ray_end_y = sy - math.sin(global_angle) * ray_length * wall_dist
            
            pygame.draw.line(self.screen, (100, 100, 255), (sx, sy), (int(ray_end_x), int(ray_end_y)), 1)
        
        # Draw enemy direction indicator if has LOS
        has_los = obs[3] > 0.5
        cos_delta = obs[0]
        sin_delta = obs[1]
        dist_norm = obs[2]
        would_hit = obs[5] > 0.5
        
        if has_los and dist_norm < 0.99:
            # Calculate world angle to enemy
            enemy_angle = agent_angle + math.atan2(sin_delta, cos_delta)
            indicator_len = self.scale * 1.5
            
            ex = sx + math.cos(enemy_angle) * indicator_len
            ey = sy - math.sin(enemy_angle) * indicator_len
            
            # Green line pointing to enemy
            pygame.draw.line(self.screen, (0, 255, 0), (sx, sy), (int(ex), int(ey)), 2)
            
            # Small circle at end
            pygame.draw.circle(self.screen, (0, 255, 0), (int(ex), int(ey)), 3)
        
        # Show key values as text below agent
        info_y = sy + radius + 22
        
        # LOS indicator (with would-hit highlight)
        if would_hit:
            los_text = "HIT!"
            los_color = (255, 255, 0)
        elif has_los:
            los_text = "LOS"
            los_color = (0, 255, 0)
        else:
            los_text = "---"
            los_color = (150, 150, 150)
        los_label = tiny_font.render(los_text, True, los_color)
        self.screen.blit(los_label, (sx - los_label.get_width()//2, info_y))
        
        # Distance to enemy
        info_y += 12
        dist_text = f"d:{dist_norm:.2f}"
        dist_label = tiny_font.render(dist_text, True, (200, 200, 200))
        self.screen.blit(dist_label, (sx - dist_label.get_width()//2, info_y))
        
        # Aim accuracy (cos_delta: 1.0 = perfect aim)
        info_y += 12
        aim_text = f"aim:{cos_delta:.2f}"
        aim_color = (0, 255, 0) if cos_delta > 0.95 else (255, 255, 0) if cos_delta > 0.7 else (200, 200, 200)
        aim_label = tiny_font.render(aim_text, True, aim_color)
        self.screen.blit(aim_label, (sx - aim_label.get_width()//2, info_y))
    
    def _render_info_panel(self, agents, step, player_score, ai_score):
        """Render the info panel on the right."""
        panel_x = self.game_width
        pygame.draw.rect(self.screen, (40, 40, 50), 
                        (panel_x, 0, self.info_panel_width, self.screen_height))
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (panel_x, 0), (panel_x, self.screen_height), 2)
        
        y = 15
        
        # Title
        title = self.title_font.render("PLAYER vs AI", True, WHITE)
        self.screen.blit(title, (panel_x + 15, y))
        y += 45
        
        # Score
        score_text = self.font.render(f"Score: {player_score} - {ai_score}", True, WHITE)
        self.screen.blit(score_text, (panel_x + 15, y))
        y += 35
        
        # Step
        step_text = self.small_font.render(f"Step: {step}/200", True, (180, 180, 180))
        self.screen.blit(step_text, (panel_x + 15, y))
        y += 30
        
        # Health bars
        pygame.draw.line(self.screen, (80, 80, 80), (panel_x + 10, y), (panel_x + self.info_panel_width - 10, y), 1)
        y += 15
        
        for i, (label, color) in enumerate([("YOU", TEAM_COLORS[0]), ("AI", TEAM_COLORS[1])]):
            health = agents[i]["health"] if agents[i]["alive"] else 0
            
            text = self.font.render(f"{label}: {health:.0f} HP", True, color)
            self.screen.blit(text, (panel_x + 15, y))
            y += 25
            
            # Health bar
            bar_width = self.info_panel_width - 40
            pygame.draw.rect(self.screen, (60, 60, 70), (panel_x + 15, y, bar_width, 12))
            health_pct = health / 100.0
            health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color, (panel_x + 15, y, int(bar_width * health_pct), 12))
            y += 25
        
        # Controls
        y += 20
        pygame.draw.line(self.screen, (80, 80, 80), (panel_x + 10, y), (panel_x + self.info_panel_width - 10, y), 1)
        y += 15
        
        controls_title = self.font.render("Controls", True, WHITE)
        self.screen.blit(controls_title, (panel_x + 15, y))
        y += 28
        
        controls = [
            "W/S - Up/Down",
            "A/D - Left/Right",
            "←/→ - Rotate",
            "Space - Shoot",
            "R - Reset",
            "ESC - Quit",
        ]
        
        for ctrl in controls:
            text = self.small_font.render(ctrl, True, (150, 150, 150))
            self.screen.blit(text, (panel_x + 20, y))
            y += 22
    
    def show_message(self, message, color=WHITE):
        """Show a centered message on screen."""
        text = self.title_font.render(message, True, color)
        rect = text.get_rect(center=(self.game_width // 2, self.game_height // 2))
        
        # Background box
        padding = 20
        bg_rect = rect.inflate(padding * 2, padding * 2)
        pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)
        pygame.draw.rect(self.screen, color, bg_rect, 3)
        
        self.screen.blit(text, rect)
        pygame.display.flip()
    
    def close(self):
        pygame.quit()


def get_player_action(keys):
    """Convert keyboard state to action.
    
    Actions: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_left, 6=rotate_right, 7=shoot
    """
    if keys[pygame.K_SPACE]:
        return 7  # Shoot
    if keys[pygame.K_LEFT]:
        return 5  # Rotate left
    if keys[pygame.K_RIGHT]:
        return 6  # Rotate right
    if keys[pygame.K_w]:
        return 1  # Up
    if keys[pygame.K_s]:
        return 2  # Down
    if keys[pygame.K_a]:
        return 3  # Left
    if keys[pygame.K_d]:
        return 4  # Right
    return 0  # Nothing


def play_vs_agent(model_path=None, fps=10):
    """Main game loop for playing against an AI agent."""
    
    # Load AI model
    if model_path is None:
        model_path = find_best_agent()
    
    print(f"Loading AI from: {model_path}")
    ai_model = PPO.load(model_path)
    
    # Create environment
    env = Worms3DPettingZooEnv(render_mode=None)
    unwrapped = env.unwrapped
    
    # Create renderer
    renderer = GameRenderer(
        map_size=unwrapped.SIZE,
        scale=20,
        obstacles=unwrapped.OBSTACLES
    )
    
    player_score = 0
    ai_score = 0
    
    print("\n" + "="*50)
    print("PLAYER vs AI - Worms 3D")
    print("="*50)
    print("Controls: WASD=move, Arrows=rotate, Space=shoot")
    print("="*50 + "\n")
    
    running = True
    while running:
        # Reset episode
        obs, _ = env.reset()
        step = 0
        episode_done = False
        
        # Randomize which side player starts on
        player_is_agent_0 = random.choice([True, False])
        player_idx = 0 if player_is_agent_0 else 1
        ai_idx = 1 if player_is_agent_0 else 0
        side = "left" if player_is_agent_0 else "right"
        print(f"You are starting on the {side} side")
        
        while not episode_done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        episode_done = True  # Reset
            
            if not running:
                break
            
            # Get actions
            keys = pygame.key.get_pressed()
            player_action = get_player_action(keys)
            
            # AI action - AI uses whichever agent the player is NOT
            ai_obs_key = f"agent_{ai_idx}"
            ai_action, _ = ai_model.predict(obs[ai_obs_key], deterministic=True)
            
            # Assign actions based on which agent is player/AI
            actions = {
                f"agent_{player_idx}": player_action,
                f"agent_{ai_idx}": int(ai_action)
            }
            
            # Step environment
            obs, rewards, terminations, truncations, infos = env.step(actions)
            step += 1
            
            # Render - reorder agents so player is always shown first
            display_agents = [
                unwrapped.agents[player_idx],
                unwrapped.agents[ai_idx]
            ]
            # Actions for display: [player_action, ai_action]
            display_actions = [player_action, int(ai_action)]
            # Observations for display: [player_obs, ai_obs]
            display_obs = [
                obs[f"agent_{player_idx}"],
                obs[f"agent_{ai_idx}"]
            ]
            renderer.render(
                display_agents, 
                unwrapped.last_shots, 
                step=step,
                player_score=player_score,
                ai_score=ai_score,
                actions=display_actions,
                observations=display_obs
            )
            
            # Check for episode end
            terminated = terminations["agent_0"]
            truncated = truncations["agent_0"]
            
            if terminated or truncated:
                episode_done = True
                
                # Determine winner
                player_alive = unwrapped.agents[player_idx]["alive"]
                ai_alive = unwrapped.agents[ai_idx]["alive"]
                
                if player_alive and not ai_alive:
                    player_score += 1
                    renderer.show_message("YOU WIN!", (80, 255, 80))
                elif ai_alive and not player_alive:
                    ai_score += 1
                    renderer.show_message("AI WINS!", (255, 80, 80))
                else:
                    renderer.show_message("DRAW!", (255, 255, 80))
                
                # Wait a moment
                pygame.time.wait(1500)
            
            renderer.clock.tick(fps)
    
    env.close()
    renderer.close()
    
    print(f"\nFinal Score: Player {player_score} - {ai_score} AI")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against a trained AI agent")
    parser.add_argument("--model", type=str, help="Path to AI model (auto-finds newest if not specified)")
    parser.add_argument("--fps", type=int, default=10, help="Game speed (frames per second)")
    
    args = parser.parse_args()
    
    play_vs_agent(model_path=args.model, fps=args.fps)
