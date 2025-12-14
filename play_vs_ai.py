"""Play against the latest trained AI agent.

Controls:
    W/S/A/D - Move up/down/left/right
    MOUSE   - Aim (player faces cursor)
    SPACE/LMB - Shoot
    SHIFT/RMB - Dash
    ESC     - Quit
"""
import sys
import os
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame
import numpy as np
import math
import gymnasium as gym
import worms_3d_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


def get_newest_model(models_dir="models"):
    """Find the newest .zip model file in the models directory."""
    pattern = os.path.join(models_dir, "**", "*.zip")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        return None
    
    newest = max(model_files, key=os.path.getmtime)
    return newest


# Colors
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
TEAM_COLORS = [
    (255, 80, 80),   # Red team (player)
    (80, 80, 255),   # Blue team (AI)
]


class GameRenderer:
    def __init__(self, map_size, scale=20, obstacles=None):
        self.map_size = map_size
        self.scale = scale
        self.obstacles = obstacles or []
        
        self.game_width = map_size * scale
        self.game_height = map_size * scale
        self.hud_height = 80
        self.screen_width = self.game_width
        self.screen_height = self.game_height + self.hud_height
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Worms 2D - Play vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
    
    def world_to_screen(self, x, y):
        sx = int(x * self.scale)
        sy = int((self.map_size - 1 - y) * self.scale)
        return sx, sy
    
    def screen_to_world(self, sx, sy):
        """Convert screen coordinates to world coordinates."""
        x = sx / self.scale
        y = (self.map_size - 1) - (sy / self.scale)
        return x, y
    
    def render(self, agents, shots=None, step=0, player_dash_cd=0):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None, None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, None, None
        
        # Get mouse position and calculate target angle for player
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        player_pos = agents[0]["pos"]
        player_screen_x, player_screen_y = self.world_to_screen(player_pos[0], player_pos[1])
        
        # Calculate angle from player to mouse (in world coords)
        dx = mouse_x - player_screen_x
        dy = -(mouse_y - player_screen_y)  # Flip Y for world coords
        target_angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
        
        # Get player input
        keys = pygame.key.get_pressed()
        action = self._get_player_action(keys, mouse_buttons)
        
        # Clear screen
        self.screen.fill((210, 180, 140))
        
        # Draw grid
        for x in range(0, self.game_width, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.game_height), 1)
        for y in range(0, self.game_height, self.scale * 5):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.game_width, y), 1)
        
        # Draw obstacles
        for obs in self.obstacles:
            ox1, oy1 = self.world_to_screen(obs[0], obs[3])
            ox2, oy2 = self.world_to_screen(obs[2], obs[1])
            rect = pygame.Rect(ox1, oy1, ox2 - ox1, oy2 - oy1)
            pygame.draw.rect(self.screen, (101, 67, 33), rect)
            pygame.draw.rect(self.screen, BLACK, rect, 3)
        
        # Draw shots
        if shots:
            for shot in shots:
                ox, oy = self.world_to_screen(shot["origin"][0], shot["origin"][1])
                dx, dy = shot["dx"], -shot["dy"]
                end_x = ox + dx * self.screen_width
                end_y = oy + dy * self.screen_height
                
                color = (255, 255, 0) if shot["hit"] else TEAM_COLORS[shot["team"]]
                width = 3 if shot["hit"] else 2
                pygame.draw.line(self.screen, color, (ox, oy), (int(end_x), int(end_y)), width)
        
        # Draw agents
        for agent in agents:
            if not agent["alive"]:
                continue
            
            pos = agent["pos"]
            sx, sy = self.world_to_screen(pos[0], pos[1])
            team_color = TEAM_COLORS[agent["team"]]
            
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
            label = "YOU" if agent["team"] == 0 else "AI"
            text = self.font.render(label, True, BLACK)
            self.screen.blit(text, (sx - 15, sy + radius + 5))
            
            # Health bar
            health_pct = agent["health"] / 100.0
            bar_width = self.scale * 1.5
            bar_height = 6
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 12
            
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
            health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        
        # Draw HUD
        hud_y = self.game_height
        pygame.draw.rect(self.screen, (40, 40, 50), (0, hud_y, self.screen_width, self.hud_height))
        
        # Player stats
        player = agents[0]
        ai = agents[1]
        
        # Health display
        self._draw_health_bar(20, hud_y + 15, "YOU", player["health"], TEAM_COLORS[0])
        self._draw_health_bar(self.screen_width - 170, hud_y + 15, "AI", ai["health"], TEAM_COLORS[1])
        
        # Dash cooldown
        dash_text = f"DASH: {'READY' if player_dash_cd <= 0 else f'{player_dash_cd}'}"
        dash_color = (0, 255, 0) if player_dash_cd <= 0 else (255, 100, 100)
        text = self.font.render(dash_text, True, dash_color)
        self.screen.blit(text, (20, hud_y + 50))
        
        # Step counter
        step_text = self.font.render(f"Step: {step}", True, WHITE)
        self.screen.blit(step_text, (self.screen_width // 2 - 30, hud_y + 15))
        
        # Controls hint
        controls = "WASD:Move  MOUSE:Aim  SPACE/LMB:Shoot  SHIFT/RMB:Dash"
        hint = self.small_font.render(controls, True, (150, 150, 150))
        self.screen.blit(hint, (self.screen_width // 2 - 150, hud_y + 55))
        
        pygame.display.flip()
        self.clock.tick(30)
        
        return True, action, target_angle
    
    def _draw_health_bar(self, x, y, label, health, color):
        text = self.font.render(f"{label}: {int(health)}", True, color)
        self.screen.blit(text, (x, y))
        
        bar_width = 120
        bar_height = 12
        bar_y = y + 22
        
        pygame.draw.rect(self.screen, (60, 60, 70), (x, bar_y, bar_width, bar_height))
        health_pct = health / 100.0
        health_color = (0, 255, 0) if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
        pygame.draw.rect(self.screen, health_color, (x, bar_y, int(bar_width * health_pct), bar_height))
    
    def _get_player_action(self, keys, mouse_buttons):
        # Actions: 0=nothing, 1=up, 2=down, 3=left, 4=right, 5=rotate_left, 6=rotate_right, 7=shoot, 8=dash
        # Mouse: LMB=shoot, RMB=dash
        if keys[pygame.K_SPACE] or mouse_buttons[0]:  # LMB
            return 7  # Shoot
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or mouse_buttons[2]:  # RMB
            return 8  # Dash
        if keys[pygame.K_w]:
            return 1  # Up
        if keys[pygame.K_s]:
            return 2  # Down
        if keys[pygame.K_a]:
            return 3  # Left
        if keys[pygame.K_d]:
            return 4  # Right
        return 0  # Nothing
    
    def show_result(self, winner):
        # Show result overlay
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        if winner == 0:
            text = "YOU WIN!"
            color = (0, 255, 0)
        elif winner == 1:
            text = "AI WINS!"
            color = (255, 0, 0)
        else:
            text = "DRAW!"
            color = (255, 255, 0)
        
        big_font = pygame.font.Font(None, 72)
        result = big_font.render(text, True, color)
        rect = result.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 30))
        self.screen.blit(result, rect)
        
        hint = self.font.render("Press SPACE to play again, ESC to quit", True, WHITE)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 30))
        self.screen.blit(hint, hint_rect)
        
        pygame.display.flip()
        
        # Wait for input
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_SPACE:
                        return True
            self.clock.tick(30)
        return False
    
    def close(self):
        pygame.quit()


def main():
    # Load model
    model_path = get_newest_model()
    model = None
    
    if model_path:
        print(f"Loading AI model: {model_path}")
        model = PPO.load(model_path)
    else:
        print("No model found! AI will take random actions.")
    
    # Create environment with frame stacking (to match training)
    env = DummyVecEnv([lambda: gym.make("Worms3D-v0")])
    env = VecFrameStack(env, n_stack=4)
    unwrapped = env.envs[0].unwrapped
    
    # Create renderer
    renderer = GameRenderer(
        map_size=unwrapped.SIZE,
        scale=20,
        obstacles=unwrapped.OBSTACLES
    )
    
    print("\n=== PLAY VS AI ===")
    print("Controls: WASD=Move, MOUSE=Aim, LMB/SPACE=Shoot, RMB/SHIFT=Dash, ESC=Quit\n")
    
    running = True
    while running:
        obs = env.reset()
        step = 0
        done = False
        
        while not done:
            # Get player dash cooldown for HUD
            player_dash_cd = unwrapped.agents[0].get("dash_cooldown", 0)
            
            # Render and get player input
            cont, player_action, target_angle = renderer.render(
                unwrapped.agents, 
                unwrapped.last_shots, 
                step,
                player_dash_cd
            )
            
            if not cont:
                running = False
                break
            
            # Set player angle to face mouse cursor
            unwrapped.agents[0]["angle"] = target_angle
            
            # Get AI action
            if model:
                ai_action, _ = model.predict(obs, deterministic=True)
                ai_action = int(ai_action.flatten()[1])  # Agent 1's action
            else:
                ai_action = np.random.randint(0, 9)
            
            # Combine actions: [player, AI]
            actions = np.array([[player_action, ai_action]])
            
            obs, reward, done_arr, info = env.step(actions)
            done = done_arr[0]
            step += 1
        
        if running:
            # Determine winner
            if unwrapped.agents[0]["alive"] and not unwrapped.agents[1]["alive"]:
                winner = 0  # Player wins
            elif unwrapped.agents[1]["alive"] and not unwrapped.agents[0]["alive"]:
                winner = 1  # AI wins
            else:
                winner = -1  # Draw
            
            running = renderer.show_result(winner)
    
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
