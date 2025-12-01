"""Play zombie survival mode with a trained combat model.

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
RED = (255, 80, 80)
GREEN = (80, 255, 80)
ZOMBIE_COLOR = (100, 150, 100)  # Greenish for zombies


class ZombieRenderer:
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
        pygame.display.set_caption("Zombie Survival")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.big_font = pygame.font.Font(None, 48)
    
    def world_to_screen(self, x, y):
        sx = int(x * self.scale)
        sy = int((self.map_size - 1 - y) * self.scale)
        return sx, sy
    
    def render(self, player, zombies, shots, step, kills, player_dash_cd, powerups=None, speed_boost=0):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None, None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False, None, None
        
        # Get mouse position for aiming
        mouse_x, mouse_y = pygame.mouse.get_pos()
        mouse_buttons = pygame.mouse.get_pressed()
        player_screen_x, player_screen_y = self.world_to_screen(player["pos"][0], player["pos"][1])
        
        dx = mouse_x - player_screen_x
        dy = -(mouse_y - player_screen_y)
        target_angle = math.atan2(dy, dx) if (dx != 0 or dy != 0) else 0
        
        keys = pygame.key.get_pressed()
        action = self._get_player_action(keys, mouse_buttons)
        
        # Clear screen
        self.screen.fill((40, 50, 40))  # Dark green-ish for zombie theme
        
        # Draw grid
        for x in range(0, self.game_width, self.scale * 5):
            pygame.draw.line(self.screen, (60, 70, 60), (x, 0), (x, self.game_height), 1)
        for y in range(0, self.game_height, self.scale * 5):
            pygame.draw.line(self.screen, (60, 70, 60), (0, y), (self.game_width, y), 1)
        
        # Draw obstacles
        for obs in self.obstacles:
            ox1, oy1 = self.world_to_screen(obs[0], obs[3])
            ox2, oy2 = self.world_to_screen(obs[2], obs[1])
            rect = pygame.Rect(ox1, oy1, ox2 - ox1, oy2 - oy1)
            pygame.draw.rect(self.screen, (80, 60, 40), rect)
            pygame.draw.rect(self.screen, BLACK, rect, 3)
        
        # Draw shots
        if shots:
            for shot in shots:
                ox, oy = self.world_to_screen(shot["origin"][0], shot["origin"][1])
                sdx, sdy = shot["dx"], -shot["dy"]
                end_x = ox + sdx * self.screen_width
                end_y = oy + sdy * self.screen_height
                
                color = (255, 255, 0) if shot["hit"] else (255, 100, 100)
                width = 3 if shot["hit"] else 2
                pygame.draw.line(self.screen, color, (ox, oy), (int(end_x), int(end_y)), width)
        
        # Draw powerups
        if powerups:
            for powerup in powerups:
                if not powerup["active"]:
                    continue
                sx, sy = self.world_to_screen(powerup["pos"][0], powerup["pos"][1])
                # Yellow diamond for speed boost
                size = int(self.scale * 0.4)
                points = [(sx, sy - size), (sx + size, sy), (sx, sy + size), (sx - size, sy)]
                pygame.draw.polygon(self.screen, (255, 255, 0), points)
                pygame.draw.polygon(self.screen, (200, 200, 0), points, 2)
        
        # Draw zombies
        for zombie in zombies:
            if not zombie["alive"]:
                continue
            
            sx, sy = self.world_to_screen(zombie["pos"][0], zombie["pos"][1])
            radius = int(self.scale * 0.4)
            
            # Zombie body
            pygame.draw.circle(self.screen, ZOMBIE_COLOR, (sx, sy), radius)
            pygame.draw.circle(self.screen, (50, 80, 50), (sx, sy), radius, 2)
            
            # Facing direction
            angle = zombie.get("angle", 0)
            dir_len = self.scale * 0.5
            end_x = sx + math.cos(angle) * dir_len
            end_y = sy - math.sin(angle) * dir_len
            pygame.draw.line(self.screen, (50, 80, 50), (sx, sy), (int(end_x), int(end_y)), 2)
            
            # Health bar
            health_pct = zombie["health"] / 50.0
            bar_width = self.scale
            bar_height = 4
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 8
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
            pygame.draw.rect(self.screen, (200, 50, 50), (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        
        # Draw player
        if player["alive"]:
            sx, sy = self.world_to_screen(player["pos"][0], player["pos"][1])
            radius = int(self.scale * 0.5)
            
            # Glow effect when speed boosted
            if speed_boost > 0:
                glow_radius = radius + 5
                pygame.draw.circle(self.screen, (255, 255, 100), (sx, sy), glow_radius)
            
            pygame.draw.circle(self.screen, RED, (sx, sy), radius)
            pygame.draw.circle(self.screen, BLACK, (sx, sy), radius, 2)
            
            # Use target angle for display (where mouse is pointing)
            dir_len = self.scale * 0.7
            end_x = sx + math.cos(target_angle) * dir_len
            end_y = sy - math.sin(target_angle) * dir_len
            pygame.draw.line(self.screen, BLACK, (sx, sy), (int(end_x), int(end_y)), 3)
            
            # Health bar
            health_pct = player["health"] / 100.0
            bar_width = self.scale * 1.5
            bar_height = 6
            bar_x = sx - bar_width // 2
            bar_y = sy - radius - 12
            pygame.draw.rect(self.screen, BLACK, (bar_x, bar_y, bar_width, bar_height))
            health_color = GREEN if health_pct > 0.5 else (255, 255, 0) if health_pct > 0.25 else (255, 0, 0)
            pygame.draw.rect(self.screen, health_color, (bar_x, bar_y, int(bar_width * health_pct), bar_height))
        
        # Draw HUD
        hud_y = self.game_height
        pygame.draw.rect(self.screen, (30, 30, 35), (0, hud_y, self.screen_width, self.hud_height))
        
        # Health
        health_text = self.font.render(f"HP: {int(player['health'])}", True, RED)
        self.screen.blit(health_text, (20, hud_y + 10))
        
        # Kills
        kills_text = self.big_font.render(f"KILLS: {kills}", True, GREEN)
        self.screen.blit(kills_text, (self.screen_width // 2 - 60, hud_y + 15))
        
        # Dash cooldown
        dash_text = f"DASH: {'READY' if player_dash_cd <= 0 else f'{player_dash_cd}'}"
        dash_color = GREEN if player_dash_cd <= 0 else (255, 100, 100)
        text = self.font.render(dash_text, True, dash_color)
        self.screen.blit(text, (20, hud_y + 40))
        
        # Speed boost indicator
        if speed_boost > 0:
            boost_text = self.font.render(f"SPEED: {speed_boost}", True, (255, 255, 0))
            self.screen.blit(boost_text, (120, hud_y + 40))
        
        # Zombie count
        alive_zombies = sum(1 for z in zombies if z["alive"])
        zombie_text = self.font.render(f"Zombies: {alive_zombies}", True, ZOMBIE_COLOR)
        self.screen.blit(zombie_text, (self.screen_width - 120, hud_y + 10))
        
        # Controls
        controls = "WASD:Move  MOUSE:Aim  LMB:Shoot  RMB:Dash"
        hint = self.small_font.render(controls, True, (120, 120, 120))
        self.screen.blit(hint, (self.screen_width // 2 - 130, hud_y + 55))
        
        pygame.display.flip()
        self.clock.tick(30)
        
        return True, action, target_angle
    
    def _get_player_action(self, keys, mouse_buttons):
        if keys[pygame.K_SPACE] or mouse_buttons[0]:
            return 7
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or mouse_buttons[2]:
            return 8
        if keys[pygame.K_w]:
            return 1
        if keys[pygame.K_s]:
            return 2
        if keys[pygame.K_a]:
            return 3
        if keys[pygame.K_d]:
            return 4
        return 0
    
    def show_game_over(self, kills):
        overlay = pygame.Surface((self.screen_width, self.screen_height))
        overlay.set_alpha(200)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        title = self.big_font.render("GAME OVER", True, RED)
        title_rect = title.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 50))
        self.screen.blit(title, title_rect)
        
        score = self.big_font.render(f"Kills: {kills}", True, GREEN)
        score_rect = score.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 10))
        self.screen.blit(score, score_rect)
        
        hint = self.font.render("Press SPACE to play again, ESC to quit", True, WHITE)
        hint_rect = hint.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 60))
        self.screen.blit(hint, hint_rect)
        
        pygame.display.flip()
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_SPACE:
                        return True
            self.clock.tick(30)
    
    def close(self):
        pygame.quit()


def main():
    # Create environment
    env = gym.make("ZombieSurvival-v0")
    unwrapped = env.unwrapped
    
    # Create renderer
    renderer = ZombieRenderer(
        map_size=unwrapped.SIZE,
        scale=20,
        obstacles=unwrapped.OBSTACLES
    )
    
    print("\n=== ZOMBIE SURVIVAL ===")
    print("Controls: WASD=Move, MOUSE=Aim, LMB=Shoot, RMB=Dash, ESC=Quit")
    print("Survive as long as you can!\n")
    
    running = True
    while running:
        obs, _ = env.reset()
        done = False
        
        while not done:
            player = unwrapped.agent
            player_dash_cd = player.get("dash_cooldown", 0)
            speed_boost = player.get("speed_boost", 0)
            
            cont, action, target_angle = renderer.render(
                player,
                unwrapped.zombies,
                unwrapped.last_shots,
                unwrapped.current_step,
                unwrapped.kills,
                player_dash_cd,
                unwrapped.powerups,
                speed_boost
            )
            
            if not cont:
                running = False
                break
            
            # Set player angle to face mouse
            player["angle"] = target_angle
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        if running:
            running = renderer.show_game_over(unwrapped.kills)
    
    renderer.close()
    env.close()


def run_with_ai():
    """Run with trained AI controlling the player."""
    model_path = get_newest_model()
    
    if not model_path:
        print("No model found! Run main() for manual play.")
        return
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    # Need frame stacking to match training
    env = DummyVecEnv([lambda: gym.make("ZombieSurvival-v0")])
    env = VecFrameStack(env, n_stack=4)
    unwrapped = env.envs[0].unwrapped
    
    renderer = ZombieRenderer(
        map_size=unwrapped.SIZE,
        scale=20,
        obstacles=unwrapped.OBSTACLES
    )
    
    print("\n=== ZOMBIE SURVIVAL (AI Mode) ===")
    print("Watching trained agent fight zombies...\n")
    
    obs = env.reset()
    done = False
    
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                done = True
                break
        
        if done:
            break
        
        # AI predicts action
        action, _ = model.predict(obs, deterministic=True)
        action = int(action.flatten()[0])  # Single agent action
        
        player = unwrapped.agent
        
        # Render
        renderer.render(
            player,
            unwrapped.zombies,
            unwrapped.last_shots,
            unwrapped.current_step,
            unwrapped.kills,
            player.get("dash_cooldown", 0)
        )
        
        obs, reward, done_arr, info = env.step([action])
        done = done_arr[0]
    
    if unwrapped.kills > 0:
        renderer.show_game_over(unwrapped.kills)
    
    renderer.close()
    env.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--ai":
        run_with_ai()
    else:
        main()
