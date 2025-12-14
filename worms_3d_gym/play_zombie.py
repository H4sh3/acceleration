#!/usr/bin/env python3
"""Play the Zombie Survival environment with keyboard controls.

Controls:
    W/Up    - Set move up
    S/Down  - Set move down
    A/Left  - Set move left
    D/Right - Set move right
    Q/E     - Rotate left/right (fine, ±1°)
    Z/C     - Rotate left/right (less fine, ±5°)
    X/V     - Rotate left/right (coarse, ±45°)
    Space   - Shoot
    R       - Reset
    Escape  - Quit
"""
import sys
import math
import pygame
import numpy as np

sys.path.insert(0, "/home/h4sh3/code/acceleration")
from worms_3d_gym.envs.zombie_env import ZombieSurvivalEnv


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
DARK_GREEN = (0, 100, 0)
ORANGE = (255, 165, 0)


class ZombieGameRenderer:
    def __init__(self, env, scale=15, view_size=40):
        self.env = env
        self.scale = scale
        self.view_size = view_size  # World units visible around agent
        self.width = int(view_size * scale)
        self.height = int(view_size * scale)
        
        pygame.init()
        pygame.display.set_caption("Zombie Survival - Infinite World")
        self.screen = pygame.display.set_mode((self.width, self.height + 80))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 48)
    
    def world_to_screen(self, pos):
        """Convert world position to screen position (agent-centered camera)."""
        agent_pos = self.env.agent["pos"]
        # Relative to agent, then center on screen
        rel_x = pos[0] - agent_pos[0]
        rel_y = pos[1] - agent_pos[1]
        screen_x = int(self.width / 2 + rel_x * self.scale)
        screen_y = int(self.height / 2 - rel_y * self.scale)  # Flip Y
        return (screen_x, screen_y)
    
    def draw(self):
        self.screen.fill(DARK_GREEN)
        
        agent = self.env.agent
        agent_pos = agent["pos"]
        
        # Draw grid (relative to agent position)
        grid_spacing = 5
        # Calculate grid offset based on agent position
        offset_x = agent_pos[0] % grid_spacing
        offset_y = agent_pos[1] % grid_spacing
        
        for i in range(-self.view_size // grid_spacing - 1, self.view_size // grid_spacing + 2):
            # Vertical lines
            world_x = i * grid_spacing - offset_x + agent_pos[0]
            screen_pos = self.world_to_screen(np.array([world_x, agent_pos[1]]))
            pygame.draw.line(self.screen, (0, 80, 0), 
                           (screen_pos[0], 0), (screen_pos[0], self.height), 1)
            # Horizontal lines
            world_y = i * grid_spacing - offset_y + agent_pos[1]
            screen_pos = self.world_to_screen(np.array([agent_pos[0], world_y]))
            pygame.draw.line(self.screen, (0, 80, 0), 
                           (0, screen_pos[1]), (self.width, screen_pos[1]), 1)
        
        # Draw fixed reference markers at origin and cardinal points
        # These help visualize agent movement in the infinite world
        origin_screen = self.world_to_screen(np.array([0.0, 0.0]))
        if 0 <= origin_screen[0] <= self.width and 0 <= origin_screen[1] <= self.height:
            pygame.draw.circle(self.screen, WHITE, origin_screen, 8, 2)
            pygame.draw.line(self.screen, WHITE, (origin_screen[0]-12, origin_screen[1]), (origin_screen[0]+12, origin_screen[1]), 2)
            pygame.draw.line(self.screen, WHITE, (origin_screen[0], origin_screen[1]-12), (origin_screen[0], origin_screen[1]+12), 2)
        
        # Draw reference markers every 10 units
        for rx in range(-50, 51, 10):
            for ry in range(-50, 51, 10):
                if rx == 0 and ry == 0:
                    continue  # Skip origin, already drawn
                marker_pos = self.world_to_screen(np.array([float(rx), float(ry)]))
                if 0 <= marker_pos[0] <= self.width and 0 <= marker_pos[1] <= self.height:
                    pygame.draw.circle(self.screen, (80, 80, 80), marker_pos, 3)
        
        # Draw aiming range circle
        center = (self.width // 2, self.height // 2)
        range_radius = int(7.0 * self.scale)  # PROJECTILE_MAX_RANGE
        pygame.draw.circle(self.screen, (0, 60, 0), center, range_radius, 1)
        
        # Draw zombies
        for zombie in self.env.zombies:
            if not zombie["alive"]:
                continue
            pos = self.world_to_screen(zombie["pos"])
            # Only draw if on screen
            if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                pygame.draw.circle(self.screen, RED, pos, 12)
                # Health bar
                health_pct = zombie["health"] / 50
                bar_width = 20
                pygame.draw.rect(self.screen, BLACK, (pos[0] - bar_width//2, pos[1] - 20, bar_width, 4))
                pygame.draw.rect(self.screen, RED, (pos[0] - bar_width//2, pos[1] - 20, int(bar_width * health_pct), 4))
        
        # Draw projectiles
        for proj in self.env.projectiles:
            if proj["active"]:
                pos = self.world_to_screen(proj["pos"])
                if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                    pygame.draw.circle(self.screen, YELLOW, pos, 4)
        
        # Draw agent (always at center)
        if agent["alive"]:
            pos = center
            pygame.draw.circle(self.screen, BLUE, pos, 10)
            
            # Draw direction indicator
            angle = agent["angle"]
            end_x = pos[0] + int(20 * math.cos(angle))
            end_y = pos[1] - int(20 * math.sin(angle))  # Flip Y
            pygame.draw.line(self.screen, WHITE, pos, (end_x, end_y), 3)
            
            # Draw movement direction indicators
            move_len = 15
            toggles = self.env.move_toggles
            if toggles[0]:  # Up
                pygame.draw.line(self.screen, ORANGE, pos, (pos[0], pos[1] - move_len), 2)
            if toggles[1]:  # Down
                pygame.draw.line(self.screen, ORANGE, pos, (pos[0], pos[1] + move_len), 2)
            if toggles[2]:  # Left
                pygame.draw.line(self.screen, ORANGE, pos, (pos[0] - move_len, pos[1]), 2)
            if toggles[3]:  # Right
                pygame.draw.line(self.screen, ORANGE, pos, (pos[0] + move_len, pos[1]), 2)
            
            # Draw aim line
            range_dist = 7.0 * self.scale
            ex = pos[0] + int(range_dist * math.cos(angle))
            ey = pos[1] - int(range_dist * math.sin(angle))
            pygame.draw.line(self.screen, (0, 100, 150), pos, (ex, ey), 2)
        
        # Draw HUD
        hud_y = self.height + 5
        pygame.draw.rect(self.screen, BLACK, (0, self.height, self.width, 80))
        
        # Health bar
        health_pct = max(0, agent["health"] / 100)
        pygame.draw.rect(self.screen, GRAY, (10, hud_y, 200, 20))
        pygame.draw.rect(self.screen, GREEN if health_pct > 0.3 else RED, 
                        (10, hud_y, int(200 * health_pct), 20))
        health_text = self.font.render(f"HP: {int(agent['health'])}", True, WHITE)
        self.screen.blit(health_text, (220, hud_y))
        
        # Kills
        kills_text = self.font.render(f"Kills: {self.env.kills}", True, WHITE)
        self.screen.blit(kills_text, (300, hud_y))
        
        # Projectiles
        proj_text = self.font.render(f"Shots: {10 - len(self.env.projectiles)}/10", True, WHITE)
        self.screen.blit(proj_text, (400, hud_y))
        
        # Position
        pos_text = self.font.render(f"Pos: ({int(agent_pos[0])}, {int(agent_pos[1])})", True, WHITE)
        self.screen.blit(pos_text, (520, hud_y))
        
        # Movement toggles display
        toggle_labels = ["↑", "↓", "←", "→"]
        toggle_x = 10
        toggle_y = hud_y + 28
        for i, (label, active) in enumerate(zip(toggle_labels, self.env.move_toggles)):
            color = ORANGE if active else GRAY
            text = self.font.render(label, True, color)
            self.screen.blit(text, (toggle_x + i * 30, toggle_y))
        
        move_hint = self.font.render("Move:", True, WHITE)
        self.screen.blit(move_hint, (toggle_x + 130, toggle_y))
        
        # Controls hint
        hint = self.font.render("WASD:move QE:1° ZC:5° XV:45° Space:shoot R:reset", True, GRAY)
        self.screen.blit(hint, (10, hud_y + 55))
        
        # Game over
        if not agent["alive"]:
            text = self.big_font.render("GAME OVER - Press R to restart", True, RED)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()


def main():
    env = ZombieSurvivalEnv()
    obs, _ = env.reset()
    
    renderer = ZombieGameRenderer(env)
    running = True
    
    print("Zombie Survival Game")
    print("=" * 40)
    print("Controls:")
    print("  WASD / Arrow keys - Set movement direction")
    print("  Q/E - Fine rotation (±1°)")
    print("  Z/C - Less fine rotation (±5°)")
    print("  X/V - Coarse rotation (±45°)")
    print("  Space - Shoot")
    print("  R - Reset")
    print("  Escape - Quit")
    print("=" * 40)
    
    while running:
        # Discrete action: 0=noop, 1-4=move, 5-10=rotate, 11=shoot
        action = 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    print(f"Reset! Starting new game...")
        
        # Build discrete action from key presses
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            action = 1  # Up
        elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
            action = 2  # Down
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            action = 3  # Left
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            action = 4  # Right
        # Rotation
        elif keys[pygame.K_q]:
            action = 5  # Fine rotate left
        elif keys[pygame.K_e]:
            action = 6  # Fine rotate right
        elif keys[pygame.K_z]:
            action = 7  # Less fine rotate left
        elif keys[pygame.K_c]:
            action = 8  # Less fine rotate right
        elif keys[pygame.K_x]:
            action = 9  # Coarse rotate left
        elif keys[pygame.K_v]:
            action = 10  # Coarse rotate right
        # Shoot
        elif keys[pygame.K_SPACE]:
            action = 11
        
        # Step environment
        if env.agent["alive"]:
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated:
                print(f"Game Over! Final kills: {info['kills']}")
            elif truncated:
                print(f"Time's up! Final kills: {info['kills']}")
        
        renderer.draw()
        renderer.clock.tick(30)
    
    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
