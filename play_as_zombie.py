#!/usr/bin/env python3
"""Play as a zombie against the trained AI agent.

You control a zombie with WASD. Get close to the agent to kill it!

Controls:
    W/Up    - Move up
    S/Down  - Move down  
    A/Left  - Move left
    D/Right - Move right
    R       - Reset
    Escape  - Quit
"""
import sys
import os
import glob
import math
import pygame
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from worms_3d_gym.envs.zombie_env import (
    ZombieSurvivalEnv, MAX_HEALTH, ZOMBIE_ATTACK_RANGE,
    PROJECTILE_MAX_RANGE, ZOMBIE_BASE_SPEED, ZOMBIE_MAX_SPEED,
    N_DIRECTIONS, ACCEL_STEPS_TO_MAX
)


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
PURPLE = (150, 0, 200)
CYAN = (0, 200, 200)


def find_latest_zombie_model(models_dir="models"):
    """Find the latest zombie survival model."""
    patterns = [
        os.path.join(models_dir, "zombie_*", "zombie_final.zip"),
        os.path.join(models_dir, "zombie_*", "zombie_checkpoint_*_steps.zip"),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    
    if not all_files:
        raise FileNotFoundError(f"No zombie models found in {models_dir}")
    
    # Get the newest by modification time
    newest = max(all_files, key=os.path.getmtime)
    print(f"Found model: {newest}")
    return newest


class ZombieVsAgentRenderer:
    """Renderer for playing as zombie against AI agent."""
    
    def __init__(self, env, scale=15, view_size=40):
        self.env = env
        self.scale = scale
        self.view_size = view_size
        self.width = int(view_size * scale)
        self.height = int(view_size * scale)
        
        pygame.init()
        pygame.display.set_caption("Play as Zombie vs AI Agent")
        self.screen = pygame.display.set_mode((self.width, self.height + 100))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.big_font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 32)
    
    def world_to_screen(self, pos, center_pos):
        """Convert world position to screen position (centered on center_pos)."""
        rel_x = pos[0] - center_pos[0]
        rel_y = pos[1] - center_pos[1]
        screen_x = int(self.width / 2 + rel_x * self.scale)
        screen_y = int(self.height / 2 - rel_y * self.scale)
        return (screen_x, screen_y)
    
    def draw(self, player_zombie_pos, agent_alive, player_wins, agent_wins,
             herder_pos=None, herder_in_position=False):
        self.screen.fill(DARK_GREEN)
        
        agent = self.env.agent
        center_pos = player_zombie_pos  # Camera follows player zombie
        
        # Draw grid
        grid_spacing = 5
        offset_x = center_pos[0] % grid_spacing
        offset_y = center_pos[1] % grid_spacing
        
        for i in range(-self.view_size // grid_spacing - 1, self.view_size // grid_spacing + 2):
            world_x = i * grid_spacing - offset_x + center_pos[0]
            screen_pos = self.world_to_screen(np.array([world_x, center_pos[1]]), center_pos)
            pygame.draw.line(self.screen, (0, 80, 0), 
                           (screen_pos[0], 0), (screen_pos[0], self.height), 1)
            world_y = i * grid_spacing - offset_y + center_pos[1]
            screen_pos = self.world_to_screen(np.array([center_pos[0], world_y]), center_pos)
            pygame.draw.line(self.screen, (0, 80, 0), 
                           (0, screen_pos[1]), (self.width, screen_pos[1]), 1)
        
        # Draw attack range circle around player zombie (at center)
        center = (self.width // 2, self.height // 2)
        attack_radius = int(ZOMBIE_ATTACK_RANGE * self.scale)
        pygame.draw.circle(self.screen, (100, 0, 0), center, attack_radius, 2)
        
        # Draw AI zombies (other zombies in the env)
        for zombie in self.env.zombies:
            if not zombie["alive"]:
                continue
            pos = self.world_to_screen(zombie["pos"], center_pos)
            if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                pygame.draw.circle(self.screen, (150, 50, 50), pos, 10)
                # Health bar
                health_pct = zombie["health"] / self.env.zombie_health
                bar_width = 16
                pygame.draw.rect(self.screen, BLACK, (pos[0] - bar_width//2, pos[1] - 18, bar_width, 4))
                pygame.draw.rect(self.screen, RED, (pos[0] - bar_width//2, pos[1] - 18, int(bar_width * health_pct), 4))
        
        # Draw projectiles
        for proj in self.env.projectiles:
            if proj["active"]:
                pos = self.world_to_screen(proj["pos"], center_pos)
                if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                    pygame.draw.circle(self.screen, YELLOW, pos, 5)
        
        # Draw AI agent
        if agent["alive"]:
            pos = self.world_to_screen(agent["pos"], center_pos)
            if 0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height:
                pygame.draw.circle(self.screen, BLUE, pos, 12)
                pygame.draw.circle(self.screen, WHITE, pos, 12, 2)
                
                # Direction indicator
                angle = agent["angle"]
                end_x = pos[0] + int(25 * math.cos(angle))
                end_y = pos[1] - int(25 * math.sin(angle))
                pygame.draw.line(self.screen, WHITE, pos, (end_x, end_y), 3)
                
                # Label
                label = self.font.render("AI", True, WHITE)
                self.screen.blit(label, (pos[0] - label.get_width()//2, pos[1] + 18))
                
                # Health bar
                health_pct = agent["health"] / MAX_HEALTH
                bar_width = 30
                pygame.draw.rect(self.screen, BLACK, (pos[0] - bar_width//2, pos[1] - 22, bar_width, 6))
                pygame.draw.rect(self.screen, GREEN if health_pct > 0.3 else RED, 
                               (pos[0] - bar_width//2, pos[1] - 22, int(bar_width * health_pct), 6))
        
        # Draw player zombie (always at center)
        pygame.draw.circle(self.screen, PURPLE, center, 14)
        pygame.draw.circle(self.screen, WHITE, center, 14, 2)
        label = self.font.render("YOU", True, WHITE)
        self.screen.blit(label, (center[0] - label.get_width()//2, center[1] + 20))
        
        # Draw herder entity
        if herder_pos is not None:
            herder_screen = self.world_to_screen(herder_pos, center_pos)
            if 0 <= herder_screen[0] <= self.width and 0 <= herder_screen[1] <= self.height:
                # Color changes when in position (ready to close in)
                herder_color = GREEN if herder_in_position else CYAN
                pygame.draw.circle(self.screen, herder_color, herder_screen, 12)
                pygame.draw.circle(self.screen, WHITE, herder_screen, 12, 2)
                # Draw attack range when in position
                if herder_in_position:
                    attack_radius = int(ZOMBIE_ATTACK_RANGE * self.scale)
                    pygame.draw.circle(self.screen, (0, 150, 0), herder_screen, attack_radius, 1)
                label = self.font.render("HERDER", True, WHITE)
                self.screen.blit(label, (herder_screen[0] - label.get_width()//2, herder_screen[1] + 18))
        
        # Draw distance to agent
        if agent["alive"]:
            dist = np.linalg.norm(agent["pos"] - player_zombie_pos)
            dist_text = self.font.render(f"Distance: {dist:.1f}", True, WHITE)
            self.screen.blit(dist_text, (10, 10))
            
            # Arrow pointing to agent
            to_agent = agent["pos"] - player_zombie_pos
            if np.linalg.norm(to_agent) > 0:
                to_agent = to_agent / np.linalg.norm(to_agent)
                arrow_len = 40
                arrow_x = center[0] + int(to_agent[0] * arrow_len)
                arrow_y = center[1] - int(to_agent[1] * arrow_len)
                pygame.draw.line(self.screen, YELLOW, center, (arrow_x, arrow_y), 3)
        
        # HUD
        hud_y = self.height + 5
        pygame.draw.rect(self.screen, BLACK, (0, self.height, self.width, 100))
        
        # Title
        title = self.title_font.render("🧟 YOU ARE THE ZOMBIE 🧟", True, PURPLE)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, hud_y))
        
        # Score
        score_text = self.font.render(f"Wins - You: {player_wins}  Agent: {agent_wins}", True, WHITE)
        self.screen.blit(score_text, (10, hud_y + 30))
        
        # Agent status
        status = "ALIVE" if agent["alive"] else "DEAD"
        status_color = GREEN if agent["alive"] else RED
        status_text = self.font.render(f"Agent: {status}", True, status_color)
        self.screen.blit(status_text, (10, hud_y + 55))
        
        # Controls hint
        hint = self.font.render("WASD: move | Get close to kill! | R: reset | ESC: quit", True, GRAY)
        self.screen.blit(hint, (10, hud_y + 78))
        
        # Victory/defeat message
        if not agent["alive"]:
            text = self.big_font.render("YOU KILLED THE AGENT!", True, GREEN)
            text_rect = text.get_rect(center=(self.width // 2, self.height // 2))
            pygame.draw.rect(self.screen, BLACK, text_rect.inflate(20, 10))
            self.screen.blit(text, text_rect)
        
        pygame.display.flip()
    
    def close(self):
        pygame.quit()


def make_env():
    """Create environment for AI agent."""
    def _init():
        return ZombieSurvivalEnv()
    return _init


def main(model_path=None, fps=30):
    # Load AI model
    if model_path is None:
        model_path = find_latest_zombie_model()
    
    print(f"Loading AI agent from: {model_path}")
    
    # Create vectorized env for the AI (it expects frame stacking)
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecFrameStack(vec_env, n_stack=8)
    
    ai_model = PPO.load(model_path)
    
    # Create a separate env instance for the game
    game_env = ZombieSurvivalEnv()
    
    renderer = ZombieVsAgentRenderer(game_env)
    
    player_wins = 0
    agent_wins = 0
    
    print("\n" + "=" * 50)
    print("PLAY AS ZOMBIE vs AI AGENT")
    print("=" * 50)
    print("You are the ZOMBIE! Use WASD to move.")
    print("Get close to the agent to kill it!")
    print("The agent will try to shoot you and dodge.")
    print("=" * 50 + "\n")
    
    running = True
    while running:
        # Reset game
        obs, _ = game_env.reset()
        
        # Reset frame stack for AI
        vec_obs = vec_env.reset()
        
        # Player zombie starts near the agent
        agent_pos = game_env.agent["pos"].copy()
        # Spawn player zombie at a random position around the agent
        angle = np.random.uniform(0, 2 * np.pi)
        spawn_dist = np.random.uniform(8, 12)
        player_zombie_pos = agent_pos + np.array([
            np.cos(angle) * spawn_dist,
            np.sin(angle) * spawn_dist
        ])
        
        # Herder entity - starts far from agent, will position itself strategically
        herder_angle = np.random.uniform(0, 2 * np.pi)
        herder_spawn_dist = np.random.uniform(20, 30)
        herder_pos = agent_pos + np.array([
            np.cos(herder_angle) * herder_spawn_dist,
            np.sin(herder_angle) * herder_spawn_dist
        ])
        herder_speed = ZOMBIE_BASE_SPEED * 1.2
        herder_in_position = False
        
        # Create herder zombie entry so agent can see it
        herder_zombie = {
            "id": -2,
            "team": 1,
            "pos": herder_pos,
            "health": 9999,  # Invincible
            "angle": 0.0,
            "alive": True,
            "velocity": 0.0,
            "move_dir": -1,
            "accel_steps": 0
        }
        
        # Spawn many AI zombies around the agent
        game_env.min_zombies = 10  # Start with 10 zombies
        for _ in range(10):
            game_env._spawn_zombie()
        
        # Player zombie acceleration state
        player_velocity = 0.0
        player_move_dir = -1
        player_accel_steps = 0
        player_base_speed = ZOMBIE_BASE_SPEED * 1.5  # Slightly faster than AI zombies
        player_max_speed = ZOMBIE_MAX_SPEED * 1.5
        
        # Create a fake zombie entry for the player so the agent can "see" us
        player_zombie = {
            "id": -1,
            "team": 1,
            "pos": player_zombie_pos,
            "health": 9999,  # Invincible
            "angle": 0.0,
            "alive": True,
            "velocity": 0.0,
            "move_dir": -1,
            "accel_steps": 0
        }
        
        episode_done = False
        step = 0
        
        while not episode_done and running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        episode_done = True
            
            if not running:
                break
            
            # Player zombie movement with acceleration
            keys = pygame.key.get_pressed()
            move_vec = np.array([0.0, 0.0])
            
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                move_vec[1] += 1
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:
                move_vec[1] -= 1
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                move_vec[0] -= 1
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                move_vec[0] += 1
            
            if np.linalg.norm(move_vec) > 0.01:
                move_vec = move_vec / np.linalg.norm(move_vec)
                move_angle = math.atan2(move_vec[1], move_vec[0])
                # Snap to nearest 45° direction
                new_dir = int(round((move_angle + math.pi) / (math.pi / 4))) % N_DIRECTIONS
                
                # Check if same direction
                if new_dir == player_move_dir:
                    player_accel_steps += 1
                else:
                    player_accel_steps = 0
                    player_move_dir = new_dir
                
                # Calculate speed with acceleration
                accel_progress = min(1.0, player_accel_steps / ACCEL_STEPS_TO_MAX)
                player_velocity = player_base_speed + (player_max_speed - player_base_speed) * accel_progress
                
                player_zombie_pos = player_zombie_pos + move_vec * player_velocity
            else:
                # Not moving - decelerate
                player_velocity = max(0, player_velocity - 0.05)
                player_accel_steps = 0
                player_move_dir = -1
            
            # Update player zombie position in the env so agent can see us
            player_zombie["pos"] = player_zombie_pos.copy()
            
            # Inject player zombie and herder into the zombie list so agent can see them
            # Remove any previous player zombie/herder entries first
            game_env.zombies = [z for z in game_env.zombies if z["id"] not in (-1, -2)]
            game_env.zombies.insert(0, player_zombie)
            
            # Update and inject herder
            herder_zombie["pos"] = herder_pos.copy()
            game_env.zombies.insert(0, herder_zombie)
            
            # Herder AI behavior
            agent_pos = game_env.agent["pos"]
            # Calculate center of all zombies (excluding player zombie and herder)
            ai_zombies = [z for z in game_env.zombies if z["id"] not in (-1, -2) and z["alive"]]
            if len(ai_zombies) > 0:
                zombie_center = np.mean([z["pos"] for z in ai_zombies], axis=0)
            else:
                zombie_center = agent_pos + np.array([5.0, 0.0])  # Default offset
            
            # Herder wants to be positioned so:
            # - Agent is to its right
            # - Zombie center is to the right of the agent (from herder's perspective)
            # This means herder should be on the opposite side of zombies from agent
            
            # Vector from zombie center to agent
            zc_to_agent = agent_pos - zombie_center
            if np.linalg.norm(zc_to_agent) > 0.1:
                zc_to_agent_norm = zc_to_agent / np.linalg.norm(zc_to_agent)
            else:
                zc_to_agent_norm = np.array([1.0, 0.0])
            
            # Target position: on the line extending from zombie_center through agent, past the agent
            target_dist_from_agent = 8.0  # How far past the agent to position
            target_pos = agent_pos + zc_to_agent_norm * target_dist_from_agent
            
            # Check if herder is in position (on the correct side)
            herder_to_agent = agent_pos - herder_pos
            herder_to_zombies = zombie_center - herder_pos
            
            # In position if: agent is closer than zombie center AND roughly aligned
            dist_to_agent = np.linalg.norm(herder_to_agent)
            dist_to_zombies = np.linalg.norm(herder_to_zombies)
            
            # Check alignment: dot product of (herder->agent) and (agent->zombies) should be positive
            agent_to_zombies = zombie_center - agent_pos
            if dist_to_agent > 0.1 and np.linalg.norm(agent_to_zombies) > 0.1:
                alignment = np.dot(herder_to_agent / dist_to_agent, 
                                   agent_to_zombies / np.linalg.norm(agent_to_zombies))
                herder_in_position = alignment > 0.7 and dist_to_agent < 15
            
            # Move herder
            if not herder_in_position:
                # Move toward target position (flanking position)
                move_dir = target_pos - herder_pos
                if np.linalg.norm(move_dir) > 0.1:
                    move_dir = move_dir / np.linalg.norm(move_dir)
                    herder_pos = herder_pos + move_dir * herder_speed
            else:
                # In position - now close in on the agent
                move_dir = agent_pos - herder_pos
                if np.linalg.norm(move_dir) > 0.1:
                    move_dir = move_dir / np.linalg.norm(move_dir)
                    herder_pos = herder_pos + move_dir * herder_speed * 0.8
            
            # Herder can also kill agent
            if np.linalg.norm(agent_pos - herder_pos) < ZOMBIE_ATTACK_RANGE and game_env.agent["alive"]:
                game_env.agent["health"] = 0
                game_env.agent["alive"] = False
                player_wins += 1
                print(f"Herder killed the agent! Score: You {player_wins} - {agent_wins} Agent")
            
            # Check if player zombie kills agent
            dist_to_agent = np.linalg.norm(game_env.agent["pos"] - player_zombie_pos)
            if dist_to_agent < ZOMBIE_ATTACK_RANGE and game_env.agent["alive"]:
                game_env.agent["health"] = 0
                game_env.agent["alive"] = False
                player_wins += 1
                print(f"You killed the agent! Score: You {player_wins} - {agent_wins} Agent")
            
            # AI agent takes action
            if game_env.agent["alive"]:
                # Get AI action
                ai_action, _ = ai_model.predict(vec_obs, deterministic=True)
                
                # Step the environment (this moves the agent and other zombies)
                obs, reward, terminated, truncated, info = game_env.step(int(ai_action[0]))
                
                # Update frame stack
                vec_obs = vec_env.env_method("step", int(ai_action[0]))[0]
                if isinstance(vec_obs, tuple):
                    vec_obs = vec_obs[0]
                # Actually we need to sync the vec_env observation properly
                # Simpler: just rebuild observation from game_env state
                single_obs = game_env._get_obs()
                # Stack it manually (repeat 8 times for frame stack)
                vec_obs = np.tile(single_obs, 8).reshape(1, -1)
                
                if terminated or truncated:
                    if not game_env.agent["alive"]:
                        # Agent died to AI zombies
                        player_wins += 1
                        print(f"Agent died! Score: You {player_wins} - {agent_wins} Agent")
                    episode_done = True
            
            # Check if agent survived long enough (timeout = win for agent)
            step += 1
            if step > 1000 and game_env.agent["alive"]:
                agent_wins += 1
                print(f"Agent survived! Score: You {player_wins} - {agent_wins} Agent")
                episode_done = True
            
            # Render
            renderer.draw(player_zombie_pos, game_env.agent["alive"], player_wins, agent_wins,
                         herder_pos=herder_pos, herder_in_position=herder_in_position)
            renderer.clock.tick(fps)
            
            # Small delay after kill for dramatic effect
            if not game_env.agent["alive"]:
                pygame.time.wait(1500)
                episode_done = True
    
    renderer.close()
    game_env.close()
    vec_env.close()
    
    print(f"\nFinal Score: You {player_wins} - {agent_wins} Agent")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Play as a zombie against the trained AI agent")
    parser.add_argument("--model", type=str, help="Path to AI model (auto-finds latest if not specified)")
    parser.add_argument("--fps", type=int, default=30, help="Game speed")
    
    args = parser.parse_args()
    
    main(model_path=args.model, fps=args.fps)
