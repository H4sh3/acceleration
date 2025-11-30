"""Minimal Aim Training Environment.

Single agent learns to rotate and shoot at a randomly placed stationary target.
Focused on aiming mechanics only - no movement, no enemy AI.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# =============================================================================
# Constants
# =============================================================================
MAX_HEALTH = 100.0
MAX_COOLDOWN = 5
OBS_DIM = 6  # Minimal observation for aiming


class AimTrainerEnv(gym.Env):
    """Minimal aim training environment.
    
    Agent is fixed at center, target spawns randomly around it.
    Agent can only rotate and shoot.
    
    Observation (6 dims):
        0: cos_delta - cosine of angle to target (1.0 = facing target)
        1: sin_delta - sine of angle to target (sign indicates turn direction)
        2: dist_norm - normalized distance to target
        3: cooldown_norm - shot cooldown (0 = can shoot)
        4: would_hit - 1.0 if shooting now would hit
        5: hit_last_step - 1.0 if hit target last step
    
    Actions:
        0: nothing
        1: rotate left (CCW)
        2: rotate right (CW)
        3: shoot
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    ARENA_SIZE = 30
    N_ACTIONS = 4
    
    # Target spawn config
    MIN_DIST = 5.0
    MAX_DIST = 12.0
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(self.N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        
        self.agent_pos = np.array([self.ARENA_SIZE / 2, self.ARENA_SIZE / 2])
        self.agent_angle = 0.0
        self.cooldown = 0
        
        self.target_pos = None
        self.target_alive = True
        
        self.hit_last_step = 0.0
        self.current_step = 0
        self.targets_hit = 0
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Agent at center, random facing
        self.agent_pos = np.array([self.ARENA_SIZE / 2, self.ARENA_SIZE / 2])
        self.agent_angle = self.np_random.uniform(-math.pi, math.pi)
        self.cooldown = 0
        
        # Spawn target
        self._spawn_target()
        
        self.hit_last_step = 0.0
        self.current_step = 0
        self.targets_hit = 0
        
        return self._get_obs(), {}
    
    def _spawn_target(self):
        """Spawn target at random position around agent."""
        angle = self.np_random.uniform(-math.pi, math.pi)
        dist = self.np_random.uniform(self.MIN_DIST, self.MAX_DIST)
        
        self.target_pos = self.agent_pos + np.array([
            math.cos(angle) * dist,
            math.sin(angle) * dist
        ])
        self.target_alive = True
    
    def step(self, action):
        self.current_step += 1
        reward = 0.0
        self.hit_last_step = 0.0
        
        # Decrease cooldown
        if self.cooldown > 0:
            self.cooldown -= 1
        
        ROTATE_STEP = 0.25  # ~14 degrees
        
        # Process action
        if action == 1:  # Rotate left (CCW)
            self.agent_angle += ROTATE_STEP
        elif action == 2:  # Rotate right (CW)
            self.agent_angle -= ROTATE_STEP
        elif action == 3:  # Shoot
            if self.cooldown == 0:
                hit = self._shoot()
                self.cooldown = MAX_COOLDOWN
                
                if hit:
                    reward += 10.0
                    self.hit_last_step = 1.0
                    self.targets_hit += 1
                    # Respawn target
                    self._spawn_target()
                else:
                    reward -= 0.5  # Miss penalty
        
        # Small shaping reward for aiming well
        cos_delta, _ = self._get_angle_to_target()
        if cos_delta > 0.95:
            reward += 0.1  # Bonus for good aim
        
        # Episode ends after max steps or enough targets hit
        terminated = self.targets_hit >= 10
        truncated = self.current_step >= 200
        
        return self._get_obs(), reward, terminated, truncated, {
            "targets_hit": self.targets_hit
        }
    
    def _get_angle_to_target(self):
        """Get cos/sin of angle from facing direction to target."""
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        
        angle_to_target = math.atan2(dy, dx)
        delta = angle_to_target - self.agent_angle
        
        return math.cos(delta), math.sin(delta)
    
    def _get_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        
        # Angle to target
        cos_delta, sin_delta = self._get_angle_to_target()
        obs[0] = cos_delta
        obs[1] = sin_delta
        
        # Distance
        dx = self.target_pos[0] - self.agent_pos[0]
        dy = self.target_pos[1] - self.agent_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        obs[2] = np.clip(dist / self.MAX_DIST, 0.0, 1.0)
        
        # Cooldown
        obs[3] = self.cooldown / MAX_COOLDOWN
        
        # Would hit
        obs[4] = 1.0 if self._would_hit() else 0.0
        
        # Hit feedback
        obs[5] = self.hit_last_step
        
        return obs
    
    def _would_hit(self):
        """Check if shooting now would hit the target."""
        dx = math.cos(self.agent_angle)
        dy = math.sin(self.agent_angle)
        
        to_target = self.target_pos - self.agent_pos
        proj = to_target[0] * dx + to_target[1] * dy
        
        if proj <= 0:
            return False
        
        closest = np.array([dx * proj, dy * proj])
        perp_dist = np.linalg.norm(to_target - closest)
        
        return perp_dist < 1.5  # Hit radius
    
    def _shoot(self):
        """Execute shot, return True if hit."""
        return self._would_hit()
    
    def render(self):
        if self.render_mode != "human" or not PYGAME_AVAILABLE:
            return
        
        # Initialize pygame on first render
        if not hasattr(self, '_screen'):
            pygame.init()
            self._scale = 15
            size = int(self.ARENA_SIZE * self._scale)
            self._screen = pygame.display.set_mode((size, size))
            pygame.display.set_caption("Aim Trainer")
            self._clock = pygame.time.Clock()
            self._font = pygame.font.Font(None, 24)
        
        # Colors
        BG = (210, 180, 140)
        BLACK = (0, 0, 0)
        GREEN = (80, 255, 80)
        RED = (255, 80, 80)
        YELLOW = (255, 255, 0)
        
        self._screen.fill(BG)
        
        def to_screen(x, y):
            return int(x * self._scale), int((self.ARENA_SIZE - y) * self._scale)
        
        # Draw target
        tx, ty = to_screen(self.target_pos[0], self.target_pos[1])
        pygame.draw.circle(self._screen, RED, (tx, ty), 12)
        pygame.draw.circle(self._screen, BLACK, (tx, ty), 12, 2)
        
        # Draw agent
        ax, ay = to_screen(self.agent_pos[0], self.agent_pos[1])
        pygame.draw.circle(self._screen, GREEN, (ax, ay), 10)
        pygame.draw.circle(self._screen, BLACK, (ax, ay), 10, 2)
        
        # Draw facing direction
        dir_len = 25
        end_x = ax + math.cos(self.agent_angle) * dir_len
        end_y = ay - math.sin(self.agent_angle) * dir_len  # Y inverted
        pygame.draw.line(self._screen, BLACK, (ax, ay), (int(end_x), int(end_y)), 3)
        
        # Draw would-hit indicator
        if self._would_hit():
            # Draw shot line to target
            pygame.draw.line(self._screen, YELLOW, (ax, ay), (tx, ty), 2)
        
        # Draw info
        cos_delta, sin_delta = self._get_angle_to_target()
        info_text = f"Targets: {self.targets_hit}  cos:{cos_delta:.2f}  sin:{sin_delta:.2f}"
        text = self._font.render(info_text, True, BLACK)
        self._screen.blit(text, (10, 10))
        
        hit_text = "AIM: HIT!" if self._would_hit() else "AIM: ---"
        color = YELLOW if self._would_hit() else (100, 100, 100)
        text2 = self._font.render(hit_text, True, color)
        self._screen.blit(text2, (10, 35))
        
        pygame.display.flip()
        self._clock.tick(30)
        
        # Handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
    
    def close(self):
        if hasattr(self, '_screen'):
            pygame.quit()
            del self._screen


# Register the environment
if __name__ == "__main__":
    # Quick test
    env = AimTrainerEnv()
    obs, _ = env.reset()
    print(f"Initial obs: {obs}")
    
    for _ in range(20):
        # Simple policy: rotate toward target based on sin_delta
        if obs[4] > 0.5 and obs[3] == 0:  # Would hit and can shoot
            action = 3
        elif obs[1] > 0:  # sin_delta > 0 -> rotate left
            action = 1
        else:
            action = 2
        
        obs, reward, term, trunc, info = env.step(action)
        print(f"Action {action}: reward={reward:.1f} cos={obs[0]:.2f} sin={obs[1]:.2f} "
              f"would_hit={obs[4]:.0f} targets={info['targets_hit']}")
        
        if term or trunc:
            break
