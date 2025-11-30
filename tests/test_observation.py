"""Tests for agent observation computation (20-dim egocentric spec)."""
import numpy as np
import math
import pytest

from worms_3d_gym.envs.worms_3d_env import (
    compute_agent_observation, 
    line_hits_obstacle, 
    ray_cast,
    OBS_DIM, MAX_HEALTH, MAX_COOLDOWN, MAX_SPEED_FORWARD, N_RAYS, RAY_MAX_RANGE
)

ARENA_SIZE = 30
OBSTACLE = [[10, 10, 20, 20]]  # List of obstacles (each obstacle is [x_min, y_min, x_max, y_max])


class TestLineHitsObstacle:
    """Tests for line_hits_obstacle function."""
    
    def test_line_through_obstacle(self):
        """Line passing through obstacle should return True."""
        assert line_hits_obstacle(0, 15, 30, 15, OBSTACLE) is True
    
    def test_line_around_obstacle(self):
        """Line not touching obstacle should return False."""
        assert line_hits_obstacle(0, 0, 8, 8, OBSTACLE) is False
    
    def test_line_ending_in_obstacle(self):
        """Line ending inside obstacle should return True."""
        assert line_hits_obstacle(0, 15, 15, 15, OBSTACLE) is True
    
    def test_line_starting_in_obstacle(self):
        """Line starting inside obstacle should return True."""
        assert line_hits_obstacle(15, 15, 30, 30, OBSTACLE) is True
    
    def test_diagonal_miss(self):
        """Diagonal line missing obstacle should return False."""
        assert line_hits_obstacle(0, 0, 5, 25, OBSTACLE) is False


class TestRayCast:
    """Tests for ray_cast function."""
    
    def test_ray_hits_wall(self):
        """Ray should hit arena boundary."""
        # From near edge, facing right (+x)
        dist = ray_cast(25, 5, 0, RAY_MAX_RANGE, ARENA_SIZE, OBSTACLE)
        assert dist < RAY_MAX_RANGE  # Should hit wall before max range
    
    def test_ray_hits_obstacle(self):
        """Ray should hit obstacle."""
        # From left side, facing right toward obstacle at x=12
        dist = ray_cast(5, 15, 0, RAY_MAX_RANGE, ARENA_SIZE, OBSTACLE)
        assert dist < 8  # Should hit obstacle at x=12
    
    def test_ray_max_range(self):
        """Ray should return max_range if no hit."""
        # Short max range that won't hit anything
        dist = ray_cast(2, 2, 0, 2.0, ARENA_SIZE, OBSTACLE)
        assert dist == 2.0


class TestComputeAgentObservation:
    """Tests for 20-dim compute_agent_observation function."""
    
    @pytest.fixture
    def agent(self):
        return {
            "pos": np.array([3.0, 3.0]),
            "health": MAX_HEALTH,
            "angle": 0.0,
            "alive": True,
            "velocity": np.array([0.0, 0.0]),
            "cooldown": 0
        }
    
    @pytest.fixture
    def enemy(self):
        return {
            "pos": np.array([12.0, 3.0]),
            "health": 75.0,
            "angle": math.pi,
            "alive": True
        }
    
    def test_observation_shape(self, agent, enemy):
        """Observation should have 26 elements."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs.shape == (OBS_DIM,)
        assert obs.shape == (26,)
        assert obs.dtype == np.float32
    
    # =========================================================================
    # 3.1 Self state tests (indices 0-1)
    # =========================================================================
    def test_health_normalized(self, agent, enemy):
        """Test health normalization at index 0."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[0] == pytest.approx(1.0)  # Full health
        
        agent["health"] = 50.0
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[0] == pytest.approx(0.5)
    
    def test_cooldown_normalized(self, agent, enemy):
        """Test cooldown normalization at index 1."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[1] == pytest.approx(0.0)  # Ready to fire
        
        agent["cooldown"] = MAX_COOLDOWN
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[1] == pytest.approx(1.0)
    
    # =========================================================================
    # 3.2 Enemy info tests (indices 2-5)
    # =========================================================================
    def test_enemy_direction_facing(self, agent, enemy):
        """Test cos/sin delta when facing enemy."""
        # Agent at (3,3) facing right (angle=0), enemy at (12,3) - directly ahead
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[2] == pytest.approx(1.0)  # cos_delta_enemy = 1 (directly ahead)
        assert obs[3] == pytest.approx(0.0)  # sin_delta_enemy = 0
    
    def test_enemy_direction_behind(self, agent, enemy):
        """Test cos/sin delta when enemy is behind."""
        agent["angle"] = math.pi  # Facing left, enemy is to the right
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[2] == pytest.approx(-1.0)  # cos_delta_enemy = -1 (behind)
        assert obs[3] == pytest.approx(0.0, abs=0.01)  # sin_delta_enemy ≈ 0
    
    def test_enemy_direction_left(self, agent, enemy):
        """Test cos/sin delta when enemy is to the left."""
        agent["angle"] = -math.pi / 2  # Facing down
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[2] == pytest.approx(0.0, abs=0.01)  # cos ≈ 0
        assert obs[3] == pytest.approx(1.0)  # sin = 1 (enemy to left)
    
    def test_enemy_distance_normalized(self, agent, enemy):
        """Test distance normalization at index 4."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        max_dist = ARENA_SIZE * math.sqrt(2)
        expected_dist = 9.0 / max_dist  # Distance from (3,3) to (12,3) = 9
        assert obs[4] == pytest.approx(expected_dist, rel=0.01)
    
    def test_has_los_clear(self, agent, enemy):
        """Test line of sight when clear at index 5."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[5] == 1.0  # Clear LOS
    
    def test_has_los_blocked(self, agent, enemy):
        """Test line of sight when blocked by obstacle."""
        # Obstacle is at [10, 10, 20, 20], so line through y=15 should be blocked
        agent["pos"] = np.array([5.0, 15.0])
        enemy["pos"] = np.array([25.0, 15.0])
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[5] == 0.0  # Blocked by obstacle
    
    def test_enemy_dead_defaults(self, agent, enemy):
        """Test enemy info defaults when enemy is dead."""
        enemy["alive"] = False
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[2] == 0.0  # cos_delta_enemy
        assert obs[3] == 0.0  # sin_delta_enemy
        assert obs[4] == 1.0  # dist_enemy_norm
        assert obs[5] == 0.0  # has_los
    
    # =========================================================================
    # 3.3 Ray sensors tests (indices 6-13)
    # =========================================================================
    def test_ray_sensors_count(self, agent, enemy):
        """Test that we have 8 ray sensors."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        ray_values = obs[6:14]
        assert len(ray_values) == N_RAYS
        assert len(ray_values) == 8
    
    def test_ray_sensors_normalized(self, agent, enemy):
        """Test that ray values are in [0, 1]."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        for i in range(N_RAYS):
            assert 0.0 <= obs[6 + i] <= 1.0
    
    def test_ray_sensors_detect_wall(self, agent, enemy):
        """Test rays detect arena boundary."""
        # Agent near left wall
        agent["pos"] = np.array([1.0, 7.5])
        agent["angle"] = math.pi  # Facing left
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        # Forward ray should detect wall very close
        assert obs[6 + N_RAYS // 2] < 0.2  # Middle ray (forward) should be short
    
    # =========================================================================
    # 3.4 Step feedback tests (indices 14-15)
    # =========================================================================
    def test_step_feedback_defaults(self, agent, enemy):
        """Test step feedback defaults to 0."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE)
        assert obs[14] == 0.0  # was_hit_last_step
        assert obs[15] == 0.0  # hit_enemy_last_step
    
    def test_step_feedback_was_hit(self, agent, enemy):
        """Test was_hit_last_step flag."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE, 
                                        was_hit_last_step=1.0)
        assert obs[14] == 1.0
        assert obs[15] == 0.0
    
    def test_step_feedback_hit_enemy(self, agent, enemy):
        """Test hit_enemy_last_step flag."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE,
                                        hit_enemy_last_step=1.0)
        assert obs[14] == 0.0
        assert obs[15] == 1.0
    
    def test_step_feedback_both(self, agent, enemy):
        """Test both feedback flags set."""
        obs = compute_agent_observation(agent, enemy, OBSTACLE, ARENA_SIZE,
                                        was_hit_last_step=1.0, hit_enemy_last_step=1.0)
        assert obs[14] == 1.0
        assert obs[15] == 1.0


class TestShooting:
    """Tests for shooting mechanics."""
    
    def test_direct_hit_same_y(self):
        """Shot should hit when perfectly aligned on same Y."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        # Place agents on same Y, facing each other
        env.game_agents[0]["pos"] = np.array([5.0, 15.0])
        env.game_agents[0]["angle"] = 0.0  # Facing right
        env.game_agents[1]["pos"] = np.array([25.0, 15.0])
        env.game_agents[1]["health"] = 100.0
        
        # Agent 0 shoots
        hit = env._shoot_direction(env.game_agents[0], math.cos(0), math.sin(0))
        
        assert hit is True
        assert env.game_agents[1]["health"] == 75.0  # 100 - 25 damage
    
    def test_direct_hit_small_y_offset(self):
        """Shot should hit with small Y offset (within hit radius 1.5)."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        # Place agents with 1.0 Y offset (within 1.5 hit radius)
        env.game_agents[0]["pos"] = np.array([5.0, 15.0])
        env.game_agents[0]["angle"] = 0.0  # Facing right
        env.game_agents[1]["pos"] = np.array([25.0, 16.0])  # 1 unit Y offset
        env.game_agents[1]["health"] = 100.0
        
        hit = env._shoot_direction(env.game_agents[0], math.cos(0), math.sin(0))
        
        assert hit is True
        assert env.game_agents[1]["health"] == 75.0
    
    def test_miss_large_y_offset(self):
        """Shot should miss with large Y offset (outside hit radius 1.5)."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        # Place agents with 2.0 Y offset (outside 1.5 hit radius)
        env.game_agents[0]["pos"] = np.array([5.0, 15.0])
        env.game_agents[0]["angle"] = 0.0  # Facing right
        env.game_agents[1]["pos"] = np.array([25.0, 17.0])  # 2 units Y offset
        env.game_agents[1]["health"] = 100.0
        
        hit = env._shoot_direction(env.game_agents[0], math.cos(0), math.sin(0))
        
        assert hit is False
        assert env.game_agents[1]["health"] == 100.0  # No damage
    
    def test_aimed_shot_hits(self):
        """Shot aimed directly at enemy should hit regardless of distance."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        # Place agents with Y offset, but aim directly at enemy
        env.game_agents[0]["pos"] = np.array([5.0, 15.0])
        env.game_agents[1]["pos"] = np.array([25.0, 20.0])  # 5 units Y offset
        env.game_agents[1]["health"] = 100.0
        
        # Calculate angle to enemy
        dx = 25.0 - 5.0
        dy = 20.0 - 15.0
        angle_to_enemy = math.atan2(dy, dx)
        env.game_agents[0]["angle"] = angle_to_enemy
        
        hit = env._shoot_direction(env.game_agents[0], math.cos(angle_to_enemy), math.sin(angle_to_enemy))
        
        assert hit is True
        assert env.game_agents[1]["health"] == 75.0
    
    def test_shot_behind_misses(self):
        """Shot in opposite direction should miss."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        env.game_agents[0]["pos"] = np.array([5.0, 15.0])
        env.game_agents[0]["angle"] = math.pi  # Facing left (away from enemy)
        env.game_agents[1]["pos"] = np.array([25.0, 15.0])
        env.game_agents[1]["health"] = 100.0
        
        hit = env._shoot_direction(env.game_agents[0], math.cos(math.pi), math.sin(math.pi))
        
        assert hit is False
        assert env.game_agents[1]["health"] == 100.0
    
    def test_opposite_corners_aimed_hit(self):
        """Shot from one corner aimed at enemy in opposite corner should hit."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        # Agent 0 in bottom-left corner, Agent 1 in top-right corner
        env.game_agents[0]["pos"] = np.array([2.0, 2.0])
        env.game_agents[1]["pos"] = np.array([28.0, 28.0])
        env.game_agents[1]["health"] = 100.0
        
        # Calculate angle to enemy (diagonal)
        dx = 28.0 - 2.0
        dy = 28.0 - 2.0
        angle_to_enemy = math.atan2(dy, dx)  # Should be π/4 (45°)
        env.game_agents[0]["angle"] = angle_to_enemy
        
        hit = env._shoot_direction(env.game_agents[0], math.cos(angle_to_enemy), math.sin(angle_to_enemy))
        
        assert hit is True
        assert env.game_agents[1]["health"] == 75.0
    
    def test_opposite_corners_horizontal_miss(self):
        """Horizontal shot from corner should miss enemy in opposite corner."""
        from worms_3d_gym.envs.worms_pettingzoo import Worms3DPettingZooEnv
        
        env = Worms3DPettingZooEnv()
        env.reset()
        
        # Agent 0 in bottom-left, Agent 1 in top-right
        env.game_agents[0]["pos"] = np.array([2.0, 2.0])
        env.game_agents[0]["angle"] = 0.0  # Facing right (horizontal)
        env.game_agents[1]["pos"] = np.array([28.0, 28.0])
        env.game_agents[1]["health"] = 100.0
        
        # Horizontal shot - enemy is 26 units up, way outside 1.5 hit radius
        hit = env._shoot_direction(env.game_agents[0], math.cos(0), math.sin(0))
        
        assert hit is False
        assert env.game_agents[1]["health"] == 100.0
