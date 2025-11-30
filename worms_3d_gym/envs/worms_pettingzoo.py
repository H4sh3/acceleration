"""PettingZoo ParallelEnv wrapper for Worms3D.

Exposes the environment as a multi-agent parallel environment where each agent
has its own observation space, action space, and reward.
"""
import functools
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
import math

from .worms_3d_env import (
    Worms3DEnv, OBS_DIM, compute_agent_observation
)


class Worms3DPettingZooEnv(ParallelEnv):
    """PettingZoo parallel environment wrapper for Worms3D.
    
    Each agent gets:
    - Its own 28-dim egocentric observation
    - Discrete(8) action space
    - Individual reward signal
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "name": "worms3d_v0"}
    
    def __init__(self, render_mode=None, curriculum_phase=0):
        super().__init__()
        
        # Create the underlying environment
        self._env = Worms3DEnv(render_mode=render_mode, curriculum_phase=curriculum_phase)
        self._curriculum_phase = curriculum_phase
        
        # Agent IDs
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()
        
        # Per-agent spaces (28-dim observation, 8 discrete actions)
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self._action_space = spaces.Discrete(self._env.N_ACTIONS)
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_space
    
    def reset(self, seed=None, options=None):
        """Reset and return per-agent observations."""
        self._env.reset(seed=seed, options=options)
        self.agents = self.possible_agents.copy()
        
        # Track previous distances for distance shaping
        self._prev_distances = [None, None]
        
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions):
        """Step with per-agent actions, return per-agent results.
        
        Args:
            actions: Dict mapping agent_id -> action (int)
        
        Returns:
            observations: Dict mapping agent_id -> obs array
            rewards: Dict mapping agent_id -> float reward
            terminations: Dict mapping agent_id -> bool
            truncations: Dict mapping agent_id -> bool
            infos: Dict mapping agent_id -> info dict
        """
        # Convert dict actions to array for underlying env
        action_array = np.array([
            actions.get("agent_0", 0),
            actions.get("agent_1", 0)
        ])
        
        # Step underlying environment (we need to bypass the reward summing)
        # Replicate the step logic but keep rewards separate
        rewards_array = self._step_with_separate_rewards(action_array)
        
        # Check termination/truncation
        alive_teams = set(a["team"] for a in self._env.agents if a["alive"])
        terminated = len(alive_teams) <= 1
        truncated = self._env.current_step >= 200
        
        # Penalty if episode ends and enemy survives (encourages finishing the fight)
        if truncated and not terminated:
            for i, agent in enumerate(self._env.agents):
                enemy = self._env.agents[1 - i]
                if agent["alive"] and enemy["alive"]:
                    rewards_array[i] -= 500.0  # Penalty for not killing enemy
        
        # Build per-agent outputs
        observations = self._get_observations()
        rewards = {
            "agent_0": float(rewards_array[0]),
            "agent_1": float(rewards_array[1])
        }
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {"alive_teams": list(alive_teams)} for agent in self.agents}
        
        # Remove dead agents from active list
        if terminated or truncated:
            self.agents = []
        
        return observations, rewards, terminations, truncations, infos
    
    def _step_with_separate_rewards(self, actions):
        """Execute step logic and return separate rewards per agent.
        
        This replicates the core step logic from Worms3DEnv but keeps rewards separate.
        Uses dense reward shaping to guide learning:
        - Continuous aiming reward
        - Distance shaping
        - Shoot accuracy bonus/penalty
        """
        
        env = self._env
        rewards = np.zeros(env.n_agents)
        env.last_shots = []
        
        # Store previous health
        prev_health = [a["health"] for a in env.agents]
        env.was_hit = [0.0, 0.0]
        env.hit_enemy = [0.0, 0.0]
        
        MOVE_STEP = 1
        ROTATE_STEP = 0.15  # Faster rotation (~8.6 degrees per step)
        
        for i, agent in enumerate(env.agents):
            if not agent["alive"]:
                continue
            
            act = actions[i]
            enemy = env.agents[1 - i]
            
            # Movement actions
            if act == 1:  # Up
                agent["pos"][1] += MOVE_STEP
            elif act == 2:  # Down
                agent["pos"][1] -= MOVE_STEP
            elif act == 3:  # Left
                agent["pos"][0] -= MOVE_STEP
            elif act == 4:  # Right
                agent["pos"][0] += MOVE_STEP
            elif act == 5:  # Rotate left
                agent["angle"] += ROTATE_STEP
            elif act == 6:  # Rotate right
                agent["angle"] -= ROTATE_STEP
            elif act == 7:  # Shoot
                dx = math.cos(agent["angle"])
                dy = math.sin(agent["angle"])
                hit = env._shoot_direction(agent, dx, dy)
                
                # Reward/penalty based on whether shot was well-aimed
                if not hit:
                    rewards[i] -= 2.0  # Miss penalty (hit reward comes from damage)
            
            env._handle_collision(agent)
        
        # Post-action rewards: aiming, distance shaping, damage
        for i, agent in enumerate(env.agents):
            enemy = env.agents[1 - i]
            
            if agent["alive"] and enemy["alive"]:
                obs = compute_agent_observation(agent, enemy, env.OBSTACLES, env.SIZE)
                cos_delta_enemy = obs[0]  # How well aimed at enemy (-1 to 1)
                dist_enemy_norm = obs[2]  # Normalized distance (0 to 1)
                has_los = obs[3]  # Line of sight (0 or 1)
                
                # =============================================================
                # CONTINUOUS AIMING REWARD (dense signal)
                # =============================================================
                # Reward for facing enemy: scales from 0 (perpendicular) to 1.0 (perfect aim)
                # Only when we have line of sight
                if has_los > 0.5:
                    # cos_delta ranges from -1 (facing away) to 1 (facing toward)
                    # Transform to 0-1 range and apply threshold
                    aim_quality = max(0, cos_delta_enemy)  # 0 when perpendicular/away, 1 when aimed
                    rewards[i] += aim_quality * 0.5  # Up to +0.5 per step for good aim
                    
                    # Extra bonus for very good aim (within ~25 degrees)
                    if cos_delta_enemy > 0.9:
                        rewards[i] += 1.0  # Strong signal: you're aimed, now shoot!
                
                # =============================================================
                # DISTANCE SHAPING (encourage approaching)
                # =============================================================
                # Reward for getting closer to enemy
                if self._prev_distances[i] is not None:
                    dist_delta = self._prev_distances[i] - dist_enemy_norm
                    rewards[i] += dist_delta * 5.0  # Reward for closing distance
                
                self._prev_distances[i] = dist_enemy_norm
            
            # Time penalty - encourage faster kills
            rewards[i] -= 0.1
            
            # Damage rewards: +2.0 per damage point (50 per hit)
            enemy_damage = prev_health[1 - i] - enemy["health"]
            if enemy_damage > 0 and env.hit_enemy[i] > 0:
                rewards[i] += enemy_damage * 2  # +50 per hit (25 dmg * 2)
            
            # Kill/death rewards
            if prev_health[1 - i] > 0 and not enemy["alive"]:
                rewards[i] += 10000  # Kill bonus
            
            if prev_health[i] > 0 and not agent["alive"]:
                rewards[i] -= 400  # Death penalty
        
        env.current_step = getattr(env, 'current_step', 0) + 1
        
        return rewards
    
    def _get_observations(self):
        """Get per-agent observations."""
        env = self._env
        a0, a1 = env.agents[0], env.agents[1]
        
        obs0 = compute_agent_observation(
            a0, a1, env.OBSTACLES, env.SIZE,
            was_hit_last_step=env.was_hit[0],
            hit_enemy_last_step=env.hit_enemy[0]
        )
        obs1 = compute_agent_observation(
            a1, a0, env.OBSTACLES, env.SIZE,
            was_hit_last_step=env.was_hit[1],
            hit_enemy_last_step=env.hit_enemy[1]
        )
        
        return {
            "agent_0": obs0,
            "agent_1": obs1
        }
    
    def render(self):
        return self._env.render()
    
    def close(self):
        self._env.close()
    
    @property
    def unwrapped(self):
        return self._env
    
    @property
    def curriculum_phase(self):
        return self._curriculum_phase
    
    @curriculum_phase.setter
    def curriculum_phase(self, value):
        self._curriculum_phase = value
        self._env.curriculum_phase = value
