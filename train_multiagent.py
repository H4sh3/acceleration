"""Multi-agent training with independent PPO networks per agent.

Uses PettingZoo interface where each agent has its own neural network.
"""
import sys
import os
from datetime import datetime
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from worms_3d_gym.envs import Worms3DPettingZooEnv


class MultiAgentLogger(BaseCallback):
    """Callback to log multi-agent training progress to TensorBoard."""
    
    def __init__(self, agent_name, verbose=0):
        super().__init__(verbose)
        self.agent_name = agent_name
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self):
        # Track rewards and length
        if len(self.locals.get("rewards", [])) > 0:
            self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1
        
        # Check for episode end
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Track win/loss from info
            info = infos[0] if infos else {}
            alive_teams = info.get("alive_teams", [])
            agent_team = 0 if self.agent_name == "agent_0" else 1
            
            if len(alive_teams) == 1:
                if agent_team in alive_teams:
                    self.wins += 1
                else:
                    self.losses += 1
            else:
                self.draws += 1
            
            # Log to TensorBoard
            n_episodes = len(self.episode_rewards)
            self.logger.record(f"{self.agent_name}/episode_reward", self.current_episode_reward)
            self.logger.record(f"{self.agent_name}/episode_length", self.current_episode_length)
            self.logger.record(f"{self.agent_name}/total_episodes", n_episodes)
            self.logger.record(f"{self.agent_name}/wins", self.wins)
            self.logger.record(f"{self.agent_name}/losses", self.losses)
            self.logger.record(f"{self.agent_name}/draws", self.draws)
            
            if n_episodes >= 10:
                self.logger.record(f"{self.agent_name}/mean_reward_10ep", np.mean(self.episode_rewards[-10:]))
                self.logger.record(f"{self.agent_name}/win_rate", self.wins / n_episodes)
            
            if n_episodes % 50 == 0:
                mean_rew = np.mean(self.episode_rewards[-50:])
                win_rate = self.wins / n_episodes if n_episodes > 0 else 0
                print(f"[{self.agent_name}] Ep {n_episodes}: mean_rew={mean_rew:.2f}, "
                      f"wins={self.wins}, losses={self.losses}, win_rate={win_rate:.2%}")
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


class SingleAgentEnvWrapper(gym.Env):
    """Wraps PettingZoo env to look like a single-agent Gym env for one agent.
    
    This allows using standard SB3 algorithms with PettingZoo.
    The other agent's actions come from its own policy.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, pz_env, agent_id, other_policy=None):
        """
        Args:
            pz_env: PettingZoo parallel environment
            agent_id: Which agent this wrapper controls ("agent_0" or "agent_1")
            other_policy: Policy for the other agent (None = random)
        """
        super().__init__()
        self.env = pz_env
        self.agent_id = agent_id
        self.other_id = "agent_1" if agent_id == "agent_0" else "agent_0"
        self.other_policy = other_policy
        
        # Expose spaces for this agent
        self.observation_space = pz_env.observation_space(agent_id)
        self.action_space = pz_env.action_space(agent_id)
        
        self._last_obs = None
    
    def reset(self, seed=None, options=None):
        observations, infos = self.env.reset(seed=seed, options=options)
        self._last_obs = observations
        return observations[self.agent_id], infos.get(self.agent_id, {})
    
    def step(self, action):
        # Get other agent's action
        if self.other_policy is not None:
            other_obs = self._last_obs[self.other_id]
            other_action, _ = self.other_policy.predict(other_obs, deterministic=False)
        else:
            other_action = self.env.action_space(self.other_id).sample()
        
        # Build action dict
        actions = {
            self.agent_id: action,
            self.other_id: int(other_action)
        }
        
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self._last_obs = observations
        
        obs = observations[self.agent_id]
        reward = rewards[self.agent_id]
        terminated = terminations[self.agent_id]
        truncated = truncations[self.agent_id]
        info = infos.get(self.agent_id, {})
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def get_run_dir(prefix="multiagent"):
    """Create a unique run directory with timestamp."""
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def create_ppo_agent(env, run_dir, agent_name):
    """Create a PPO agent with tuned hyperparameters for combat learning."""
    return PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, f"logs_{agent_name}"),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.05,  # Reduced from 0.2 - less random exploration, more exploitation
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])  # Larger network
        ),
    )


def train_population(n_agents=5, steps_per_match=5000, n_rounds=100, n_envs=4, 
                     phase1_rounds=50, phase2_rounds=50):
    """Train a population of agents via round-robin tournament with curriculum.
    
    Curriculum:
    - Phase 1 (rounds 1 to phase1_rounds): No obstacles, learn basic aiming
    - Phase 2 (remaining rounds): Add center obstacle
    
    Each round, every agent fights every other agent once.
    For 3 agents: A-B, A-C, B-C (3 matches per round).
    
    Args:
        n_agents: Number of agents in the population
        steps_per_match: Training steps per match
        n_rounds: Number of tournament rounds (phase1_rounds + phase2_rounds)
        n_envs: Number of parallel environments per match (speedup)
        phase1_rounds: Rounds in phase 1 (no obstacles)
        phase2_rounds: Rounds in phase 2 (with obstacles)
    """
    n_rounds = phase1_rounds + phase2_rounds
    from itertools import combinations
    
    print(f"Setting up population training with {n_agents} agents...")
    
    run_dir = get_run_dir(prefix=f"population_{n_agents}agents")
    print(f"Run directory: {run_dir}")
    
    # Generate all matchups
    agent_names = [f"agent_{i}" for i in range(n_agents)]
    matchups = list(combinations(range(n_agents), 2))
    print(f"Matchups per round: {len(matchups)} -> {matchups}")
    
    # Create one PettingZoo env per matchup (reused across rounds)
    # Start with phase 0 (no obstacles)
    envs = {}
    for i, j in matchups:
        envs[(i, j)] = Worms3DPettingZooEnv(render_mode=None, curriculum_phase=0)
    
    print(f"Observation Space: {envs[matchups[0]].observation_space('agent_0')}")
    print(f"Action Space: {envs[matchups[0]].action_space('agent_0')}")
    
    # Initialize all PPO models (initially with random opponents)
    models = {}
    loggers = {}
    
    # Create a dummy env for initialization
    dummy_env = Worms3DPettingZooEnv(render_mode=None)
    
    for i in range(n_agents):
        name = agent_names[i]
        # Each model needs its own wrapped env for initialization
        init_env = SingleAgentEnvWrapper(dummy_env, "agent_0", other_policy=None)
        models[i] = create_ppo_agent(init_env, run_dir, name)
        loggers[i] = MultiAgentLogger(name, verbose=1)
        print(f"Created {name}")
    
    dummy_env.close()
    
    print(f"\nStarting round-robin tournament training with curriculum...")
    print(f"Agents: {n_agents}")
    print(f"Matches per round: {len(matchups)}")
    print(f"Steps per match: {steps_per_match}")
    print(f"Phase 1 (no obstacles): {phase1_rounds} rounds")
    print(f"Phase 2 (with obstacles): {phase2_rounds} rounds")
    print(f"Total rounds: {n_rounds}")
    
    training_start_time = time.time()
    
    current_phase = 0
    
    for round_idx in range(n_rounds):
        round_start_time = time.time()
        
        # Check for phase transition
        new_phase = 0 if round_idx < phase1_rounds else 1
        if new_phase != current_phase:
            current_phase = new_phase
            print(f"\n{'#'*50}")
            print(f"### CURRICULUM PHASE {current_phase + 1}: {'No obstacles' if current_phase == 0 else 'With obstacles'} ###")
            print(f"{'#'*50}")
            # Update all environments to new phase
            for env in envs.values():
                env.curriculum_phase = current_phase
        
        print(f"\n{'='*50}")
        print(f"=== Round {round_idx + 1}/{n_rounds} (Phase {current_phase + 1}) ===")
        print(f"{'='*50}")
        
        # Each matchup in the round
        for match_idx, (i, j) in enumerate(matchups):
            print(f"\n--- Match {match_idx + 1}/{len(matchups)}: {agent_names[i]} vs {agent_names[j]} ---")
            
            pz_env = envs[(i, j)]
            
            # Randomly assign which network plays as agent_0 vs agent_1
            if np.random.random() < 0.5:
                slot_i, slot_j = "agent_0", "agent_1"
            else:
                slot_i, slot_j = "agent_1", "agent_0"
            
            # Train agent i against agent j
            env_i = SingleAgentEnvWrapper(pz_env, slot_i, other_policy=models[j])
            models[i].set_env(env_i)
            print(f"Training {agent_names[i]} (as {slot_i}) vs {agent_names[j]} for {steps_per_match} steps...")
            models[i].learn(total_timesteps=steps_per_match, reset_num_timesteps=False, callback=loggers[i])
            
            # Train agent j against agent i (swap slots)
            env_j = SingleAgentEnvWrapper(pz_env, slot_j, other_policy=models[i])
            models[j].set_env(env_j)
            print(f"Training {agent_names[j]} (as {slot_j}) vs {agent_names[i]} for {steps_per_match} steps...")
            models[j].learn(total_timesteps=steps_per_match, reset_num_timesteps=False, callback=loggers[j])
        
        # Print round summary
        round_elapsed = time.time() - round_start_time
        total_elapsed = time.time() - training_start_time
        print(f"\n--- Round {round_idx + 1} Summary (took {round_elapsed:.1f}s, total: {total_elapsed:.1f}s) ---")
        for i in range(n_agents):
            logger = loggers[i]
            win_rate = logger.wins / len(logger.episode_rewards) if logger.episode_rewards else 0
            print(f"{agent_names[i]}: {logger.wins}W / {logger.losses}L / {logger.draws}D (win rate: {win_rate:.1%})")
        
        # Save checkpoints
        for i in range(n_agents):
            models[i].save(os.path.join(run_dir, f"{agent_names[i]}_round{round_idx + 1}"))
            print(f"Saved checkpoints at round {round_idx + 1}")
    
    total_training_time = time.time() - training_start_time
    print("\n" + "="*50)
    print(f"Training finished! Total time: {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
    print("="*50)
    
    # Save final models
    for i in range(n_agents):
        models[i].save(os.path.join(run_dir, f"{agent_names[i]}_final"))
    print(f"Final models saved to {run_dir}")
    
    # Final summary
    print("\n--- Final Results ---")
    for i in range(n_agents):
        logger = loggers[i]
        total_games = len(logger.episode_rewards)
        win_rate = logger.wins / total_games if total_games else 0
        mean_reward = np.mean(logger.episode_rewards[-100:]) if logger.episode_rewards else 0
        print(f"{agent_names[i]}: {logger.wins}W / {logger.losses}L / {logger.draws}D "
              f"(win rate: {win_rate:.1%}, mean reward: {mean_reward:.1f})")
    
    # Cleanup
    for env in envs.values():
        env.close()
    
    return models

if __name__ == "__main__":
    train_population(
        n_agents=5, 
        steps_per_match=4096, 
        phase1_rounds=50,  # 50 rounds without obstacles
        phase2_rounds=50   # 50 rounds with obstacles
    )
