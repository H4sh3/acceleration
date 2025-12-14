"""Train an agent on the Zombie Survival environment."""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import worms_3d_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


def make_env(log_dir, rank):
    """Create a monitored environment."""
    def _init():
        env = gym.make("ZombieSurvival-v0")
        env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init


def get_run_dir():
    """Create a unique run directory with timestamp."""
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"zombie_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def train(n_envs=8, total_timesteps=5_000_000, resume_from=None):
    print("Setting up Zombie Survival environment for training...")
    
    # Create run directory
    run_dir = get_run_dir()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Using {n_envs} parallel environments")
    
    # Create vectorized environment with multiple workers
    env = SubprocVecEnv([make_env(log_dir, i) for i in range(n_envs)])
    env = VecFrameStack(env, n_stack=8)
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Larger network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 128],  # Policy network
            vf=[256, 256, 128],  # Value function network
        )
    )
    
    # Initialize or load PPO Agent
    if resume_from and os.path.exists(resume_from): 
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env, tensorboard_log=log_dir)
    else:
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.1,  # Exploration
            n_epochs=10,
            clip_range=0.2,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,  # Save every ~50k total steps
        save_path=run_dir,
        name_prefix='zombie_checkpoint'
    )
    
    print(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print("Training finished.")
    
    # Save final model
    final_path = os.path.join(run_dir, "zombie_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train zombie survival agent")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=100_000_000, help="Total training timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    
    args = parser.parse_args()
    
    train(n_envs=args.envs, total_timesteps=args.steps, resume_from=args.resume)
