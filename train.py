import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import worms_3d_gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

def make_env(log_dir):
    """Create a monitored environment."""
    def _init():
        env = gym.make("Worms3D-v0", render_mode=None)
        env = Monitor(env, log_dir)  # Wrap with Monitor to track ep_rew and ep_len
        return env
    return _init

def get_run_dir():
    """Create a unique run directory with timestamp."""
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    # Use timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir

def train():
    print("Setting up Worms3D environment for training...")
    
    # Create run directory
    run_dir = get_run_dir()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    
    # Initialize environment with Monitor wrapper for episode stats
    env = DummyVecEnv([make_env(log_dir)])
    env = VecFrameStack(env, n_stack=4)  # Stack last 4 observations
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Initialize PPO Agent
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.2,  # High exploration to learn movement
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=run_dir,
        name_prefix='checkpoint'
    )
    
    print("Starting training...")
    total_timesteps = 5_000_000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print("Training finished.")
    
    # Save final model
    final_path = os.path.join(run_dir, "final_model")
    model.save(final_path)
    print(f"Model saved to {final_path}")

if __name__ == "__main__":
    train()
