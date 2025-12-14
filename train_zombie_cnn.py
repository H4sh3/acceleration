"""Train a CNN-based agent on the Zombie Survival environment.

Uses MultiInputPolicy which combines:
- CNN for processing the egocentric grid observation
- MLP for processing the vector observation (health, shots, aim)
"""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


class CustomCNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for the egocentric grid observation.
    
    Processes:
    - "image": (4, 21, 21) egocentric grid -> CNN -> features
    - "vector": (3,) health/shots/aim -> concatenated directly
    """
    
    def __init__(self, observation_space, features_dim=128):
        # Calculate the size of CNN output + vector input
        super().__init__(observation_space, features_dim)
        
        image_space = observation_space.spaces["image"]
        vector_space = observation_space.spaces["vector"]
        
        n_channels = image_space.shape[0]  # 4 channels
        grid_size = image_space.shape[1]   # 21
        
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 21 -> 11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 11 -> 6
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with th.no_grad():
            sample = th.zeros(1, n_channels, grid_size, grid_size)
            cnn_output_size = self.cnn(sample).shape[1]
        
        vector_size = vector_space.shape[0]  # 3
        
        # Combined MLP
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_size + vector_size, features_dim),
            nn.ReLU(),
        )
        
        self._features_dim = features_dim
    
    def forward(self, observations):
        image = observations["image"]
        vector = observations["vector"]
        
        # Process image through CNN
        cnn_features = self.cnn(image)
        
        # Concatenate with vector
        combined = th.cat([cnn_features, vector], dim=1)
        
        # Final linear layer
        return self.linear(combined)


def make_env(log_dir, rank):
    """Create a monitored environment."""
    def _init():
        # Import here so subprocess workers can find the environment
        import worms_3d_gym
        env = gym.make("ZombieSurvivalCNN-v0")
        env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init


def get_run_dir():
    """Create a unique run directory with timestamp."""
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"zombie_cnn_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def train(n_envs=8, total_timesteps=5_000_000, resume_from=None):
    print("Setting up Zombie Survival CNN environment for training...")
    
    # Create run directory
    run_dir = get_run_dir()
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Using {n_envs} parallel environments")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(log_dir, i) for i in range(n_envs)])
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Policy kwargs for custom CNN extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCNNExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    
    # Initialize or load PPO Agent
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env, tensorboard_log=log_dir)
    else:
        model = PPO(
            "MultiInputPolicy",  # Handles Dict observation spaces
            env, 
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.05,  # Reduced from 0.1 for less random exploration
            n_epochs=10,
            clip_range=0.2,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model.policy)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=run_dir,
        name_prefix='zombie_cnn_checkpoint'
    )
    
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print("Training finished.")
    
    # Save final model
    final_path = os.path.join(run_dir, "zombie_cnn_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train zombie survival agent with CNN")
    parser.add_argument("--envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--steps", type=int, default=50_000_000, help="Total training timesteps")
    parser.add_argument("--resume", type=str, default=None, help="Path to model to resume from")
    
    args = parser.parse_args()
    
    train(n_envs=args.envs, total_timesteps=args.steps, resume_from=args.resume)
