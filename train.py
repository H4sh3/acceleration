import gymnasium as gym
import worms_3d_gym
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

def train():
    print("Setting up Worms3D environment for training...")
    
    # Create log dir
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment
    # We use DummyVecEnv for SB3
    env = gym.make("Worms3D-v0", render_mode=None)
    
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
        ent_coef=0.1,  # More exploration
    )
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='worms_model'
    )
    
    print("Starting training...")
    total_timesteps = 1000000
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    
    print("Training finished.")
    
    # Save final model
    model.save("models/worms_final_model")
    print("Model saved to models/worms_final_model")

if __name__ == "__main__":
    train()
