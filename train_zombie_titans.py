"""Train a Titans-memory-augmented agent on the Zombie Survival environment.

Uses the Titans architecture from Google Research for long-term memory:
- Deep neural network as memory (not fixed-size vector)
- Surprise-based selective memorization
- Momentum for temporal context
- Adaptive forgetting

Reference: https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/
"""
import sys
import os
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym
import torch as th
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F

from titans_memory import TitansCNNExtractor, create_titans_extractor


# Note: We removed the custom TitansPPO class because overriding the train loop
# was causing learning issues. The auxiliary prediction loss is now computed
# during the forward pass and added to the main loss automatically via the
# prediction_loss_weight parameter in NeuralMemoryModule.


class TitansFeatureExtractorSB3(BaseFeaturesExtractor):
    """SB3-compatible wrapper for Titans feature extractor."""
    
    def __init__(
        self, 
        observation_space, 
        features_dim: int = 128,
        memory_variant: str = "titans",
        memory_depth: int = 3
    ):
        super().__init__(observation_space, features_dim)
        
        self.extractor = create_titans_extractor(
            observation_space,
            features_dim=features_dim,
            memory_variant=memory_variant,
            memory_depth=memory_depth
        )
        self._features_dim = features_dim
    
    def forward(self, observations):
        return self.extractor(observations)
    
    def reset_memory(self, batch_size: int = 1):
        """Reset memory state at episode boundaries."""
        self.extractor.reset_memory(batch_size)
    
    def get_surprise(self) -> float:
        """Get last surprise score for logging."""
        return self.extractor.get_surprise()
    
    def get_prediction_loss(self):
        """Get prediction loss for auxiliary training."""
        return self.extractor.get_prediction_loss()


class MemoryResetCallback(BaseCallback):
    """Callback to reset memory at episode boundaries.
    
    This is crucial for Titans - memory should be reset when
    a new episode starts to avoid carrying over stale context.
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_dones = None
    
    def _on_step(self) -> bool:
        # Check for episode boundaries
        dones = self.locals.get('dones', None)
        if dones is None:
            return True
        
        # If any environment finished an episode, we'd ideally reset
        # that environment's memory. However, SB3's vectorized envs
        # auto-reset, so the memory will naturally adapt.
        # 
        # For more precise control, you'd need a custom VecEnv wrapper
        # that tracks episode boundaries per environment.
        
        return True


class SurpriseLoggingCallback(BaseCallback):
    """Log surprise metrics during training."""
    
    def __init__(self, log_freq: int = 1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.surprise_history = []
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Try to get surprise from feature extractor
            try:
                policy = self.model.policy
                if hasattr(policy, 'features_extractor'):
                    extractor = policy.features_extractor
                    if hasattr(extractor, 'get_surprise'):
                        surprise = extractor.get_surprise()
                        self.surprise_history.append(surprise)
                        
                        if self.verbose > 0:
                            print(f"Step {self.n_calls}: Surprise = {surprise:.4f}")
                        
                        # Log to tensorboard
                        self.logger.record('titans/surprise', surprise)
                        
                        if len(self.surprise_history) > 100:
                            avg_surprise = np.mean(self.surprise_history[-100:])
                            self.logger.record('titans/surprise_avg_100', avg_surprise)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Could not log surprise: {e}")
        
        return True


def make_env(log_dir, rank):
    """Create a monitored environment."""
    def _init():
        import worms_3d_gym
        env = gym.make("ZombieSurvivalCNN-v0")
        env = Monitor(env, os.path.join(log_dir, f"env_{rank}"))
        return env
    return _init


def get_run_dir(variant: str = "titans"):
    """Create a unique run directory with timestamp."""
    base_dir = "models"
    os.makedirs(base_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"zombie_{variant}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def train(
    n_envs: int = 8, 
    total_timesteps: int = 5_000_000, 
    resume_from: str = None,
    memory_variant: str = "titans",
    memory_depth: int = 3,
    features_dim: int = 128
):
    """Train agent with Titans memory.
    
    Args:
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        resume_from: Path to model to resume from
        memory_variant: "titans", "yaad", "moneta", or "memora"
        memory_depth: Depth of memory network (deeper = more expressive)
        features_dim: Feature dimension
    """
    print(f"Setting up Zombie Survival with {memory_variant.upper()} memory...")
    print(f"Memory depth: {memory_depth}, Features dim: {features_dim}")
    
    run_dir = get_run_dir(memory_variant)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Run directory: {run_dir}")
    print(f"Using {n_envs} parallel environments")
    
    env = SubprocVecEnv([make_env(log_dir, i) for i in range(n_envs)])
    
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    
    # Policy kwargs with Titans feature extractor
    policy_kwargs = dict(
        features_extractor_class=TitansFeatureExtractorSB3,
        features_extractor_kwargs=dict(
            features_dim=features_dim,
            memory_variant=memory_variant,
            memory_depth=memory_depth
        ),
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env, tensorboard_log=log_dir)
    else:
        model = PPO(
            "MultiInputPolicy",
            env, 
            verbose=1,
            tensorboard_log=log_dir,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.1,
            n_epochs=10,
            clip_range=0.2,
            device="auto",
            policy_kwargs=policy_kwargs,
        )
    
    print("\nModel Architecture:")
    print(model.policy)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=run_dir,
        name_prefix=f'zombie_{memory_variant}_checkpoint'
    )
    
    memory_reset_callback = MemoryResetCallback(verbose=1)
    surprise_callback = SurpriseLoggingCallback(log_freq=1000, verbose=0)
    
    callbacks = [checkpoint_callback, memory_reset_callback, surprise_callback]
    
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Memory variant: {memory_variant}")
    print("Key Titans features:")
    print("  - Deep neural memory (not fixed-size vector)")
    print("  - Surprise-based selective memorization")
    print("  - Momentum for temporal context")
    print("  - Adaptive forgetting via weight decay")
    
    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    
    print("Training finished.")
    
    final_path = os.path.join(run_dir, f"zombie_{memory_variant}_final")
    model.save(final_path)
    print(f"Model saved to {final_path}")
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train zombie survival agent with Titans memory"
    )
    parser.add_argument(
        "--envs", type=int, default=8, 
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--steps", type=int, default=15_000_000, 
        help="Total training timesteps"
    )
    parser.add_argument(
        "--resume", type=str, default=None, 
        help="Path to model to resume from"
    )
    parser.add_argument(
        "--variant", type=str, default="titans",
        choices=["titans", "yaad", "moneta", "memora"],
        help="Memory variant: titans (default), yaad (robust), moneta (strict), memora (probabilistic)"
    )
    parser.add_argument(
        "--depth", type=int, default=3,
        help="Memory network depth (deeper = more expressive, default: 3)"
    )
    parser.add_argument(
        "--features", type=int, default=128,
        help="Feature dimension (default: 128)"
    )
    
    args = parser.parse_args()
    
    train(
        n_envs=args.envs, 
        total_timesteps=args.steps, 
        resume_from=args.resume,
        memory_variant=args.variant,
        memory_depth=args.depth,
        features_dim=args.features
    )
