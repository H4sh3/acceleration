"""Train an agent on the aim trainer environment."""
import os
import argparse
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from worms_3d_gym.envs.aim_trainer_env import AimTrainerEnv


class TargetHitCallback(BaseCallback):
    """Log targets hit during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_targets = []
    
    def _on_step(self):
        # Check for episode ends
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals.get("infos", [{}])[i]
                targets = info.get("targets_hit", 0)
                self.episode_targets.append(targets)
                
                if len(self.episode_targets) % 100 == 0:
                    recent = self.episode_targets[-100:]
                    avg = sum(recent) / len(recent)
                    print(f"Episodes: {len(self.episode_targets)}, "
                          f"Avg targets (last 100): {avg:.1f}")
        return True


def make_env():
    def _init():
        return AimTrainerEnv()
    return _init


def train(total_timesteps=500_000, n_envs=8, save_dir="models/aim_trainer"):
    """Train PPO agent on aim trainer."""
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    env = VecMonitor(env)
    
    # Create eval environment
    eval_env = AimTrainerEnv()
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=10000 // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    target_callback = TargetHitCallback()
    
    # Create model - small network for simple task
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        verbose=1,
        tensorboard_log=run_dir
    )
    
    print(f"Training for {total_timesteps:,} timesteps...")
    print(f"Saving to: {run_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, target_callback],
        progress_bar=False
    )
    
    # Save final model
    final_path = os.path.join(run_dir, "aim_agent_final")
    model.save(final_path)
    print(f"Saved final model to: {final_path}.zip")
    
    env.close()
    eval_env.close()
    
    return final_path + ".zip"


def evaluate(model_path, n_episodes=10):
    """Evaluate a trained model."""
    model = PPO.load(model_path)
    env = AimTrainerEnv()
    
    total_targets = 0
    total_rewards = 0
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            done = term or trunc
        
        targets = info.get("targets_hit", 0)
        total_targets += targets
        total_rewards += ep_reward
        print(f"Episode {ep+1}: targets={targets}, reward={ep_reward:.1f}")
    
    print(f"\nAverage: {total_targets/n_episodes:.1f} targets, "
          f"{total_rewards/n_episodes:.1f} reward")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--eval", type=str, help="Path to model to evaluate")
    args = parser.parse_args()
    
    if args.eval:
        evaluate(args.eval)
    else:
        model_path = train(total_timesteps=args.timesteps, n_envs=args.envs)
        print("\nEvaluating trained model...")
        evaluate(model_path)
