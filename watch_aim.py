"""Watch a trained aim agent."""
import argparse
import glob
import os

from stable_baselines3 import PPO
from worms_3d_gym.envs.aim_trainer_env import AimTrainerEnv


def find_latest_model(base_dir="models/aim_trainer"):
    """Find the most recent aim trainer model."""
    pattern = os.path.join(base_dir, "**", "*.zip")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError(f"No models found in {base_dir}")
    return max(files, key=os.path.getmtime)


def watch(model_path=None, n_episodes=5, fps=15):
    """Watch the agent play."""
    if model_path is None:
        model_path = find_latest_model()
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    env = AimTrainerEnv(render_mode="human")
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            env.render()
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        
        print(f"Episode {ep+1}: {info['targets_hit']} targets hit")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()
    
    watch(model_path=args.model, n_episodes=args.episodes)
