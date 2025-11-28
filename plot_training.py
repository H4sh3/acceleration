import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import os

def load_tensorboard_logs(log_dir="logs"):
    """Load training data from tensorboard logs."""
    # Find the latest run
    runs = [d for d in os.listdir(log_dir) if d.startswith("PPO_")]
    if not runs:
        print("No PPO runs found in logs/")
        return None
    
    latest_run = sorted(runs)[-1]
    run_path = os.path.join(log_dir, latest_run)
    
    print(f"Loading from: {run_path}")
    
    # Find event file
    event_files = [f for f in os.listdir(run_path) if f.startswith("events")]
    if not event_files:
        print("No event files found")
        return None
    
    event_path = os.path.join(run_path, event_files[0])
    
    # Load events
    ea = event_accumulator.EventAccumulator(event_path)
    ea.Reload()
    
    return ea

def plot_training_curves(ea):
    """Plot training curves from tensorboard data."""
    if ea is None:
        return
    
    # Get available tags
    tags = ea.Tags()
    scalar_tags = tags.get('scalars', [])
    print(f"Available metrics: {scalar_tags}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Statistics", fontsize=14)
    
    # Plot 1: Episode Reward
    if 'rollout/ep_rew_mean' in scalar_tags:
        data = ea.Scalars('rollout/ep_rew_mean')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 0].plot(steps, values, 'b-', linewidth=1)
        axes[0, 0].set_title('Episode Reward (Mean)')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Length
    if 'rollout/ep_len_mean' in scalar_tags:
        data = ea.Scalars('rollout/ep_len_mean')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[0, 1].plot(steps, values, 'g-', linewidth=1)
        axes[0, 1].set_title('Episode Length (Mean)')
        axes[0, 1].set_xlabel('Timesteps')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Policy Loss
    if 'train/policy_gradient_loss' in scalar_tags:
        data = ea.Scalars('train/policy_gradient_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 0].plot(steps, values, 'r-', linewidth=1)
        axes[1, 0].set_title('Policy Gradient Loss')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Value Loss
    if 'train/value_loss' in scalar_tags:
        data = ea.Scalars('train/value_loss')
        steps = [d.step for d in data]
        values = [d.value for d in data]
        axes[1, 1].plot(steps, values, 'm-', linewidth=1)
        axes[1, 1].set_title('Value Loss')
        axes[1, 1].set_xlabel('Timesteps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Saved training_curves.png")
    plt.show()

if __name__ == "__main__":
    ea = load_tensorboard_logs()
    plot_training_curves(ea)
