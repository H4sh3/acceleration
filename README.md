# Worms 2D Arena Shooter

A 2D multi-agent combat environment built with Gymnasium, trained using PPO (Stable Baselines3).

## Overview

Two agents compete in a 30×30 arena with a central obstacle. Each agent must locate, aim at, and shoot the enemy while avoiding being hit. The environment uses egocentric observations with ray sensors, enabling agents to learn spatial awareness and combat tactics.

### Quick Summary

| Component | Details |
|-----------|---------|
| **Arena** | 30 × 30 units with 10 × 10 central obstacle |
| **Agents** | 2 agents, 100 HP each, hitscan shooting |
| **Actions** | 8 discrete: nothing, move (4 dirs), rotate left/right, shoot |
| **Observations** | 28 dims per agent (56 total), 4-frame stacking → 224 dims |
| **Training** | PPO with MlpPolicy, ~500k-2M timesteps |

### Reward Design

The reward structure uses dense, continuous feedback to guide learning. Each step, agents receive:
- **Distance penalty**: −0.015 × (distance / arena_diagonal) — pressure to close in
- **Time penalty**: −0.02 per step — encourages fast fights
- **Enemy in rays**: +0.30 if enemy detected in ±90° cone — facing the right direction
- **Clear LOS**: +1.00 if a shot would hit — the biggest signal ("I can shoot now")
- **Shooting**: −0.04 cost, but +1.50 bonus if LOS clear (net +1.46 for good shots)
- **Damage**: +10 × (damage_dealt / 100), −10 × (damage_taken / 100)
- **Terminal**: +80 for kill, −80 for death

This creates a smooth gradient: agents learn to approach, face the enemy, gain line of sight, and fire only when the shot will land.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```
Models are saved to `models/run_<timestamp>/` with checkpoints every 10k steps.

### Visualization
```bash
python pygame_renderer.py
```
Runs the trained model with a Pygame visualization showing agent observations.

### TensorBoard
```bash
tensorboard --logdir=models
```

## Environment Details

### Observation Space (28 dims per agent)

| Index | Name | Description |
|-------|------|-------------|
| 0-1 | `cos_θ`, `sin_θ` | Agent heading (egocentric) |
| 2 | `v_forward` | Forward velocity (normalized) |
| 3 | `health` | Health (0-1) |
| 4 | `ammo` | Ammo (0-1) |
| 5 | `cooldown` | Weapon cooldown (0-1) |
| 6-7 | `cos_δ`, `sin_δ` | Angle to enemy (relative to heading) |
| 8 | `dist_enemy` | Distance to enemy (normalized by arena diagonal) |
| 9 | `has_los` | Line of sight to enemy (0 or 1) |
| 10-17 | `ray_0` - `ray_7` | 8 ray sensors: distance to wall/obstacle (normalized) |
| 18 | `was_hit` | Hit by enemy last step (0 or 1) |
| 19 | `hit_enemy` | Hit enemy last step (0 or 1) |
| 20-27 | `enemy_ray_0` - `enemy_ray_7` | 8 rays: 1 if enemy detected, 0 otherwise |

### Action Space

| Action | Effect |
|--------|--------|
| 0 | Nothing |
| 1 | Move up (+Y) |
| 2 | Move down (-Y) |
| 3 | Move left (-X) |
| 4 | Move right (+X) |
| 5 | Rotate left (CCW) |
| 6 | Rotate right (CW) |
| 7 | Shoot |

### Reward Structure

| Event | Reward |
|-------|--------|
| Distance penalty | −0.015 × (dist / 42.426) |
| Time penalty | −0.02 per step |
| Enemy in rays (±90°) | +0.30 |
| Clear line of sight | +1.00 |
| Shoot action | −0.04 |
| Shoot with clear LOS | +1.50 (net +1.46) |
| Damage dealt | +10 × (damage / 100) |
| Damage taken | −10 × (damage / 100) |
| Kill enemy | +80 |
| Death | −80 |

## Project Structure

```
├── train.py                 # PPO training script
├── pygame_renderer.py       # Visualization with observation panel
├── worms_3d_gym/
│   └── envs/
│       └── worms_3d_env.py  # Gymnasium environment
├── tests/
│   └── test_observation.py  # Observation function tests
└── models/                  # Saved models and logs
    └── run_<timestamp>/
        ├── checkpoint_*.zip
        ├── final_model.zip
        └── logs/
```

## Configuration

Key constants in `worms_3d_env.py`:

```python
SIZE = 30                    # Arena size
OBSTACLE = [10, 10, 20, 20]  # Central obstacle bounds
MAX_HEALTH = 100.0
MAX_SPEED_FORWARD = 0.3      # Units per step
RAY_MAX_RANGE = 30.0         # Ray sensor range
N_RAYS = 8                   # Number of ray sensors
HIT_THRESHOLD = 1.5          # Beam width for hits
```
