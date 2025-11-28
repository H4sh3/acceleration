# Worms 3D Gym Environment

An OpenAI Gym (Gymnasium) environment for a Multi-Agent 3D Artillery game similar to Worms.

## Features
- **3 Teams of 3 Agents** (Total 9 agents).
- **3D Voxel Map**: Fully destructible terrain.
- **3 Weapons**:
    - **Shot**: Instant hit-scan.
    - **Rocket**: Projectile with explosive radius (destroys terrain).
    - **Baseball Bat**: Melee knockback.
- **Physics**: Simple gravity, collision, and velocity.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Run the demo script to see the agents in action. If a trained model exists, it will use it; otherwise, it runs random actions.
```bash
python run_demo.py
```

To train the agents using PPO:
```bash
python train.py
```

## Environment Details
- **Observation Space**: `Dict`
    - `map`: 3D Voxel Grid (30x30x20)
    - `agents`: Array of agent states [x, y, z, health, team, yaw, pitch]
- **Action Space**: `MultiDiscrete` (Flattened for all 9 agents)
    - Controls all agents simultaneously as a single "super-agent".
    - Shape: `[MoveX, MoveY, Jump, Yaw, Pitch, Fire]` * 9 agents.

## Rendering
The environment uses Matplotlib for 3D visualization. `run_demo.py` will display the plot step-by-step.
