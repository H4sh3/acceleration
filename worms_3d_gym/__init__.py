from gymnasium.envs.registration import register

register(
    id="Worms3D-v0",
    entry_point="worms_3d_gym.envs:Worms3DEnv",
)
