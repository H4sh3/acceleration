from gymnasium.envs.registration import register

register(
    id="Worms3D-v0",
    entry_point="worms_3d_gym.envs:Worms3DEnv",
)

register(
    id="ZombieSurvival-v0",
    entry_point="worms_3d_gym.envs:ZombieSurvivalEnv",
)

register(
    id="ZombieSurvivalCNN-v0",
    entry_point="worms_3d_gym.envs:ZombieSurvivalCNNEnv",
)
