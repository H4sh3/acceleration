from gymnasium.envs.registration import register

register(
    id="ZombieSurvival-v0",
    entry_point="worms_3d_gym.envs:ZombieSurvivalEnv",
)
