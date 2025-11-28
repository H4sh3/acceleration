import gymnasium as gym
import worms_3d_gym

env = gym.make("Worms3D-v0")
print(f"Env class: {env.unwrapped.__class__}")
print(f"Action space type: {type(env.action_space)}")
print(f"Action space: {env.action_space}")
