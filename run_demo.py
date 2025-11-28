import gymnasium as gym
import worms_3d_gym
import numpy as np
import os

def main():
    print("Initializing Worms3D-v0 environment...")
    try:
        env = gym.make("Worms3D-v0", render_mode="rgb_array")
        
        model = None
        model_path = "models/worms_final_model.zip"
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        else:
            print("No trained model found. Using random actions.")

        print("Resetting environment...")
        obs, info = env.reset()
        
        frames = []
        
        print("Running simulation loop for 100 steps...")
        for step in range(100):
            if model:
                actions, _ = model.predict(obs, deterministic=True)
            else:
                actions = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Capture frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Only print every 10 steps to reduce spam
            if step % 10 == 0:
                print(f"Step {step+1}: Reward={reward}, Active Teams={info.get('alive_teams')}")
            
            if terminated or truncated:
                print("Episode finished.")
                obs, info = env.reset()
                
        env.close()
        print("Demo completed successfully.")
        
        # Save GIF
        if len(frames) > 0:
            print(f"Saving GIF with {len(frames)} frames...")
            from PIL import Image
            # Convert numpy arrays to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in frames]
            pil_frames[0].save(
                'worms_demo.gif',
                save_all=True,
                append_images=pil_frames[1:],
                duration=100,
                loop=0
            )
            print("Saved worms_demo.gif")
            
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
