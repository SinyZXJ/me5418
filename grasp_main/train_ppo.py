import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gripper_env import GripperEnv

env = GripperEnv()
check_env(env)

model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=100000)  # time stamps random set now :(
model.save("ppo_gripper")

obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
