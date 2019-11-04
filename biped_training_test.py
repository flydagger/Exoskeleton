import gym
import os
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import PPO1

import bipedGymEnv_6DOF_Phoenix_20191104


def model_evaluation(model, env, episode_num):
    avg_reward = 0.0
    for i in range(episode_num):
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action, _state = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            avg_reward += rewards

    return avg_reward / episode_num


# env = bipedGymEnv_6DOF_Phoenix_20191104.BipedRobot(isGUI=False, demonstration=False, reset_status=True, log_flag=True)
# # env = DummyVecEnv([lambda: env])
# # model = PPO2(MlpPolicy(net_arch=[300, 200, 100]), env, learning_rate=0.00008, nminibatches=128000)
# model = PPO1(MlpPolicy, env)
# model.learn(total_timesteps=1e4)
# model.save("20191104_biped6DOF_ppo1_1e4_v1")

for i in range(1, 333):
    env = bipedGymEnv_6DOF_Phoenix_20191104.BipedRobot(isGUI=False, demonstration=False, reset_status=True, log_flag=True)
#      env = DummyVecEnv([lambda: env])
    if i == 1:
        model = PPO1(MlpPolicy, env)
    else:
        model = PPO1.load('20191104_biped6DOF_ppo1_' + str(i-1) + 'e6_v1')
        model.set_env(env)
    model.learn(total_timesteps=int(1e6))
    model.save("20191104_biped6DOF_ppo1_" + str(i) + "e6_v1")
    del model


# with open("20191015_biped6DOF_ppo1_1e7_v1_score_report.txt", "a") as f:
#     f.write("score before training: %s\n" % score_before_training)
#     f.write("score after training: %s\n" % score_after_training)

# del model  # remove to demonstrate saving and loading
#
# env = bipedGymEnv_6DOF_Phoenix_20191104.BipedRobot(isGUI=True, demonstration=True, reset_status=True)
# # env = gym.make("RoboschoolHumanoid-v1")
# model = PPO1.load('20191104_biped6DOF_ppo1_1e4_v1')
# # model.set_env(env)
# step_counter = 0
#
# while True:
#     done = False
#     obs = env.reset()
#
#     while not done:
#         env.render()
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         step_counter += 1
#         if step_counter % 240 == 0:
#             step_counter = 0
#             print("Action: " + str(action))
