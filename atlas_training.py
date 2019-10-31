import gym
import os
import numpy as np
import roboschool

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2


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


# multiprocess environment
n_cpu = 4

env = SubprocVecEnv([lambda: gym.make('RoboschoolAtlasForwardWalk-v1') for j in range(n_cpu)])
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=int(1e6))
# model.save("20191008_RoboschoolAtlasForwardWalk_ppo2_1e6_v1")
# del model

for i in range(82, 111):
    model = PPO2.load('20191008_RoboschoolAtlasForwardWalk_ppo2_' + str(i-1) + 'e6_v1')
    model.set_env(env)
    model.learn(total_timesteps=int(1e6))
    model.save("20191008_RoboschoolAtlasForwardWalk_ppo2_" + str(i) + "e6_v1")
    del model

# env = bipedGymEnv_Phoenix.BipedRobot(isGUI=False, demonstration=False, reset_status=True)
# # model = PPO1(MlpPolicy, env)
# model = PPO1.load('20190920_biped12DOF_ppo1_1e6_v1')
# model.set_env(env)
# # score_before_training = model_evaluation(model, env, 10)
# model.learn(total_timesteps=1e6)
# # score_after_training = model_evaluation(model, env, 10)
# model.save("20190920_biped12DOF_ppo1_1e6_v2")

# env = gym.make("RoboschoolAtlasForwardWalk-v1")
# model = PPO1(MlpPolicy, env)
# model.learn(total_timesteps=1e8)
# model.save("20190920_RoboschoolAtlasForwardWalk_ppo1_1e8_v1")


# with open("20190919_biped12DOF_ppo1_5e6_v1_score_report.txt", "a") as f:
#     f.write("score before training: %s\n" % score_before_training)
#     f.write("score after training: %s\n" % score_after_training)

# del model  # remove to demonstrate saving and loading

# env = bipedGymEnv_Phoenix.BipedRobot(isGUI=True, demonstration=True)
# env = gym.make("RoboschoolAtlasForwardWalk-v1")
# model = PPO2.load('20191008_RoboschoolAtlasForwardWalk_ppo2_81e6_v1')
#
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
#
