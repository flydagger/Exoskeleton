"""
Author: Phoenix Fan
Date: 25-10-2019
Specification:  Recover penaly on y axis and z axis.
                Change control mode to VELOCITY_CONTROL
"""

import pybullet as p
import pybullet_data
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import time
import os
import logging


class BipedRobot(gym.Env):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def __init__(self, isGUI=True, useFixedBase=False, demonstration=False, reset_status=True,
                 control_mode=p.VELOCITY_CONTROL,):

        super()
        self.number_joint = 7
        self.number_link = 8
        self.nan_recorded = True  # If the status of environment has been recorded, this variable becomes False.
        self.time_limit = 0  # time_limit is the maximum time of a episode for standing purpose
        self.step_counter = 0
        self.p = p
        self.bot_id = 0  # the id of the robot in the envrionment
        self.maxForce = 1000
        self.GRAVITY = -9.8  # gravity of the environment
        self.timeStep = 0.01  # unclear variable
        self.destination = [10, 0, 0]  # the target position of the task: x 10, y 0, z 0.
        self.demonstration = demonstration  # a sign indicating whether activate time.sleep()
        self.control_mode = control_mode  # VELOCITY_CONTROL=0; TORQUE_CONTROL=1; POSITION_CONTROL=2
        # the vertical length of torso is 0.4, the vertical length of legs is 0.6
        # the assumptive angle of legs is 60 degree
        # self.highestTorsoCenterHeight = 0.6 + 0.4 / 2
        self.highestTorsoCenterHeight = 0.8
        self.lowestTorsoCenterHeight = 0.7 * math.sin(60 * math.pi / 180) + 0.1
        self.avgTorsoCenterHeight = (self.highestTorsoCenterHeight + self.lowestTorsoCenterHeight) / 2
        self.deadLine = 0.3  # If the center of torso is lower than this height, it is dead.
        self.p = p
        self.useFixedBase = useFixedBase
        self.reset_status = reset_status
        if isGUI:
            self.physicsClient = self.p.connect(p.GUI)
        else:
            self.physicsClient = self.p.connect(p.DIRECT)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # self.p.resetSimulation()
        # p.setRealTimeSimulation(True)
        self.p.setGravity(0, 0, self.GRAVITY)
        self.p.loadSDF("/home/antikythera2/workplace/bullet3-master/examples/pybullet/gym/pybullet_data/stadium.sdf")
        self.cubeStartPos = [0, 0, self.highestTorsoCenterHeight]
        self.cubeStartOrientation = self.p.getQuaternionFromEuler([0, 0, 0])
        self.previous_velocity_link = np.zeros([self.number_link - 1, 3], dtype=float)
        self.previous_orientation_link = np.zeros([self.number_link - 1, 4], dtype=float)
        self.previous_position_link = np.zeros([self.number_link - 1, 3], dtype=float)
        self.previous_status_joint = np.zeros([self.number_joint - 1, 2], dtype=float)  # velocity, position
        self.center_datum = np.zeros([2, 3, ], dtype=float)  # previous, current
        self.load_bot()
        self.status_record = np.zeros(shape=(100, 42), dtype=float)
        self.status_start_pointer = 0
        self.seed()
        self.get_limitation_space()

    def load_bot(self):
        loc = '/home/antikythera2/workplace/Exoskeleton_1/biped_6DOF_20191024.urdf'
        self.center_datum[:, :] = 0.
        self.bot_id = self.p.loadURDF(loc,
                                      self.cubeStartPos,
                                      self.cubeStartOrientation,
                                      useFixedBase=self.useFixedBase)
        for index_link in range(self.number_link - 1):
            self.previous_position_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[0]
            self.previous_orientation_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[1]
            self.previous_velocity_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[6]
            self.center_datum[1] += self.p.getLinkState(self.bot_id, index_link, 1)[0]  # center_datum[1] means current center

        self.center_datum[1] /= (self.number_link - 1)

        for index_joint in range(1, self.number_joint):
            self.previous_status_joint[index_joint - 1, 0] = self.p.getJointState(self.bot_id, index_joint)[1]  # velocity
            self.previous_status_joint[index_joint - 1, 1] = self.p.getJointState(self.bot_id, index_joint)[0]  # position

    def get_limitation_space(self):
        """
        Define observation_space and action_space.
        action_space is the limitation of joint position.
        :return: None
        """
        action_limit = np.zeros([self.number_joint - 1, 2, ], dtype=float)
        joint_upper_velocity_limit = np.zeros([self.number_joint - 1, ], dtype=float)
        joint_lower_velocity_limit = np.zeros([self.number_joint - 1, ], dtype=float)
        obs_tmp = np.full([137, 2, ], fill_value=10, dtype=float)  # each column : [lower_limit, upper_limit]
        for joint_index in range(1, self.number_joint):
            joint_info = self.p.getJointInfo(self.bot_id, joint_index)
            action_limit[joint_index - 1] = [-joint_info[11], joint_info[11]]  # joint position limitation for action_space
            obs_tmp[119 + (joint_index - 1) * 3, 0] = -joint_info[11]  # joint lower velocity limit
            obs_tmp[119 + (joint_index - 1) * 3, 1] = joint_info[11]  # joint upper velocity limit, namely the other direction
            joint_upper_velocity_limit[joint_index - 1] = joint_info[11]  # joint maximum velocity
            joint_lower_velocity_limit[joint_index - 1] = -joint_info[11]  # joint maximum velocity in the other direction

        # torque_lower_limit, torque_upper_limit = zip(*torque_limit)
        self.action_space = spaces.Box(low=np.array(action_limit[:, 0]), high=np.array(action_limit[:, 1]), dtype=np.float)
        self.observation_space = spaces.Box(low=np.array(obs_tmp[:, 0]), high=np.array(obs_tmp[:, 1]), dtype=float)
        return

    def test_joints(self):
        for jointId in range(self.p.getNumJoints(self.bot_id)):
            self.p.setJointMotorControl2(self.bot_id, jointId, self.control_mode, force=self.maxForce)

    def disconnect(self):
        self.p.disconnect()

    def real_time_simulation(self):
        self.p.setRealTimeSimulation(1)

    def step_simulation(self, total=50, sleep=0.01):
        for i in range(total):
            self.p.stepSimulation()

    def get_observations(self):
        """
        observations = status_link*7 + status_joint*6 = 17*7+3*6 = 137 (6 DOF robot)
            status_link = velocity
                          + acceleration
                          + relative position to torso
                          + orientation (4 floats)
                          + orientation_variance (4 floats)
                          = 17 floats
                acceleration = velocity[i] - velocity[i-1]
                relative position = position[link] - position[torso]
            status_joint = velocity + acceleration + relative position = 3 floats
                relative position = position[current] - position[previous]
        :return:
        """
        status_links = np.zeros([(self.number_link - 1) * 17, ], dtype=float)
        status_joints = np.zeros([(self.number_joint - 1) * 3, ], dtype=float)
        torso_position = self.p.getLinkState(self.bot_id, 0, 1)
        for index_link in range(self.number_link - 1):
            link_status = self.p.getLinkState(self.bot_id, index_link, 1)
            status_links[index_link * 3 + 0] = link_status[6][0]  # x velocity
            status_links[index_link * 3 + 1] = link_status[6][1]  # y velocity
            status_links[index_link * 3 + 2] = link_status[6][2]  # z velocity
            status_links[index_link * 3 + 3] = link_status[6][0] - self.previous_velocity_link[index_link][0]  # x acceleration
            status_links[index_link * 3 + 4] = link_status[6][1] - self.previous_velocity_link[index_link][1]  # y acceleration
            status_links[index_link * 3 + 5] = link_status[6][2] - self.previous_velocity_link[index_link][2]  # z acceleration
            status_links[index_link * 3 + 6] = link_status[0][0] - torso_position[0][0]  # x difference
            status_links[index_link * 3 + 7] = link_status[0][1] - torso_position[0][1]  # y difference
            status_links[index_link * 3 + 8] = link_status[0][2] - torso_position[0][2]  # z difference
            status_links[index_link * 3 + 9] = link_status[1][0]  # x quaternion
            status_links[index_link * 3 + 10] = link_status[1][1]  # y quaternion
            status_links[index_link * 3 + 11] = link_status[1][2]  # z quaternion
            status_links[index_link * 3 + 12] = link_status[1][3]  # w quaternion
            status_links[index_link * 3 + 13] = link_status[1][0] - self.previous_orientation_link[index_link][0]  # x difference quaternion
            status_links[index_link * 3 + 14] = link_status[1][1] - self.previous_orientation_link[index_link][1]  # y difference quaternion
            status_links[index_link * 3 + 15] = link_status[1][2] - self.previous_orientation_link[index_link][2]  # z difference quaternion
            status_links[index_link * 3 + 16] = link_status[1][3] - self.previous_orientation_link[index_link][3]  # w difference quaternion
            
        for index_joint in range(1, self.number_joint):
            joint_status = self.p.getJointState(self.bot_id, index_joint)
            status_joints[(index_joint - 1) * 3 + 0] = joint_status[1]  # velocity
            status_joints[(index_joint - 1) * 3 + 1] = joint_status[1] - self.previous_status_joint[index_joint - 1][0]  # acceleration
            status_joints[(index_joint - 1) * 3 + 2] = joint_status[0] - self.previous_status_joint[index_joint - 1][1]  # relative position

        link_nan_flag, joint_nan_flag = False, False
        for i in status_links:
            if math.isnan(i):
                link_nan_flag = True
                break

        for i in status_joints:
            if math.isnan(i):
                joint_nan_flag = True
                break

        if link_nan_flag or joint_nan_flag:
            raise Exception("Not a number error.")

        result = np.concatenate((status_links, status_joints), axis=0)
        # if result.__len__() == 137:
        #     return result
        # else:
        #     return None
        return result

    def reward(self, dead):
        """
        Robot is expected to move towards positive x direction as fast as possible while preventing falling down.
        :param: True means the robot is dead.
        :return:
        """
        x_reward = self.center_datum[1, 0] - self.center_datum[0, 0]  # deduct previous position from current position
        # y_reward = abs(self.center_datum[0, 1]) - abs(self.center_datum[1, 1])
        # z_reward = -abs(self.center_datum[1, 2] - self.avgTorsoCenterHeight)
        if dead:
            survival = 0
        else:
            survival = 0.1

        return x_reward + survival

    def done(self):
        """
        Remove all death criterion except falling down.
        :return:
        """
        if self.center_datum[1, 2] < self.deadLine or self.time_limit > 240 * 10:
            return True
        return False

    def step(self, action, total_step=0):
        """
        VELOCITY_CONTROL=0; TORQUE_CONTROL=1; POSITION_CONTROL=2
        :param action:
        :param total_step:
        :return:
        """
        # action *= 500
        self.p.setJointMotorControlArray(bodyUniqueId=self.bot_id,
                                         jointIndices=[i for i in range(1, self.number_joint)],
                                         controlMode=self.control_mode,
                                         forces=action)

        self.p.stepSimulation()

        self.center_datum[0] = self.center_datum[1]
        for index_link in range(self.number_link - 1):
            self.previous_position_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[0]
            self.previous_orientation_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[1]
            self.previous_velocity_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[6]
            self.center_datum[1] += self.p.getLinkState(self.bot_id, index_link, 1)[0]  # center_datum[1] means current center

        self.center_datum[1] /= (self.number_link - 1)  # calculate current center

        for index_joint in range(1, self.number_joint):
            self.previous_status_joint[index_joint - 1, 0] = self.p.getJointState(self.bot_id, index_joint)[1]
            self.previous_status_joint[index_joint - 1, 1] = self.p.getJointState(self.bot_id, index_joint)[0]

        done = self.done()  # True means the robot is dead.
        observation = self.get_observations()  # np array (42, )

        reward = self.reward(done)

        if self.demonstration:
            time.sleep(1 / 240)

        self.time_limit += 1
        self.step_counter += 1
        return observation, reward, done, {}

    def reset(self):
        # self.p.setRealTimeSimulation(1)
        self.time_limit = 0  # initialize the time_limit variable
        self.p.removeBody(self.bot_id)
        self.p.resetSimulation()
        # self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, self.GRAVITY)

        # self.p.loadBullet("./models/SavedTerrain/plain")
        self.p.loadSDF("/home/antikythera2/workplace/bullet3-master/examples/pybullet/gym/pybullet_data/stadium.sdf")
        self.load_bot()

        # initial status for robot with 6 dof
        if self.reset_status:
            self.p.resetJointState(self.bot_id, 1, self.np_random.uniform(-0.523599, 0.523599))  # torso to right thigh [-30, 30] degree
            self.p.resetJointState(self.bot_id, 2, self.np_random.uniform(0, 0.523599))  # right thigh to shank [0, 30] degree
            self.p.resetJointState(self.bot_id, 3, self.np_random.uniform(-0.261799, 0.261799))  # right shank to foot [-15, 15] degree
            self.p.resetJointState(self.bot_id, 4, self.np_random.uniform(-0.523599, 0.523599))  # torso to left thigh [-30, 30] degree
            self.p.resetJointState(self.bot_id, 5, self.np_random.uniform(0, 0.523599))  # left thigh to shank [0, 30] degree
            self.p.resetJointState(self.bot_id, 6, self.np_random.uniform(-0.261799, 0.261799))  # left shank to foot [-15, 15] degree

        # # update the status of link and joint
        # for index_link in range(self.number_link - 1):
        #     self.previous_position_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[0]
        #     self.previous_velocity_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[6]
        #
        # for index_joint in range(1, self.number_joint):
        #     self.previous_status_joint[index_joint - 1, 0] = self.p.getJointState(self.bot_id, index_joint)[1]
        #     self.previous_status_joint[index_joint - 1, 1] = self.p.getJointState(self.bot_id, index_joint)[0]

        # # initial status for robot with 12 dof
        # if self.reset_status:
        #     self.p.resetJointState(self.bot_id, 1, self.np_random.uniform(-0.087267, 0.087267))  # torso to right virtual hip [-5, 5] degree
        #     self.p.resetJointState(self.bot_id, 2, self.np_random.uniform(-0.523599, 0.523599))  # right virtual hip to upper thigh [-30, 30] degree
        #     self.p.resetJointState(self.bot_id, 3, self.np_random.uniform(-0.261799, 0.261799))  # right upper thigh to lower thigh [-15, 15] degree
        #     self.p.resetJointState(self.bot_id, 4, self.np_random.uniform(0, 0.523599))  # right thigh to shank [0, 30] degree
        #     self.p.resetJointState(self.bot_id, 5, self.np_random.uniform(-0.087267, 0.087267))  # right shank to ankle [-5, 5] degree
        #     self.p.resetJointState(self.bot_id, 6, self.np_random.uniform(-0.261799, 0.261799))  # right ankle to foot [-15, 15] degree
        #     self.p.resetJointState(self.bot_id, 7, self.np_random.uniform(-0.087267, 0.087267))  # torso to left virtual hip [-5, 5] degree
        #     self.p.resetJointState(self.bot_id, 8, self.np_random.uniform(-0.523599, 0.523599))  # left virtual hip to upper thigh [-30, 30] degree
        #     self.p.resetJointState(self.bot_id, 9, self.np_random.uniform(-0.261799, 0.261799))  # left upper thigh to lower thigh [-15, 15] degree
        #     self.p.resetJointState(self.bot_id, 10, self.np_random.uniform(0, 0.523599))  # left thigh to shank [0, 30] degree
        #     self.p.resetJointState(self.bot_id, 11, self.np_random.uniform(-0.087267, 0.087267))  # left shank to ankle [-5, 5] degree
        #     self.p.resetJointState(self.bot_id, 12, self.np_random.uniform(-0.261799, 0.261799))  # left ankle to foot [-15, 15] degree

        if self.demonstration:
            time.sleep(0.1)
        return self.get_observations()

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.disconnect()

    def render(self, mode="human"):
        return
