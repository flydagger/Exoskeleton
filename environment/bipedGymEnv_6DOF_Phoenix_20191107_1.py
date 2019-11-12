"""
Author: Phoenix Fan
Date: 07-11-2019
Specification:  Increase x_coefficient to 100.

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

    def __init__(self, isGUI=True, useFixedBase=False, demonstration=False, reset_status=False,
                 control_mode=p.VELOCITY_CONTROL, log_flag=False, log_name='logging_Exoskeleton.txt'):

        super()
        self.log_flag = log_flag
        if self.log_flag is True:
            logging.basicConfig(filename=log_name, level=logging.DEBUG,
                                format='%(asctime)s - %(levelname)s - %(message)s')
        self.count_episode = 0
        self.number_joint = 0
        self.number_link = 0
        self.nan_recorded = True  # If the status of environment has been recorded, this variable becomes False.
        self.time_limit = 1  # time_limit is the maximum time of a episode for standing purpose
        self.step_counter = 0
        self.p = p
        self.bot_id = 0  # the id of the robot in the environment
        self.maxForce = 1000
        self.GRAVITY = -9.8  # gravity of the environment
        self.timeStep = 0.01  # unclear variable
        self.demonstration = demonstration  # a sign indicating whether activate time.sleep()
        self.control_mode = control_mode  # VELOCITY_CONTROL=0; TORQUE_CONTROL=1; POSITION_CONTROL=2
        # the vertical length of torso is 0.4, the vertical length of legs is 0.6
        # the assumptive angle of legs is 60 degree
        # self.highestTorsoCenterHeight = 0.6 + 0.4 / 2
        self.highestTorsoCenterHeight = 0.8
        self.lowestTorsoCenterHeight = 0.7 * math.sin(60 * math.pi / 180) + 0.1
        self.avgTorsoCenterHeight = (self.highestTorsoCenterHeight + self.lowestTorsoCenterHeight) / 2
        self.deadLine = 0.2  # If the center_datum is lower than this height, it is dead.
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
        self.p.loadSDF("/home/antikythera1/workplace/bullet3-master/examples/pybullet/gym/pybullet_data/stadium.sdf")
        self.cubeStartPos = [0, 0, self.highestTorsoCenterHeight]
        self.cubeStartOrientation = self.p.getQuaternionFromEuler([0, 0, 0])
        self.previous_velocity_link = np.zeros([6, 3], dtype=float)
        self.previous_orientation_link = np.zeros([6, 4], dtype=float)
        self.previous_position_link = np.zeros([6, 3], dtype=float)
        self.previous_status_joint = np.zeros([6, 2], dtype=float)  # velocity, position
        self.previous_base_status = np.zeros([7, ], dtype=float)  # velocity, orientation
        self.torso_position = np.zeros([2, 3, ], dtype=float)  # previous position, current position
        self.center_datum = np.zeros([2, 3, ], dtype=float)  # previous, current
        self.load_bot()
        self.status_record = np.zeros(shape=(100, 42), dtype=float)
        self.status_start_pointer = 0
        self.seed()
        self.get_limitation_space()
        self.avg_x_reward, self.avg_y_reward, self.avg_z_reward, self.avg_survival_reward = 0., 0., 0., 0.
        self.avg_joint_efficiency = np.zeros(shape=[6, ], dtype=float)
        self.total_reward = 0.

    def load_bot(self):
        loc = '/home/antikythera1/workplace/Exoskeleton/biped_6DOF_20191104.urdf'
        self.center_datum[:, :] = 0.
        self.bot_id = self.p.loadURDF(loc,
                                      self.cubeStartPos,
                                      self.cubeStartOrientation,
                                      useFixedBase=self.useFixedBase)
        self.number_link = self.number_joint = self.p.getNumJoints(self.bot_id)

        self.previous_base_status[0] = self.p.getBaseVelocity(self.bot_id)[1][0]  # x velocity
        self.previous_base_status[1] = self.p.getBaseVelocity(self.bot_id)[1][1]  # y velocity
        self.previous_base_status[2] = self.p.getBaseVelocity(self.bot_id)[1][2]  # z velocity
        self.previous_base_status[3] = self.p.getBasePositionAndOrientation(self.bot_id)[1][0]  # x quaternion
        self.previous_base_status[4] = self.p.getBasePositionAndOrientation(self.bot_id)[1][1]  # y quaternion
        self.previous_base_status[5] = self.p.getBasePositionAndOrientation(self.bot_id)[1][2]  # z quaternion
        self.previous_base_status[6] = self.p.getBasePositionAndOrientation(self.bot_id)[1][3]  # w quaternion
        self.center_datum[1] += self.p.getBasePositionAndOrientation(self.bot_id)[0]  # base position
        self.torso_position[0] = self.p.getBasePositionAndOrientation(self.bot_id)[0]  # store the previous base position
        self.torso_position[1] = self.p.getBasePositionAndOrientation(self.bot_id)[0]  # store the current base position

        for index_link in range(self.number_link):
            self.previous_position_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[0]
            self.previous_orientation_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[1]
            self.previous_velocity_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[6]
            self.center_datum[1] += self.p.getLinkState(self.bot_id, index_link, 1)[0]  # center_datum[1] means current center

        self.center_datum[1] /= (self.number_link + 1)  # take base into account

        for index_joint in range(self.number_joint):
            self.previous_status_joint[index_joint, 0] = self.p.getJointState(self.bot_id, index_joint)[1]  # velocity
            self.previous_status_joint[index_joint, 1] = self.p.getJointState(self.bot_id, index_joint)[0]  # position

    def get_limitation_space(self):
        """
        Define observation_space and action_space.
        action_space is the limitation of joint position.
        :return: None
        """
        action_limit = np.zeros([self.number_joint, 2, ], dtype=float)
        # joint_upper_velocity_limit = np.zeros([self.number_joint, ], dtype=float)
        # joint_lower_velocity_limit = np.zeros([self.number_joint, ], dtype=float)
        obs_tmp = np.full([134, 2, ], fill_value=1., dtype=float)  # each column : [lower_limit, upper_limit]
        for joint_index in range(self.number_joint):
            joint_info = self.p.getJointInfo(self.bot_id, joint_index)
            action_limit[joint_index] = [-joint_info[11], joint_info[11]]  # joint velocity limitation for action_space
            obs_tmp[116 + joint_index * 3] = [-joint_info[11], joint_info[11]]  # [joint lower velocity limit, joint upper velocity limit]
            obs_tmp[116 + joint_index * 3 + 1] = [-joint_info[11]*2, joint_info[11]*2]  # acceleration
            obs_tmp[116 + joint_index * 3 + 2] = [joint_info[8] - joint_info[9], joint_info[9] - joint_info[8]]  # relative_position

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

    def step_simulation(self, total=50):
        for i in range(total):
            self.p.stepSimulation()

    def get_observations(self):
        """
        observations = status_base + status_link*6 + status_joint*6 = 14+17*6+3*6 = 134 (6 DOF robot)
            status_base = velocity (3 floats)
                          + acceleration (3 floats)
                          + orientation (4 floats)
                          + orientation_variance (4 floats)
                          = 14 floats
            status_link = velocity (3 floats)
                          + acceleration (3 floats)
                          + orientation (4 floats)
                          + orientation_variance (4 floats)
                          + relative position (3 floats)
                          = 17 floats
                acceleration = velocity[i] - velocity[i-1]
                relative position = position[link] - position[torso]
            status_joint = velocity + acceleration + relative position = 3 floats
                relative position = position[current] - position[previous]
        :return:
        """
        status_base = np.zeros([14, ], dtype=float)
        status_links = np.zeros([self.number_link * 17, ], dtype=float)
        status_joints = np.zeros([self.number_joint * 3, ], dtype=float)

        # obtain the status of base
        base_status_velocity = self.p.getBaseVelocity(self.bot_id)
        base_status_position_orientation = self.p.getBasePositionAndOrientation(self.bot_id)
        status_base[0] = base_status_velocity[1][0]  # x velocity
        status_base[1] = base_status_velocity[1][1]  # y velocity
        status_base[2] = base_status_velocity[1][2]  # z velocity
        status_base[3] = base_status_velocity[1][0] - self.previous_base_status[0]  # x velocity_variance
        status_base[4] = base_status_velocity[1][1] - self.previous_base_status[1]  # y velocity_variance
        status_base[5] = base_status_velocity[1][2] - self.previous_base_status[2]  # z velocity_variance
        status_base[6] = base_status_position_orientation[1][0]  # x quaternion
        status_base[7] = base_status_position_orientation[1][1]  # y quaternion
        status_base[8] = base_status_position_orientation[1][2]  # z quaternion
        status_base[9] = base_status_position_orientation[1][3]  # w quaternion
        status_base[10] = base_status_position_orientation[1][0] - self.previous_base_status[3]  # x quaternion_variance
        status_base[11] = base_status_position_orientation[1][1] - self.previous_base_status[4]  # y quaternion_variance
        status_base[12] = base_status_position_orientation[1][2] - self.previous_base_status[5]  # z quaternion_variance
        status_base[13] = base_status_position_orientation[1][3] - self.previous_base_status[6]  # x quaternion_variance

        for index_link in range(self.number_link):
            link_status = self.p.getLinkState(self.bot_id, index_link, 1)
            status_links[index_link * 3 + 0] = link_status[6][0]  # x velocity
            status_links[index_link * 3 + 1] = link_status[6][1]  # y velocity
            status_links[index_link * 3 + 2] = link_status[6][2]  # z velocity
            status_links[index_link * 3 + 3] = link_status[6][0] - self.previous_velocity_link[index_link][0]  # x acceleration
            status_links[index_link * 3 + 4] = link_status[6][1] - self.previous_velocity_link[index_link][1]  # y acceleration
            status_links[index_link * 3 + 5] = link_status[6][2] - self.previous_velocity_link[index_link][2]  # z acceleration
            status_links[index_link * 3 + 6] = link_status[1][0]  # x quaternion
            status_links[index_link * 3 + 7] = link_status[1][1]  # y quaternion
            status_links[index_link * 3 + 8] = link_status[1][2]  # z quaternion
            status_links[index_link * 3 + 9] = link_status[1][3]  # w quaternion
            status_links[index_link * 3 + 10] = link_status[1][0] - self.previous_orientation_link[index_link][0]  # x difference quaternion
            status_links[index_link * 3 + 11] = link_status[1][1] - self.previous_orientation_link[index_link][1]  # y difference quaternion
            status_links[index_link * 3 + 12] = link_status[1][2] - self.previous_orientation_link[index_link][2]  # z difference quaternion
            status_links[index_link * 3 + 13] = link_status[1][3] - self.previous_orientation_link[index_link][3]  # w difference quaternion
            status_links[index_link * 3 + 14] = link_status[5][0] - base_status_position_orientation[0][0]  # x relative position
            status_links[index_link * 3 + 15] = link_status[5][1] - base_status_position_orientation[0][1]  # y relative position
            status_links[index_link * 3 + 16] = link_status[5][2] - base_status_position_orientation[0][2]  # z relative position

        for index_joint in range(self.number_joint):
            joint_status = self.p.getJointState(self.bot_id, index_joint)
            status_joints[index_joint * 3 + 0] = joint_status[1]  # velocity
            status_joints[index_joint * 3 + 1] = joint_status[1] - self.previous_status_joint[index_joint][0]  # acceleration
            status_joints[index_joint * 3 + 2] = joint_status[0] - self.previous_status_joint[index_joint][1]  # relative position

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
            raise FloatingPointError("Not a number error.")

        result = np.concatenate((status_base, status_links, status_joints), axis=0)
        return result

    def reward(self, dead, observations):
        """
        Robot is expected to move towards positive x direction as fast as possible while preventing falling down.
        :param: True means the robot is dead.
        :return:
        """
        x_reward = self.torso_position[1, 0] - self.torso_position[0, 0]  # deduct previous position from current position
        y_reward = abs(self.center_datum[0, 1]) - abs(self.center_datum[1, 1])
        z_reward = -abs(self.torso_position[1, 2] - self.avgTorsoCenterHeight)
        # joint_efficiency = -sum([abs(observations[i*3+116]) for i in range(6)]) / 6
        joint_efficiency = [(abs(observations[i*3+116])/self.action_space.high[i]) for i in range(6)]
        if dead:
            survival = 0
        else:
            survival = 0.1

        x_coefficient = 100
        y_coefficient = 0.1
        z_coefficient = 0.01
        survival_coefficient = 0.0
        joint_efficiency_coefficient = -0.05

        self.avg_x_reward += x_coefficient*x_reward
        self.avg_y_reward += y_coefficient*y_reward
        self.avg_z_reward += z_coefficient*z_reward
        self.avg_survival_reward += survival_coefficient*survival
        tmp_joint_efficiency = np.dot(joint_efficiency_coefficient, joint_efficiency)
        self.avg_joint_efficiency = [self.avg_joint_efficiency[i] + tmp_joint_efficiency[i] for i in range(self.number_joint)]

        return (x_coefficient*x_reward + y_coefficient*y_reward + z_coefficient*z_reward
                + survival_coefficient*survival + sum(tmp_joint_efficiency))

    def done(self):
        """
        Remove all death criterion except falling down.
        :return:
        """
        # if self.torso_position[1, 2] < self.deadLine or self.time_limit > 241 * 10:
        #     return True
        if self.time_limit > 241 * 10:
            return True
        return False

    def step(self, target_velocity):
        """
        VELOCITY_CONTROL=0; TORQUE_CONTROL=1; POSITION_CONTROL=2
        :return:
        """
        self.p.setJointMotorControlArray(bodyUniqueId=self.bot_id,
                                         jointIndices=[i for i in range(self.number_joint)],
                                         controlMode=self.control_mode,
                                         targetVelocities=target_velocity.tolist())

        self.p.stepSimulation()

        self.torso_position[0] = self.torso_position[1].copy()
        self.torso_position[1] = self.p.getBasePositionAndOrientation(self.bot_id)[0]  # get the current base position
        self.center_datum[0] = self.center_datum[1].copy()
        for index_link in range(self.number_link):
            self.previous_position_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[0]
            self.previous_orientation_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[1]
            self.previous_velocity_link[index_link] = self.p.getLinkState(self.bot_id, index_link, 1)[6]
            self.center_datum[1] += self.p.getLinkState(self.bot_id, index_link, 1)[0]  # center_datum[1] means current center

        self.center_datum[1] /= (self.number_link + 1)  # take base into account

        for index_joint in range(1, self.number_joint):
            self.previous_status_joint[index_joint - 1, 0] = self.p.getJointState(self.bot_id, index_joint)[1]
            self.previous_status_joint[index_joint - 1, 1] = self.p.getJointState(self.bot_id, index_joint)[0]

        done = self.done()  # True means the robot is dead.
        observations = self.get_observations()  # np array (42, )

        reward = self.reward(done, observations)
        self.total_reward += reward

        if self.demonstration:
            time.sleep(1 / 240)

        self.time_limit += 1
        self.step_counter += 1
        return observations, reward, done, {}

    def reset(self):
        # self.p.setRealTimeSimulation(1)

        # logging
        self.count_episode += 1
        if self.log_flag is True and self.count_episode % 50 == 0:
            logging.debug('Episode: %d' % (self.count_episode / 50))
            logging.debug('Total reward in this epsisode: %.6f' % self.total_reward)
            logging.debug('Average x reward: %.6f' % (self.avg_x_reward/self.time_limit))
            logging.debug('Average y reward: %.6f' % (self.avg_y_reward/self.time_limit))
            logging.debug('Average z reward: %.6f' % (self.avg_z_reward/self.time_limit))
            logging.debug('Average survival reward: %.6f' % (self.avg_survival_reward/self.time_limit))
            logging.debug('Average joint efficiency reward: %.6f, %.6f, %.6f, %.6f, %.6f, %.6f' %
                          (self.avg_joint_efficiency[0]/self.time_limit,
                           self.avg_joint_efficiency[1]/self.time_limit,
                           self.avg_joint_efficiency[2]/self.time_limit,
                           self.avg_joint_efficiency[3]/self.time_limit,
                           self.avg_joint_efficiency[4]/self.time_limit,
                           self.avg_joint_efficiency[5]/self.time_limit))

        self.avg_x_reward, self.avg_y_reward, self.avg_z_reward, self.avg_survival_reward = 0., 0., 0., 0.
        self.avg_joint_efficiency = np.zeros(shape=[6, ], dtype=float)
        self.total_reward = 0.

        self.time_limit = 1  # initialize the time_limit variable
        self.p.removeBody(self.bot_id)
        self.p.resetSimulation()
        # self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, self.GRAVITY)

        # self.p.loadBullet("./models/SavedTerrain/plain")
        self.p.loadSDF("/home/antikythera1/workplace/bullet3-master/examples/pybullet/gym/pybullet_data/stadium.sdf")
        self.load_bot()

        # initial status for robot with 6 dof
        if self.reset_status:
            self.p.resetJointState(self.bot_id, 0, self.np_random.uniform(-0.523599, 0.523599))  # torso to right thigh [-30, 30] degree
            self.p.resetJointState(self.bot_id, 1, self.np_random.uniform(0, 0.523599))  # right thigh to shank [0, 30] degree
            self.p.resetJointState(self.bot_id, 2, self.np_random.uniform(-0.261799, 0.261799))  # right shank to foot [-15, 15] degree
            self.p.resetJointState(self.bot_id, 3, self.np_random.uniform(-0.523599, 0.523599))  # torso to left thigh [-30, 30] degree
            self.p.resetJointState(self.bot_id, 4, self.np_random.uniform(0, 0.523599))  # left thigh to shank [0, 30] degree
            self.p.resetJointState(self.bot_id, 5, self.np_random.uniform(-0.261799, 0.261799))  # left shank to foot [-15, 15] degree

        # # update the status of link and joint
        # for index_link in range(self.number_link):
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
