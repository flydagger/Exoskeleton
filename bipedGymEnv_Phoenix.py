"""
Author: Phoenix Fan
Date: 19-09-2019
Target: Exoskeleton 12 DOF Run with a speed.
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
                 control_mode=p.TORQUE_CONTROL):

        super()
        self.nan_recorded = True  # If the status of environment has been recorded, this variable becomes False.
        self.time_limit = 0  # time_limit is the maximum time of a episode for standing purpose
        self.step_counter = 0
        self.p = p
        self.botId = 0  # the id of the robot in the envrionment
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
        self.lowestTorsoCenterHeight = 0.6 * math.sin(60 * math.pi / 180) + 0.4 / 2
        self.avgTorsoCenterHeight = (self.highestTorsoCenterHeight + self.lowestTorsoCenterHeight) / 2
        self.deadLine = 0.4  # If the center of torso is lower than this height, it is dead.
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
        # self.p.setTimeStep(self.timeStep)
        #        self.p.setTimeStep()
        #        self.planeId = self.p.loadURDF("./models/plane.urdf")
        self.p.loadSDF("/home/antikythera1/workplace/bullet3-master/examples/pybullet/gym/pybullet_data/stadium.sdf")
        self.cubeStartPos = [0, 0, self.highestTorsoCenterHeight]
        self.cubeStartOrientation = self.p.getQuaternionFromEuler([0, 0, 0])
        self.loadBot()
        self.joint_number = self.p.getNumJoints(self.botId) - 1  # number of useful joints
        self.status_record = np.zeros(shape=(100, 42), dtype=float)
        self.status_start_pointer = 0
        self.seed()
        self.gatherJointsInfo()
        self.testJoints(self.botId)
        self.previous_position = []
        # self.p.setRealTimeSimulation(1)

    def loadBot(self, loc='biped_6DOF.urdf'):
        loc = '/home/antikythera1/workplace/Exoskeleton_1/biped_12DOF.urdf'
        self.botId = self.p.loadURDF(loc,
                                     self.cubeStartPos,
                                     self.cubeStartOrientation,
                                     useFixedBase=self.useFixedBase)
        self.previous_position = self.cubeStartPos

    def gatherJointsInfo(self):
        """

        :param botId:
        :return:
        """
        jointAmount = self.p.getNumJoints(self.botId)
        velocity_spaces = np.zeros([self.joint_number, 2, ], dtype=float)  # speed limit of each joint
        joints_limit_low = np.zeros([self.joint_number * 2, ],
                                    dtype=float)  # [0, self.joint_number-1] joint position limit, [self.joint_number, self.joint_number*2-1] joint velocity limit
        joints_limit_high = np.zeros([self.joint_number * 2, ],
                                     dtype=float)  # [0, self.joint_number-1] joint position limit, [self.joint_number, self.joint_number*2-1] joint velocity limit
        feet_limit_low = np.array(
            [10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.])  # xyz position, xyz velocity
        feet_limit_high = np.array([10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10.],
                                   dtype=float)  # xyz position, xyz velocity
        torso_limit_low = np.array([10., 10., 10., 10., 10., 10.], dtype=float)  # xyz position, xyz velocity
        torso_limit_high = np.array([10., 10., 10., 10., 10., 10.], dtype=float)  # xyz position, xyz velocity
        for jointIndex in range(1, jointAmount):
            JInfo = self.p.getJointInfo(self.botId, jointIndex)
            velocity_spaces[jointIndex - 1] = [-JInfo[10], JInfo[10]]  # velocity limit

            joints_limit_low[jointIndex - 1] = JInfo[8]  # lower position limit
            joints_limit_high[jointIndex - 1] = JInfo[9]  # upper position limit
            joints_limit_low[jointIndex - 1 + self.joint_number] = -JInfo[11]  # lower velocity limit
            joints_limit_high[jointIndex - 1 + self.joint_number] = JInfo[11]  # upper velocity limit

        lowVs, highVs = zip(*velocity_spaces)
        self.action_space = spaces.Box(np.array(lowVs), np.array(highVs), dtype=np.float)
        limit_low = np.concatenate((joints_limit_low, feet_limit_low, torso_limit_low), axis=0)
        limit_high = np.concatenate((joints_limit_high, feet_limit_high, torso_limit_high), axis=0)
        self.observation_space = spaces.Box(low=np.array(limit_low), high=np.array(limit_high), dtype=float)
        return

    def testJoints(self, botId=0):
        for jointId in range(self.p.getNumJoints(self.botId)):
            self.p.setJointMotorControl2(self.botId, jointId, self.control_mode, force=self.maxForce)

    def disconnect(self):
        self.p.disconnect()

    def realTimeSimulation(self):
        self.p.setRealTimeSimulation(1)

    def stepSimulation(self, total=50, sleep=0.01):
        for i in range(total):
            self.p.stepSimulation()

    def getObservations(self, botId=0, totalStep=0):
        Nj = self.p.getNumJoints(botId)
        # self.joint_number = Nj - 1
        jointStatus = np.zeros([self.joint_number * 2, ], dtype=float)
        feetStatus = np.zeros([12, ], dtype=float)
        torsoStatus = np.zeros([6, ], dtype=float)
        for i in range(Nj):
            # joint_info = self.p.getJointInfo(botId, i)
            joint_state = self.p.getJointState(botId, i)
            link_state = self.p.getLinkState(botId, i, 1)
            if i == 0:
                torsoStatus[0] = link_state[0][0]  # x position
                torsoStatus[1] = link_state[0][1]  # y position
                torsoStatus[2] = link_state[0][2]  # z position
                torsoStatus[3] = link_state[6][0]  # x velocity
                torsoStatus[4] = link_state[6][1]  # y velocity
                torsoStatus[5] = link_state[6][2]  # z velocity
            else:
                jointStatus[i - 1] = joint_state[0]  # joint position value [0, self.joint_number-1]
                jointStatus[i - 1 + self.joint_number] = joint_state[
                    1]  # joint velocity value [self.joint_number, self.joint_number*2-1]
                if i == (self.joint_number - 1) / 2:  # right foot
                    feetStatus[0] = link_state[0][0]  # x position
                    feetStatus[1] = link_state[0][1]  # y position
                    feetStatus[2] = link_state[0][2]  # z position
                    feetStatus[3] = link_state[6][0]  # x velocity
                    feetStatus[4] = link_state[6][1]  # y velocity
                    feetStatus[5] = link_state[6][2]  # z velocity
                elif i == self.joint_number - 1:  # left foot
                    feetStatus[6] = link_state[0][0]  # x position
                    feetStatus[7] = link_state[0][1]  # y position
                    feetStatus[8] = link_state[0][2]  # z position
                    feetStatus[9] = link_state[6][0]  # x velocity
                    feetStatus[10] = link_state[6][1]  # y velocity
                    feetStatus[11] = link_state[6][2]  # z velocity

        joint_nan_flag, feet_nan_flag, torso_nan_flag = False, False, False
        for i in jointStatus:
            if math.isnan(i):
                joint_nan_flag = True
                break

        for i in feetStatus:
            if math.isnan(i):
                feet_nan_flag = True
                break

        for i in torsoStatus:
            if math.isnan(i):
                torso_nan_flag = True
                break

        if joint_nan_flag or feet_nan_flag or torso_nan_flag:
            raise Exception("Not a number occured.")

        result = np.concatenate((jointStatus, feetStatus, torsoStatus), axis=0)
        # self.status_record[self.status_start_pointer] = result
        # if self.status_start_pointer >= self.status_record.shape[0]:
        #     self.status_start_pointer = 0
        # if self.step_counter % 1e5 == 0:
        #     with open("status_record.txt", "a") as f:
        #         f.write("Step: %s. \n%s\n" % (self.step_counter, str(result)))
        #     logging.debug("Step: %s. \n%s" % (self.step_counter, str(result)))
        # if self.nan_recorded and (joint_nan_flag or feet_nan_flag or torso_nan_flag):
        #     self.nan_recorded = False
        #     with open("status_record.txt", "a") as f:
        #         f.write("Step: %s\n" % self.step_counter)
        #         row_num = self.status_start_pointer
        #         for i in range(100):
        #             f.write("%s\n" % self.status_record[row_num])
        #             row_num += 1
        #             if row_num >= 100:
        #                 row_num = 0
        #         f.write("The status firstly contains Nan:\n%s\n" % str(result))
        #     logging.debug("Step: %s. Nan occurs.\n%s" % (self.step_counter, str(result)))
        #     logging.debug("%s" % self.status_start_pointer)
        #     logging.debug("%s" % self.status_record)

        # the number of joint * 2 + xyz coordinates and xyz velocity of two foot and the torso
        if result.__len__() == (self.joint_number * 2 + 12 + 6):
            return result
        else:
            return None

    def reward(self, currenttorso_position):
        """
        Target is running with a speed target about 1 m/s.
        :param currenttorso_position:
        :return:
        """
        x_reward = currenttorso_position[0] - self.previous_position[0]
        y_reward = -abs(currenttorso_position[1])
        z_reward = -abs(currenttorso_position[2] - self.avgTorsoCenterHeight)
        speed_reward = -abs(x_reward * 240 - 1)
        return x_reward + y_reward + z_reward + speed_reward

    def done(self, torso_position):
        """
        Target is running. Horizontal limit of x is [-1, 10]. Horizontal limit of y is [-1, 1].
        :param torso_position:
        :return:
        """
        if torso_position[2] < self.deadLine \
                or self.time_limit > 240 * 10 \
                or torso_position[1] > 1 \
                or torso_position[1] < -1 \
                or torso_position[0] > 10 \
                or torso_position[0] < -1:  # assume 240 steps nearly equals to one second
            return True
        return False

    def step(self, action, total_step=0):
        """
        VELOCITY_CONTROL=0; TORQUE_CONTROL=1; POSITION_CONTROL=2
        :param action:
        :param total_step:
        :return:
        """
        # action *= 100
        self.p.setJointMotorControlArray(bodyUniqueId=self.botId,
                                         jointIndices=[i for i in range(1, self.joint_number + 1)],
                                         controlMode=self.control_mode,
                                         forces=action)

        self.p.stepSimulation()
        torso_position = self.p.getLinkState(self.botId, 0)[0]  # Cartesian position of torso, (x, y, z)
        done = self.done(torso_position)  # True means the robot is dead.
        observation = self.getObservations(self.botId, total_step)  # np array (42, )

        reward = self.reward(torso_position)
        self.previous_position = torso_position  # Cartesian position of torso, (x, y, z)
        if self.demonstration:
            time.sleep(1 / 240)

        self.time_limit += 1
        self.step_counter += 1
        return observation, reward, done, {}

    def reset(self):
        # self.p.setRealTimeSimulation(1)
        self.time_limit = 0  # initialize the time_limit variable
        self.p.removeBody(self.botId)
        self.p.resetSimulation()
        # self.p.setTimeStep(self.timeStep)
        self.p.setGravity(0, 0, self.GRAVITY)

        # self.p.loadBullet("./models/SavedTerrain/plain")
        self.p.loadSDF("/home/antikythera1/workplace/bullet3-master/examples/pybullet/gym/pybullet_data/stadium.sdf")
        self.loadBot()

        # # initial status for robot with 6 dof
        # if self.reset_status:
        #     self.p.resetJointState(self.botId, 1, self.np_random.uniform(-0.523599, 0.523599))  # torso to right thigh [-30, 30] degree
        #     self.p.resetJointState(self.botId, 2, self.np_random.uniform(0, 0.523599))  # right thigh to shank [0, 30] degree
        #     self.p.resetJointState(self.botId, 3, self.np_random.uniform(-0.261799, 0.261799))  # right shank to foot [-15, 15] degree
        #     self.p.resetJointState(self.botId, 4, self.np_random.uniform(-0.523599, 0.523599))  # torso to left thigh [-30, 30] degree
        #     self.p.resetJointState(self.botId, 5, self.np_random.uniform(0, 0.523599))  # left thigh to shank [0, 30] degree
        #     self.p.resetJointState(self.botId, 6, self.np_random.uniform(-0.261799, 0.261799))  # left shank to foot [-15, 15] degree

        # initial status for robot with 12 dof
        if self.reset_status:
            self.p.resetJointState(self.botId, 1, self.np_random.uniform(-0.087267, 0.087267))  # torso to right virtual hip [-5, 5] degree
            self.p.resetJointState(self.botId, 2, self.np_random.uniform(-0.523599, 0.523599))  # right virtual hip to upper thigh [-30, 30] degree
            self.p.resetJointState(self.botId, 3, self.np_random.uniform(-0.261799, 0.261799))  # right upper thigh to lower thigh [-15, 15] degree
            self.p.resetJointState(self.botId, 4, self.np_random.uniform(0, 0.523599))  # right thigh to shank [0, 30] degree
            self.p.resetJointState(self.botId, 5, self.np_random.uniform(-0.087267, 0.087267))  # right shank to ankle [-5, 5] degree
            self.p.resetJointState(self.botId, 6, self.np_random.uniform(-0.261799, 0.261799))  # right ankle to foot [-15, 15] degree
            self.p.resetJointState(self.botId, 7, self.np_random.uniform(-0.087267, 0.087267))  # torso to left virtual hip [-5, 5] degree
            self.p.resetJointState(self.botId, 8, self.np_random.uniform(-0.523599, 0.523599))  # left virtual hip to upper thigh [-30, 30] degree
            self.p.resetJointState(self.botId, 9, self.np_random.uniform(-0.261799, 0.261799))  # left upper thigh to lower thigh [-15, 15] degree
            self.p.resetJointState(self.botId, 10, self.np_random.uniform(0, 0.523599))  # left thigh to shank [0, 30] degree
            self.p.resetJointState(self.botId, 11, self.np_random.uniform(-0.087267, 0.087267))  # left shank to ankle [-5, 5] degree
            self.p.resetJointState(self.botId, 12, self.np_random.uniform(-0.261799, 0.261799))  # left ankle to foot [-15, 15] degree

        if self.demonstration:
            time.sleep(0.1)
        return self.getObservations(self.botId)

    # from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        self.disconnect()

    def render(self, mode="human"):
        return
