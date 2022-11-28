import gym
import pybullet_data
from gym import spaces
import numpy as np
import pybullet as p
import time
from datetime import datetime
import matplotlib.pyplot as plt
import os

from plane import *
from biped_robot import *
import motor


class wdSim(gym.Env):  # gym实际上是一个RL的架子，模拟是在pybullet中模拟，然后在二者之间转化observation and action
    # gym -> ob,action -> position, angle, velocity, acceleration -> pybullet,
    # pybullet do moving -> position, angle,,,-> ob, action -> gym
    def __init__(self):
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 2}
        self.max_episode_steps = 50
        # 定义动作空间
        # 仿真关节  实物关节
        #    0         1
        #    1         3
        #    2         2
        #    3         4
        self.r_thigh_high = 1  # 0.5
        self.r_shin_high = 1  # 0.5
        self.l_thigh_high = 1  # 0.5
        self.l_shin_high = 1  # 0.5
        high = np.array([self.r_thigh_high, self.r_shin_high, self.l_thigh_high, self.l_shin_high], dtype=np.float32)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)

        # 定义状态空间   机器人移动距离和方向
        self.observation_space = spaces.Box(low=-high, high=high)  # 这个移动距离也太小了点

        # 连接引擎
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 用到pybullet资源时要加这个
        # 计数器
        self.step_num = 0
        self.run_time = 0.336  # ？
        self.begin_time = datetime.now()
        self.end_time = datetime.now()
        self.angle_queue = []  # ？
        self.angle_queue_i = 0  # ？
        self.position_queue = []  # ？
        self.position_queue_i = 0

        self.plane = None
        self.robot = None
        self.reset()

    def reset(self):  # 重启环境
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8)

        self.plane = Plane(self.client)
        self.robot = BipedRobot(self.client)

        return self.robot.get_observation()

    def step(self, action):  # 前进一步，在pybullet中前进，根据pybullet的结果转化为obersvation，并返回
        # self.render()
        obs_before = self.robot.get_observation()
        location_before, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        self.robot.apply_action(action)
        self.step_num += 1
        state = self.robot.get_observation()

        base = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        angle_tuple = p.getEulerFromQuaternion(base[1])  # -1.4   0.8#角度四元组转为欧拉角
        angle_x1 = angle_tuple[0] - 1.57  # 前后倒
        angle_x2 = angle_tuple[1]  # 左右倒
        angle_x3 = angle_tuple[2]  # 左右转
        if angle_x1 < -1.5 or angle_x1 > 1.5 or abs(angle_x3) > 0.6:
            angle_sum = 100
        else:
            angle_sum = round(abs(angle_x1) + abs(angle_x2) + abs(angle_x3), 2)  # + abs(angle_x3)
        # print("angle_sum = ", angle_sum)

        # reward calculation
        action_reward = 0
        self.angle_queue.append(action)

        if len(self.angle_queue) > 2:
            if self.angle_queue[self.angle_queue_i][0] == self.angle_queue[self.angle_queue_i - 2][0] or \
                    self.angle_queue[self.angle_queue_i][1] == self.angle_queue[self.angle_queue_i - 2][1] or \
                    self.angle_queue[self.angle_queue_i][2] == self.angle_queue[self.angle_queue_i - 2][2] or \
                    self.angle_queue[self.angle_queue_i][3] == self.angle_queue[self.angle_queue_i - 2][3]:
                # print("gg")
                action_reward = -100
        self.angle_queue_i = self.angle_queue_i + 1

        # 如果脚的动作是极限值，应该被否定，
        if abs(action[0]) == 1 or abs(action[1]) == 1 or abs(action[2]) == 1 or abs(action[3]) == 1:
            action_reward = -100

        if action_reward == 0:
            # 如果前后动作差距过小，认为此动作是最差动作，相当于摔倒

            if abs(obs_before[0] - action[0]) >= 0.1:
                action_reward += 1
            if abs(obs_before[1] - action[1]) >= 0.1:
                action_reward += 1
            if abs(obs_before[2] - action[2]) >= 0.1:
                action_reward += 1
            if abs(obs_before[3] - action[3]) >= 0.1:
                action_reward += 1
            if action_reward < 2:  # 如果有一个或0个关节变化达标，也不行
                action_reward = -100
            else:  # 至少有两个关节能够有变化，表示此动作有效果
                action_reward = round((abs(obs_before[0] - action[0]) + abs(obs_before[1] - action[1]) + abs(
                    obs_before[2] - action[2]) + abs(obs_before[3] - action[3])) * 0.2, 2)
        location, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(
            cameraDistance=0.5,
            cameraYaw=40,
            cameraPitch=-10,
            cameraTargetPosition=location
        )

        position_reward = 0
        self.position_queue.append(location[1])
        if len(self.position_queue) > 2:
            if self.position_queue[self.position_queue_i] >= self.angle_queue[self.position_queue_i - 2][0]:
                # print("gg")
                position_reward = -100
        self.position_queue_i = self.position_queue_i + 1
        if location[1] > 0.05:
            position_reward = -100
        else:
            position_reward = 0.5
        if position_reward != -100:
            position_reward = -250 * (location[1] - location_before[1]) + position_reward
        reward = round(position_reward - angle_sum + action_reward, 2)
        if self.step_num > 29 or angle_sum == 100 or action_reward == -100 or position_reward == -100:
            done = True
            self.step_num = 0
        else:
            done = False
        info = {}
        return state, reward, done, info

    def seed(self, seed=None):
        pass

    def render(self, mode="human"):  # 渲染
        width = 800
        height = 800
        rendered_img = plt.imshow(np.zeros((width, height, 4)))
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.1, farVal=100)
        robot_pos, robot_ori = [list(l) for l in
                    p.getBasePositionAndOrientation(self.robot, self.client)]
        view_matrix = p.computeViewMatrix(
            robot_pos + np.array([0.4, -0.4, 0]),
            robot_pos,
            np.array([0, 0, 1])
        )
        (_, _, px, _, _) = p.getCameraImage(
            width, height, view_matrix, proj_matrix
        )
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        if mode == "rgb_array":
            return rgb_array
        elif mode == "human":
            rendered_img.set_data(rgb_array)
            plt.draw()
            plt.pause(1/240)
        else:
            return None

    def close(self):
        if self.client >= 0:
            p.disconnect(self.client)
        self.client = -1


