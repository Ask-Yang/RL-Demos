import gym
import pybullet_data
from gym import spaces
import numpy as np
import pybullet as p
import time
from datetime import datetime
import motor
import matplotlib.pyplot as plt
import os


class wdSim(gym.Env):  # gym实际上是一个RL的架子，模拟是在pybullet中模拟，然后在二者之间转化observation and action
    # gym -> ob,action -> position, angle, velocity, acceleration -> pybullet,
    # pybullet do moving -> position, angle,,,-> ob, action -> gym
    def __init__(self):
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 2}
        self._max_episode_steps = 50
        # 定义动作空间
        self.r_thigh_high = 1  # 0.5
        self.r_shin_high = 1  # 0.5
        self.l_thigh_high = 1  # 0.5
        self.l_shin_high = 1  # 0.5
        high = np.array([self.r_thigh_high, self.r_shin_high, self.l_thigh_high, self.l_shin_high], dtype=np.float32)
        self.action_space = spaces.Box(-high, high, dtype=np.float32)
        # 仿真关节  实物关节
        #    0         1
        #    1         3
        #    2         2
        #    3         4
        # 定义状态空间   机器人移动距离和方向
        self.observation_space = spaces.Box(
            low=np.array([-1 * self.r_thigh_high,
                          -1 * self.r_shin_high,
                          -1 * self.l_thigh_high,
                          -1 * self.l_thigh_high], dtype=np.float32),
            high=np.array([self.r_thigh_high,
                           self.r_shin_high,
                           self.l_thigh_high,
                           self.l_thigh_high], dtype=np.float32)
        )
        # 连接引擎
        self.physics_client_id = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 用到pybullet资源时要加这个
        # 起点
        self.start_point = 0.27
        # 计数器
        self.step_num = 0
        self.run_time = 0.336
        self.begin_time = datetime.now()
        self.end_time = datetime.now()
        self.angle_queue = []
        self.angle_queue_i = 0
        self.position_queue = []
        self.position_queue_i = 0

        self.reset()

    # 开始计时
    def reset_time(self):
        self.begin_time = datetime.now()  # 获得当前时间
        self.end_time = datetime.now()  # 获得当前时间

    def motor_velocity_control(self, q, _q):
        # _q 参考角度,q 当前角度, 运行时间
        q_error = _q - q
        q_speed = q_error / self.run_time
        return q_speed

    def reset(self):  # 重启环境
        p.resetSimulation(physicsClientId=self.physics_client_id)
        p.setGravity(0, 0, -9.8)
        cubeStartOrientation = p.getQuaternionFromEuler([1.57, 0, 0])
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.physics_client_id)  # 之前加了pybullet.data的绝对路径，所以这里可以直接用相对路径
        self.robot = p.loadURDF("data/walkingDroid/walkingDroid.urdf", [0., 0, 0.23], cubeStartOrientation,
                                physicsClientId=self.physics_client_id)  # python的相对路径就是当前项目下，所以这里不用改

        return self.__get_observation()

    def step(self, action):  # 前进一步，在pybullet中前进，根据pybullet的结果转化为obersvation，并返回
        # self.render()
        obs_before = self.__get_observation()
        location_before, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physics_client_id)
        self.__apply_action(action)
        self.step_num += 1
        state = self.__get_observation()

        base = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physics_client_id)
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
        location, _ = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.physics_client_id)
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
        self.rendered_img = plt.imshow(np.zeros((100, 100, 4)))
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                   nearVal=0.01, farVal=100)
        pos, ori = [list(l) for l in
                    p.getBasePositionAndOrientation(self.robot, self.physics_client_id)]
        pos[2] = 0.2

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        frame = p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
        frame = np.reshape(frame, (100, 100, 4))
        rgb_array = np.array(frame, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (100, 100, 4))
        rgb_array = rgb_array[:, :, :3]
        if mode == "rgb_array":
            return rgb_array
        elif mode == "human":
            self.rendered_img.set_data(frame)
            #plt.draw()
            #plt.pause(.00001)
        else:
            return None

    def close(self):
        if self.physics_client_id >= 0:
            p.disconnect(self.physics_client_id)
        self.physics_client_id = -1

    def __apply_action(self, action):  # 告诉pybullet如何执行gym的action
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        r_thigh, r_shin, l_thigh, l_shin = action
        # 根据运动时间计算动作分段数量
        motion_part_num = motor.get_motion_part_num(self.run_time * 1000)
        # 获取当前关节值
        JointStates = p.getJointStates(self.robot, [0, 1, 2, 3])
        JointStates = np.array([JointStates[0][0], JointStates[1][0], JointStates[2][0], JointStates[3][0]])
        # 动作分段
        motion_part_step = motor.split_motion(JointStates, np.array([r_thigh, r_shin, l_thigh, l_shin]),
                                              motion_part_num)
        motion_part_step_targetPositions = JointStates + motion_part_step

        time_flag = 0

        while 1:

            if time_flag == 0:
                a = datetime.now()  # 获得当前时间
                b = datetime.now()  # 获取当前时间
                time_flag = 1
                # print(flag)
            if (b - a).microseconds / 1000 < 12:

                p.setJointMotorControlArray(
                    bodyUniqueId=self.robot,
                    jointIndices=[0, 1, 2, 3],
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[motion_part_step_targetPositions[0], motion_part_step_targetPositions[1],
                                     motion_part_step_targetPositions[2], motion_part_step_targetPositions[3]]
                )
                p.stepSimulation()
                b = datetime.now()  # 获取当前时间

            else:
                motion_part_num = motion_part_num - 1
                time_flag = 0
                if motion_part_num == 0:
                    break
                else:
                    motion_part_step_targetPositions = motion_part_step_targetPositions + motion_part_step


    def __get_observation(self):  # 告诉Env如何从pybullet中获得oberservation
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        # 获取关节值
        JointStates = p.getJointStates(self.robot, [0, 1, 2, 3], physicsClientId=self.physics_client_id)
        obs = [JointStates[0][0], JointStates[1][0], JointStates[2][0], JointStates[3][0]]

        # return np.array([distance, angle])
        return np.around(np.asarray(obs), 2)

