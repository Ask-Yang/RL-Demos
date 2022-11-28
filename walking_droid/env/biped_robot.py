import pybullet as p
import numpy as np
import os
import motor
from datetime import datetime


class BipedRobot:
    def __init__(self, client):
        self.client = client
        filename = "../data/walkingDroid/walkingDroid.urdf"
        cube_start_orientation = p.getQuaternionFromEuler([1.57, 0, 0])
        self.biped_robot = p.loadURDF(filename,
                                      basePosition=[0, 0, 0.1],
                                      baseOrientation=cube_start_orientation,
                                      physicsClientId=self.client)
        self.biped_joint = [0, 1, 2, 3]

    def get_ids(self):
        return self.client, self.biped_robot

    def apply_action(self, action):
        assert isinstance(action, list) or isinstance(action, np.ndarray)
        r_thigh, r_shin, l_thigh, l_shin = action
        # 根据运动时间计算动作分段数量
        motion_part_num = motor.get_motion_part_num(0.336 * 1000)
        # 获取当前关节值
        joint_states = p.getJointStates(self.biped_robot, self.biped_joint)
        joint_states = np.array([joint_states[0][0], joint_states[1][0], joint_states[2][0], joint_states[3][0]])
        # 动作分段
        motion_part_step = motor.split_motion(joint_states, np.array([r_thigh, r_shin, l_thigh, l_shin]),
                                              motion_part_num)
        motion_part_step_target_positions = joint_states + motion_part_step
        # 这个动作分段感觉不是很合理
        time_flag = 0
        a = datetime.now()  # 获得当前时间
        b = datetime.now()  # 获取当前时间
        while 1:
            if time_flag == 0:
                a = datetime.now()  # 获得当前时间
                b = datetime.now()  # 获取当前时间
                time_flag = 1
            if (b - a).microseconds / 1000 < 12:
                p.setJointMotorControlArray(
                    bodyUniqueId=self.biped_robot,
                    jointIndices=self.biped_joint,
                    controlMode=p.POSITION_CONTROL,
                    targetPositions=[motion_part_step_target_positions[0], motion_part_step_target_positions[1],
                                     motion_part_step_target_positions[2], motion_part_step_target_positions[3]]
                )
                p.stepSimulation()  # 这个要尽快改掉
                b = datetime.now()  # 获取当前时间
            else:
                motion_part_num = motion_part_num - 1
                time_flag = 0
                if motion_part_num == 0:
                    break
                else:
                    motion_part_step_target_positions = motion_part_step_target_positions + motion_part_step

    def get_observation(self):
        if not hasattr(self, "robot"):
            assert Exception("robot hasn't been loaded in!")
        # 获取关节值
        JointStates = p.getJointStates(self.biped_robot, [0, 1, 2, 3], physicsClientId=self.client)
        obs = [JointStates[0][0], JointStates[1][0], JointStates[2][0], JointStates[3][0]]
        # return np.array([distance, angle])
        return np.around(np.asarray(obs), 2)
