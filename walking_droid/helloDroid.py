import pybullet as p
import pybullet_data
import time 
from pprint import pprint

# 连接物理引擎
use_gui = True
if use_gui:
    serve_id = p.connect(p.GUI)
else:
    serve_id = p.connect(p.DIRECT)
cubeStartPos = [0, 0, 0.3]
#cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([1.57, 0, 0])
# 添加资源路径
p.setAdditionalSearchPath(pybullet_data.getDataPath())
# 配置渲染机制
#p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# 设置重力，加载模型
p.setGravity(0, 0, -9.8)
_ = p.loadURDF("plane.urdf", useMaximalCoordinates=True)
robot_id = p.loadURDF("walkingDroid/walkingDroid.urdf",cubeStartPos,cubeStartOrientation)

# 预备工作结束，重新开启渲染
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
# 关闭实时模拟步
#p.setRealTimeSimulation(0)


mode =  p.POSITION_CONTROL
# 获取轮子的关节索引
# 0   1 
# 1   3
# 2   2
# 3   4
p.setJointMotorControlArray(robot_id, [0,1,2,3],
 	mode,[0,0,0,0])
#JointState=p.getJointState(robot_id,12)
#print("----------")
#print(JointState)
#关节信息
#JointInfo=p.getJointInfo(robot_id,12)
#print("----------")
'''
joint_num = p.getNumJoints(robot_id)
print("gp的节点数量为：", joint_num)

print("gp的信息：")
for joint_index in range(joint_num):
    info_tuple = p.getJointInfo(robot_id, joint_index)
    print(f"关节序号：{info_tuple[0]}\n\
            关节名称：{info_tuple[1]}\n\
            关节类型：{info_tuple[2]}\n\
            机器人第一个位置的变量索引：{info_tuple[3]}\n\
            机器人第一个速度的变量索引：{info_tuple[4]}\n\
            保留参数：{info_tuple[5]}\n\
            关节的阻尼大小：{info_tuple[6]}\n\
            关节的摩擦系数：{info_tuple[7]}\n\
            slider和revolute(hinge)类型的位移最小值：{info_tuple[8]}\n\
            slider和revolute(hinge)类型的位移最大值：{info_tuple[9]}\n\
            关节驱动的最大值：{info_tuple[10]}\n\
            关节的最大速度：{info_tuple[11]}\n\
            节点名称：{info_tuple[12]}\n\
            局部框架中的关节轴系：{info_tuple[13]}\n\
            父节点frame的关节位置：{info_tuple[14]}\n\
            父节点frame的关节方向：{info_tuple[15]}\n\
            父节点的索引，若是基座返回-1：{info_tuple[16]}\n\n")
'''
p.stepSimulation()
target_v = 10                   # 电机达到的预定角速度（rad/s）
max_force = 10                  # 电机能够提供的力，这个值决定了机器人运动时的加速度，太快会翻车，单位N
#开始录像
#log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "log/robotmove.mp4")

while 1 :

    p.stepSimulation()
    JointState=p.getJointState(robot_id,0)
    #print("----------")
    #if (JointState[0] < 0.505 and JointState[0]>0.495 ):
        #print(JointState[0])
    #else:
       # p.stepSimulation()
    location, _ = p.getBasePositionAndOrientation(robot_id)
    print("location=",location[1])
    base=p.getBasePositionAndOrientation(robot_id)
    angle_tuple=p.getEulerFromQuaternion(base[1])#-1.4   0.8#角度四元组转为欧拉角
    angle_x1=angle_tuple[0]#前后倒
    angle_x2=angle_tuple[1]#左右倒
    angle_x3=angle_tuple[2]#左右倒
    #print("angle_x1 = " ,angle_x1-1.57)

    #print("angle_x2 = " ,angle_x2)
    #print("position=",p.getJointInfo(robot_id, 1)[3]) 
    '''
    p.setJointMotorControlArray(
        bodyUniqueId=robot_id,
        jointIndices=wheel_joints_indexes,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocities=[target_v for _ in wheel_joints_indexes],
        forces=[max_force for _ in wheel_joints_indexes]
    )
    '''
    #p.setJointMotorControl2(robot_id, 0,controlMode=p.TORQUE_CONTROL, force=5)
    ##location, _ = p.getBasePositionAndOrientation(robot_id)
    #p.resetDebugVisualizerCamera(
    #    cameraDistance=0.35,
    #    cameraYaw=40,
    #    cameraPitch=-10,
    #    cameraTargetPosition=location
    #)
   
    #time.sleep(1 / 240)         # 模拟器一秒模拟迭代240步
#p.stopStateLogging(log_id)#录像结束
# 断开连接
p.disconnect(serve_id)
