import numpy as np
import pybullet as p
import time
import pybullet_data

DURATION = 10000
ALPHA = 300

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
print("data path: %s " % pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
gemId = p.loadURDF("duck_vhacd.urdf",
                   [2, 2, 1],  p.getQuaternionFromEuler([0, 0, 0]))
for i in range(DURATION):
    p.stepSimulation()
    time.sleep(1./240.)
    gemPos, gemOrn = p.getBasePositionAndOrientation(gemId)
    boxPos, boxOrn = p.getBasePositionAndOrientation(boxId)

    force = ALPHA * (np.array(gemPos) - np.array(boxPos))  # 力的大小和方向最好是经过计算的，不然只给一个值在坐标系中不够明确
    # 力的单位也存疑。重要的或许是如何通过设定轨道和完成时间来计算力，直接给出力的大小往往难以产生预期的行为
    p.applyExternalForce(objectUniqueId=boxId, linkIndex=-1,
                         forceObj=force, posObj=boxPos, flags=p.WORLD_FRAME)

    print('Applied force magnitude = {}'.format(force))
    print('Applied force vector = {}'.format(np.linalg.norm(force)))

p.disconnect()