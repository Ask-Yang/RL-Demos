import pybullet as p
from time import sleep
p.connect(p.GUI)
p.loadURDF("simplecar.urdf")
sleep(100)