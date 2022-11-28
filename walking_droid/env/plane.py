import pybullet as p
import numpy as np
import os


class Plane:
    def __init__(self, client):
        self.client = client
        filename = "plane.urdf"
        self.plane = p.loadURDF(filename,
                                basePosition=[0, 0, 0],
                                physicsClientId=self.client)