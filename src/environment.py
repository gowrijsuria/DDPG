import pybullet as p
import pybullet_data

import torch
import torchvision

import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

class CarEnvironment:
    def __init__(self, args, device):
        self.client = p.connect(p.DIRECT)
        self.car = None
        self.done = False
        self.args = args
        self.device = device
        self.targetVel = args.targetVel
        self.maxForce = args.maxForce
        self.screen_height = args.screen_height
        self.screen_width = args.screen_width
    
    def reset(self):
        p.resetSimulation()     
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        stadiumID = p.loadSDF("stadium_no_collision.sdf")
        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0) 
        self.car = p.loadURDF("husky/husky.urdf", self.args.start_x, self.args.start_y, self.args.start_z)
        # Destination 
        visualShapeId4 = p.loadURDF("soccerball.urdf", [self.args.final_x, self.args.final_y, self.args.final_z], useFixedBase=1)
        # Obstacles ID -  [5,6,7,8,9,10,11,12,13,14]
        visualShapeId5 = p.loadURDF("sphere2red.urdf", [4, 8 , 0], useFixedBase=1)
        visualShapeId6 = p.loadURDF("sphere2red.urdf", [-8, 5 , 0], useFixedBase=1)
        visualShapeId7 = p.loadURDF("sphere2red.urdf", [-10, -2 , 0], useFixedBase=1)
        visualShapeId8 = p.loadURDF("sphere2red.urdf", [-12, 0 , 0], useFixedBase=1)
        visualShapeId9 = p.loadURDF("sphere2red.urdf", [10, -9, 0], useFixedBase=1)
        visualShapeId10 = p.loadURDF("sphere2red.urdf", [14, 3, 0], useFixedBase=1)
        visualShapeId11 = p.loadURDF("sphere2red.urdf", [10, 0, 0], useFixedBase=1)
        visualShapeId12 = p.loadURDF("sphere2red.urdf", [0, -12, 0], useFixedBase=1)
        visualShapeId12 = p.loadURDF("sphere2.urdf", [-15, -2, 0], useFixedBase=1)
        visualShapeId13 = p.loadURDF("sphere2.urdf", [-8, -5, 0], useFixedBase=1)
        visualShapeId14 = p.loadURDF("sphere2.urdf", [-10, -9, 0], useFixedBase=1)
        visualShapeId15 = p.loadURDF("sphere2red.urdf", [3, 0, 0], useFixedBase=1)
        visualShapeId16 = p.loadURDF("sphere2red.urdf", [0, -2, 0], useFixedBase=1)

        self.carPos, self.carOrn = p.getBasePositionAndOrientation(self.car)
        self.done = False

    def render(self):
        proj_matrix = p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.7, farVal=1000)
        pos, ori = [list(l) for l in p.getBasePositionAndOrientation(self.car)] 
        pos[2] = 0.5

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)

        # Display image
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(self.screen_width, self.screen_height, view_matrix, proj_matrix)
        frame = rgbImg
        frame = np.reshape(frame, (self.screen_width, self.screen_height, 4))

    def ImageAtCurrentPosition(self):
        proj_matrix = p.computeProjectionMatrixFOV(fov=100, aspect=1, nearVal=0.7, farVal=1000)
        pos, ori = [list(l) for l in p.getBasePositionAndOrientation(self.car)] 
        pos[2] = 0.5

        # Rotate camera direction
        rot_mat = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        camera_vec = np.matmul(rot_mat, [1, 0, 0])
        up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
        view_matrix = p.computeViewMatrix(pos, pos + camera_vec, up_vec)
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(self.screen_width, self.screen_height, view_matrix, proj_matrix)

        areaOfObstacle = 0
        for i in range(segImg.shape[0]):
            for j in range(segImg.shape[1]):
                pixel = segImg[i][j]
                if (pixel >= 0):
                    obUid = pixel & ((1 << 24) - 1)
                    # checking if pixel belongs to obstacle ID
                    if obUid not in [0,1,2,3,4]:
                        areaOfObstacle+=1

        rgbImg = rgbImg[:,:,:3] #convert rgba to rgb
        rgbImgTensor = torchvision.transforms.ToTensor()(rgbImg) #convert HWC to CHW
        rgbImgTensor = rgbImgTensor.unsqueeze(0).to(self.device) #add batch dimension
        areaOfObstacle = torch.tensor([areaOfObstacle]).float()
        areaOfObstacle = areaOfObstacle.unsqueeze(0).to(self.device)
        # return rgbImgTensor, rgbImg, areaOfObstacle
        return {'image': rgbImgTensor, 'area_obstacle': areaOfObstacle}, rgbImg

    def outsideBoundary(self):
        stadium_halflen = 105 * 0.25  # FOOBALL_FIELD_HALFLEN
        stadium_halfwidth = 50 * 0.25  # FOOBALL_FIELD_HALFWID

        # Done by running off boundaries
        self.carPos, self.carOrn = p.getBasePositionAndOrientation(self.car)
        if (self.carPos[0] >= stadium_halflen or self.carPos[0] <= -stadium_halflen or
                self.carPos[1] >= stadium_halfwidth or self.carPos[1] <= -stadium_halfwidth):
            print("Fell off boundary")    
            self.done = True
        return self.done

    def reachedGoal(self):
        currentPosition, currentOri = [list(l) for l in p.getBasePositionAndOrientation(self.car)] 
        DistToGoal = math.sqrt(((currentPosition[0] - self.args.final_x) ** 2 +
                                  (currentPosition[1] - self.args.final_y) ** 2))
        
        if DistToGoal < self.args.threshold_dist:
            print("Reached Goal")
            self.done = True
        return self.done

    def moveForward(self):
        for joint in range(2, 6):
            p.setJointMotorControl(self.car, joint, p.VELOCITY_CONTROL, self.targetVel, self.maxForce)
        for step in range(300):
            p.stepSimulation()

    def change_direction(self, steering):
        inactive_wheels = [4, 5]
        for wheel in inactive_wheels:
            p.setJointMotorControl2(self.car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        frontleftWheelVelocity = 0
        frontrightWheelVelocity = 0
        
        frontleftWheelVelocity += (-steering*self.args.velocity_factor)*self.targetVel
        frontrightWheelVelocity+= (steering*self.args.velocity_factor)*self.targetVel

        p.setJointMotorControl(self.car, 2, p.VELOCITY_CONTROL, frontleftWheelVelocity, self.maxForce)
        p.setJointMotorControl(self.car, 3, p.VELOCITY_CONTROL, frontrightWheelVelocity, self.maxForce)
        for step in range(300):
            p.stepSimulation()
        self.moveForward()

    def checkDone(self):
        reward = 0
        if self.outsideBoundary():
            self.done = True
            reward = self.args.reward_outside_boundary
        elif self.reachedGoal(): 
            self.done = True
            reward = self.args.reward_goal
        else:
            self.done = False
        return reward

    def step(self, action):
        self.change_direction(action)
        reward = self.checkDone()
        if self.done:
            state, rgbImage = self.ImageAtCurrentPosition()
            # no next state since episode is done
            next_state = {'image': torch.zeros(state['image'].shape).to(self.device), 'area_obstacle': torch.tensor([[0.0]]).float().to(self.device)}
        else:
            next_state, rgbImage = self.ImageAtCurrentPosition()

        areaOfObstacle = next_state['area_obstacle']
        if areaOfObstacle != 0:
            reward += self.args.reward_collision*areaOfObstacle
        return self.done, next_state, reward, rgbImage

    def close(self):
        time.sleep(10)
        p.disconnect()
