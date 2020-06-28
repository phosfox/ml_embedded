from line_detection import Line_Detection
import sys
sys.path.append("~/jetcam/jetcam")
from jetcam.csi_camera import CSICamera
sys.path.append("~/jetbot/jetbot")
from jetbot import Robot
import torch
import time
import cv2
class Driver():

    def __init__(self, robot, camera, model):
        self.robot = robot
        self.camera = camera
        self.model = model
        self.speed = 0.1
        self.turn_speed = 0.1
        self.action = {"left": self.left, "right": self.right, "straight": self.straight}

    def drive(self):
        while True:
            raw_image = self.camera.read()
            prep_image = self.model.preprocess(raw_image)
            direction = self.model.predict(prep_image)
            self.action.get(direction)()
            print(direction)
            #cv2.imshow(direction, raw_image)

    def accelerate(self):
        self.speed += 0.05
        self.robot.forward(self.speed)

    def decelerate(self):
        self.speed -= 0.05
        self.robot.backward(self.speed)


    def stop(self):
        self.robot.stop()


    def left(self):
        self.stop()
        self.robot.set_motors(0, self.turn_speed)


    def right(self):
        self.stop()
        self.robot.set_motors(self.turn_speed, 0)


    def straight(self):
        self.robot.set_motors(self.speed, self.speed)


def main():
    robot = Robot()
    camera = CSICamera(width=224, height=224, capture_width=224, capture_height=224, capture_fps=1)
    line_detection_model = Line_Detection("/home/jetbot/Notebooks/models/resnet18_norm_0.90acc.pth")
    driver = Driver(robot, camera, line_detection_model)
    driver.drive() #loop

if __name__ == "__main__":
    main()
