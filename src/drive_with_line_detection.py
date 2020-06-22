from line_detection import Line_Detection
import sys
sys.path.append("~/jetcam/jetcam")
from jetcam.csi_camera import CSICamera
sys.path.append("~/jetbot/jetbot")
from jetbot import Robot
import torch

class Driver():

    def __init__(self, robot, camera, model):
        self.robot = robot
        self.camera = camera
        self.model = model
        self.speed = 0.1
        self.action = {"left": self.left, "right": self.right, "straight": self.straight}

    def drive(self):
        while True:
            image = self.camera.read()
            image = self.model.preprocess(image)
            direction = self.model.predict(image)
            self.action.get(direction)()


    def accelerate(self):
        self.speed += 0.05
        self.robot.forward(self.speed)


    def decelerate(self):
        self.speed -= 0.05
        self.robot.backward(self.speed)


    def stop(self):
        self.robot.stop()


    def left(self):
        self.robot.set_motors(self.speed/3, self.speed)


    def right(self):
        self.robot.set_motors(self.speed, self.speed/3)


    def straight(self):
        self.robot.set_motors(self.speed, self.speed)


def main():
    robot = Robot()
    camera = CSICamera(width=224, height=224, capture_width=224, capture_height=224, capture_fps=21)
    line_detection_model = Line_Detection("./models/resnet18_95acc.pth")
    driver = Driver(robot, camera, line_detection_model)
    driver.drive() #loop

if __name__ == "__main__":
    main()
