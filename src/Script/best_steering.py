#!/usr/bin/env python3

import sys
sys.path.append("~/jetbot/jetbot")
import signal
import time
from jetbot import Robot
import cv2
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image
import numpy as np
from jetbot import Camera, bgr8_to_jpeg

class Line_Detection:

    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device).half()
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def interruptHandler(self, signal, frame):
        sys.exit(0)

    def execute(self, change):
        steering = 0.00
        image = change['new']

        xy = self.model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
        x = xy[0]
        y = (0.5 - xy[1]) / 2.0
        self.angle = np.arctan2(x, y)
        pid = self.angle * self.steering_gain + (self.angle - self.angle_last) * self.steering_dgain
        self.angle_last = self.angle

        steering = pid + steering
        print(f"arctan({x}, {y}) = {self.angle}")
        print("speed:" ,self.speed)
        self.robot.left_motor.value = max(min(self.speed + steering, 1.0), 0.0)
        self.robot.right_motor.value = max(min(self.speed - steering, 1.0), 0.0)

    def __init__(self, path):
        signal.signal(signal.SIGINT, self.interruptHandler)
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(512, 2)
        self.model.load_state_dict(torch.load(path))
        self.device = torch.device('cuda')
        self.model = self.model.to(self.device)
        self.model = self.model.eval().half()
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
        self.robot = Robot()
        self.camera = Camera(width=224, height=224, capture_width=1280, capture_height=720)
        self.steering_gain = 0.025
        self.steering_dgain = 0.07
        self.speed = 0.20
        self.old_speed = self.speed
        self.angle = 0.0
        self.angle_last = 0.0
        self.execute({'new': self.camera.value})
        self.camera.observe(self.execute, names='value')


def main():
    path = '/home/jetbot/Notebooks/models/best_steering_model_xy.pth'
    ld = Line_Detection(path)

if __name__ == "__main__":
    main()
