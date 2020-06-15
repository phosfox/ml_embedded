import sys
sys.path.append("~/jetcam/jetcam")

import time
import cv2
import torch 

from jetcam.csi_camera import CSICamera

class Line_Detection:

    def predict(self, image_tensor):
        prediction = self.model(image_tensor)
        __, indices = torch.topk(prediction, 1)
        pred_class = self.class_names[indices.flatten().item()]
        return pred_class

    def get_device(self):
        return self.device

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(image)
        tensor = tensor.to(self.device)
        tensor = tensor[None]
        tensor = tensor.permute(0,3,1,2)
        tensor = tensor.float()
        return tensor

    def __init__(self, path):
        self.class_names = ["left", "right", "straight"]
        use_cuda = not torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = torch.load(path)
        self.model.to(self.device)
        self.model.eval()


def main():

    path = "./models/resnet18_95acc.pth" 
    ld = Line_Detection(path)
    camera = CSICamera(width=224, height=224, capture_width=224, capture_height=224, capture_fps=21)
    while True:
        image = camera.read()
        tensor = ld.preprocess(image)
        pred_class = ld.predict(tensor)
        print(pred_class)
        break
        

if __name__=="__main__":
    main()
