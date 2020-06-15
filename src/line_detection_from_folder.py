import sys
sys.path.append("~/jetcam/jetcam")

import time
import cv2
import torch 
import glob
from jetcam.csi_camera import CSICamera

class Line_Detection:

    def predict(self, image_tensor):
        prediction = self.model(image_tensor)
        preds, indices = torch.topk(prediction, 1)
        print("Indices:", indices)
        print("Preds:", prediction)
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
    image_paths = glob.glob("/home/jetbot/Pictures/*.jpg")

    path = "./models/resnet34_97acc.pth" 
    ld = Line_Detection(path)
    #camera = CSICamera(width=224, height=224, capture_width=224, capture_height=224, capture_fps=21)
    for idx, img in enumerate(image_paths):
        image = cv2.imread(img)
        #image = camera.read()
        tensor = ld.preprocess(image)
        pred_class = ld.predict(tensor)
        print(img, idx, pred_class)
        

if __name__=="__main__":
    main()
