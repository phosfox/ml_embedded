#!/usr/bin/env python3
import sys
sys.path.append("~/jetcam/jetcam")

import cv2

from jetcam.csi_camera import CSICamera

def main():
    camera = CSICamera(width=244, height=244, capture_width=244, capture_height=244, capture_fps=21)
    for i in range(0,100):
        image = camera.read()
        base_string = f"images/drive_{i}.jpg"
        cv2.imwrite(base_string, image, [cv2.IMWRITE_JPEG_QUALITY, i])

if __name__ == "__main__":
    main()
