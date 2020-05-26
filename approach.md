# This document is used to record our proceeding

## Detecting lines through algorithms
- CV2
- Canny
- Mask
- Hough Transformation

## Design of Scripts
### Design of script for driving the car with the controller
This script is called jstest.py and can be found in the /src/ folder 


## Camera
If the camera seems to be red download the [.isp](https://www.waveshare.com/wiki/IMX219-160_Camera) file.

Commands

`wget https://www.waveshare.com/w/upload/e/eb/Camera_overrides.tar.gz
tar zxvf Camera_overrides.tar.gz 
sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp`
---
Don't use .jpg in order to improve the picture quality. (jpg is a lossy file format)