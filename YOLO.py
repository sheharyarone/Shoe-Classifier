import os
import subprocess

weights_path = "weights_50_epochs.pt"
image = "WhatsApp Image 2023-12-23 at 20.29.38 (1).jpeg"

def yolo(image):
    source_path = image
    detection_command = f'python yolov7-main/detect.py --weights "{weights_path}" --source "{source_path}" --save-cropped'
    os.system(detection_command)




yolo(image)
