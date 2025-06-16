'''
Author: xuexufeng
Date: 2025-02-11 11:51:21
LastEditors: xuexufeng
LastEditTime: 2025-06-16 15:56:36
FilePath: /ultralytics/predict.py
Description: 

Copyright (c) 2025 by xuexufeng@pointspread.tech, All Rights Reserved. 
'''
import os
import sys

sys.path.append(os.getcwd())
from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov11-custom.yaml")  # load a pretrained model (recommended for training)
# model.load("runs/detect/finetune01/weights/best.pt")

## ONNX推理
model = YOLO("/home/xuexufeng/project/ultralytics/yolov11-custom.onnx")
result = model("20250604_105513_1976.png", conf=0.25, )
result[0].save("test1.jpg")
