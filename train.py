'''
Author: xuexufeng
Date: 2025-02-11 11:51:21
LastEditors: xuexufeng
LastEditTime: 2025-02-18 20:39:05
FilePath: /ultralytics/train.py
Description: 

Copyright (c) 2025 by xuexufeng@pointspread.tech, All Rights Reserved. 
'''
import os
import sys

sys.path.append(os.getcwd())
from ultralytics import YOLO

# model = YOLO("yolov10n.yaml")
# results = model.train(data="gesture.yaml",
#                       epochs=100,
#                       imgsz=224,
#                       device=[6],
#                       batch=1024,
#                       project="gesture",
#                       save_period=1)

# # =================VAL=========================================
# model = YOLO("yolov5n.yaml")
# model.val(data="ultralytics/cfg/datasets/gesture.yaml", batch=512, split="test", project="gesture")

# =================ONNX===============================
model = YOLO("yolov5.yaml")
model.export(imgsz=224, format="onnx")
# # =================test===============================
# model = YOLO("yolov10n.yaml")
# result = model("/mnt/data1/Dataset/Hagrid-yolo/images/train/point/00026798-047f-41a4-b8a5-d6abda5a8826_r.jpg")
# result[0].save("test.jpg")
