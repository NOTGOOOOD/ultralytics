'''
Author: xuexufeng
Date: 2025-02-11 11:51:21
LastEditors: xuexufeng
LastEditTime: 2025-06-12 19:03:16
FilePath: /ultralytics/val.py
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

# # =================ONNX===============================
# model = YOLO("yolov5.yaml")
# model.export(imgsz=224, format="onnx")
# # =================test===============================
# model = YOLO(model="yolo11x.pt",task="detect")
# result = model("/mnt/data1/Dataset/CMS/ped_veh_det_coco_bddk_rod_train/images/train/coco_yolo_new_3_000000198751.jpg")
# result[0].save("test.jpg")

# Load a model
model = YOLO("yolov11-custom.yaml")  # load a pretrained model (recommended for training)
model.load("runs/detect/train10/weights/best.pt")


# # Train the model with 2 GPUs
# results = model.train(data="/home/xuexufeng/project/ultralytics/data.yaml", epochs=100, imgsz=480, device=[0, 1, 2, 3], batch=128)
model.val(data="/home/xuexufeng/project/ultralytics/val.yaml", batch=128, split="val", project="person_vehicle", device=[-1], imgsz=640)

