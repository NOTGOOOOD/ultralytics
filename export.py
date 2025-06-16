import sys
from ultralytics import YOLO

onnx_path = sys.argv[1]
model = YOLO("yolov11-custom.yaml")
model.load(onnx_path)
model.export(imgsz=640, format="onnx")