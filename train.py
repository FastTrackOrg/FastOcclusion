from ultralytics import YOLO
import os

os.chdir("./test")

model = YOLO("yolov8n-seg.pt")
model.train(data="dataset.yml", epochs=200, workers=16, batch=16, imgsz=600, translate=0, scale=0, shear=0, fliplr=0, mosaic=0)
