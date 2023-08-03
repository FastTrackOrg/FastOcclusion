from ultralytics import YOLO
import os

os.chdir("./test")

model = YOLO("yolov8n-seg.pt")
model.train(data="dataset.yml", project="Result", name="Train", epochs=140, workers=16, batch=8, imgsz=608, translate=0, scale=0, shear=0, fliplr=0, mosaic=0)

metrics = model.val(data="dataset.yml", project="Result", name="Test", split="test")
metrics.box.map
metrics.box.map50
metrics.box.map75
metrics.box.maps
