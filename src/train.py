from ultralytics import YOLO
import os

os.chdir("./test")

model = YOLO("yolov8n-seg.pt")
model.train(data="dataset.yml", project="Result", name="Train", epochs=50, workers=8, batch=32, imgsz=608, translate=0.5, scale=0.8, shear=0, fliplr=0.5, mosaic=0)

metrics = model.val(data="dataset.yml", imgsz=608, project="Result", name="Test", split="test", iou=0, max_det=4, conf=0)
print(metrics.seg.map, metrics.seg.map50,metrics.seg.map75,metrics.seg.maps)
