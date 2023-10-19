from ultralytics import YOLO

model = YOLO(model="yolov8n.pt")

model.train(data="data.yaml", \
            epochs=100, \
            batch=32, \
            imgsz=512, \
            save=True, \
            save_period=5, \
            device="0,1", \
            project="layout")