from ultralytics import YOLO

model = YOLO(model="LayoutNet.pt")

model.val(data="data.yaml", \
          epochs=100, \
          batch=1, \
          imgsz=1024, \
          device="0",\
          project="layout", \
          save_json=True, \
          max_det=100)
